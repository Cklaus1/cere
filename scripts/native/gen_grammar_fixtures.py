"""The M2.8 grammar oracle (tasks/m28-spec.md C4, D5, D7): the grammar sidecar (`tokenizer_sidecar.py --role grammar`)
against SGLang's own request mapping and `XGrammarGrammarBackend`, run in the pinned image (CPU, no network).

Generate (writes tests/fixtures/native/grammar_cases.json and grammar_keywords.json):

    docker run --rm --network none --user UID:GID -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= \
      --entrypoint python3 -v REPO:/src:ro -v MODEL:/model:ro -v REPO/tests/fixtures/native:/out IMAGE \
      /src/scripts/native/gen_grammar_fixtures.py --model /model --out /out --image IMAGE

Self-check (reads the committed fixtures, writes nothing):

    docker run --rm --network none --user UID:GID -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES= \
      --entrypoint python3 -v REPO:/src:ro -v MODEL:/model:ro IMAGE \
      /src/scripts/native/gen_grammar_fixtures.py --model /model --selfcheck

MODEL is the prepared native artifact, IMAGE `lovedheart/qwen38-flash-next@sha256:<digest>` (the deployment lock's
`backend_build_sha256`). Both modes run every case; generate writes the documents only when every parity check
holds, the self-check prints `GRAMMAR FIXTURES SELFCHECK PASS ...` or `GRAMMAR FIXTURES SELFCHECK FAIL` with case
names (never text) and exits 0/1.

Parity (`grammar_cases.json`, cases G1-G9 of the spec's Tier 2 list plus a 40-schema keyword matrix K01-K40): the
request goes to the real sidecar process as `grammar_open` raw fields and, as the same body, through SGLang's
`ChatCompletionRequest`, `OpenAIServingChat._validate_request`/`_process_messages` (bench flags: tool-call parser
qwen3_coder, reasoning parser qwen3, enable_thinking false) and `to_sampling_params`; the key is taken in
`grammar_manager.py:144-151` order. The reference grammar is `create_grammar_backend` (xgrammar, wrapped in
`ReasonerGrammarBackend` as the scheduler builds it) over SGLang's tokenizer, the model vocabulary and
`ModelConfig.hf_eos_token_id`. `key_sha256` must match; then both walk one oracle token path built from the grammar:
a conforming instance kept in this file, consumed greedily by the longest allowed token whose xgrammar-decoded bytes
prefix the remaining bytes, then the lowest allowed stop token. Every mask along the path (after open and after each
non-final step) is byte-equal between the sidecar and the reference; `accepted`/`terminated` agree. A case both
refuse records the sidecar's error type and param. G8 is G2 with `"stream": true` in the SGLang body.

Thinking (case T1, tasks/m28-spec.md D15, ledger CR20): G2's schema opened with `"enable_thinking": true` against the
reference built for the SGLang body with `chat_template_kwargs.enable_thinking` (its `require_reasoning` must hold,
so the backend returns a `ReasonerGrammarObject` in reasoning). Both step a synthetic reasoning sequence (a fixed
text's encoding, the grammar's first instance token, the lowest model EOS, then the backend's `think_end_ids`), then
G2's oracle path; every mask is byte-equal and every step accepted with equal `terminated`. The record holds the
sequence and path digests and lengths, the all-ones mask count, and the mask digests; the think-end mask and the path
must equal G2's open mask and path.

Enforcement (`grammar_keywords.json`, D5): for every candidate keyword and every xgrammar `format` value, each
variant schema is compiled as the backend compiles JSON (`any_whitespace=True`, strict mode) and every conforming
instance must be accepted and every violating one rejected by `xgrammar.testing._is_grammar_accept_string`; stderr and
Python warnings during the compile must stay empty. A keyword or format passes only if all its variants pass;
`keywords`/`formats` are the passing sets, `annotations` the annotation-only keywords that compile and leave the
rest of the schema enforced.

Number ranges (`grammar_keywords.json` `number_range_variants`, tasks/m28-spec.md B7 [REVISED: owner 2026-10-18
number bounds]): each `type: "number"` schema bounded by an inclusive `minimum` and `maximum` is opened in the real
sidecar process as a json_schema `response_format`; an instance is accepted iff the sidecar accepts every token of its
encoding, then the lowest model EOS, and terminates. Every conforming instance must be accepted and every violating
one rejected (else generate fails). `stock_pass` records whether xgrammar's own compile of the schema enforces it.

Files hold schemas, request fields, ids counts and SHA-256 digests only (instances and paths as digests); all strings
are synthetic. Header: image_digest, sglang_version, xgrammar_version, torch_version, generator,
source_tokenizer_sha256, template_sha256.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/native"))

import gen_api_fixtures as api  # noqa: E402 - serving_chat, doc_text and the B4 tokenizer helpers
import orjson  # noqa: E402
import torch  # noqa: E402
import xgrammar  # noqa: E402
from sglang.srt.configs.model_config import ModelConfig  # noqa: E402
from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject, create_grammar_backend  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest  # noqa: E402
from sglang.srt.parser.reasoning_parser import ReasoningParser  # noqa: E402
from xgrammar.testing import _is_grammar_accept_string  # noqa: E402

GENERATOR = "scripts/native/gen_grammar_fixtures.py"
SIDECAR = ROOT / "scripts/native/tokenizer_sidecar.py"
HEADER = struct.Struct("<I")
VOCAB_SIZE = 248320
MASK_WORDS = VOCAB_SIZE // 32
DIALECT = "json-schema-2020-12/xgrammar-0.2.1"
CASES_FILE = "grammar_cases.json"
KEYWORDS_FILE = "grammar_keywords.json"

# ------------------------------------------------------------------------------------------------------------------
# Cases.


def rf_schema(schema) -> str:
    return json.dumps({"type": "json_schema", "json_schema": {"name": "s", "schema": schema}})


def tool(name: str, properties: dict, required: list, strict: bool = False) -> dict:
    fn = {"name": name, "parameters": {"type": "object", "properties": properties, "required": required}}
    if strict:
        fn["strict"] = True
    return {"type": "function", "function": fn}


def call(name: str, params: list) -> str:
    """One qwen3_coder XML tool call (`builtin_structural_tag.py` qwen_3_coder)."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


F_TOOL = tool("f", {"a": {"type": "string"}, "n": {"type": "integer"}}, ["a", "n"])
G_TOOL = tool("g", {"q": {"type": "string"}}, ["q"])
G_STRICT = tool("g", {"q": {"type": "string"}}, ["q"], strict=True)

G2_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "kind": {"enum": ["a", "b"]},
    "n": {"type": "integer", "minimum": 0, "maximum": 9},
    "xs": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3}},
    "required": ["name", "kind", "n", "xs"], "additionalProperties": False}
G2_INSTANCE = json.dumps({"name": "x", "kind": "a", "n": 3, "xs": [1, 2]})
G3_SCHEMA = {"$defs": {"pt": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}},
             "type": "object", "properties": {"p": {"$ref": "#/$defs/pt"},
                                               "v": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
             "required": ["p", "v"]}
G5_EBNF = 'root ::= expr\nexpr ::= term ("+" term)*\nterm ::= num ("*" num)*\nnum ::= [0-9]+'


def g_cases() -> list:
    """(name, sidecar fields, extra SGLang body fields, conforming instance)."""
    two = json.dumps([F_TOOL, G_TOOL])
    return [
        ("G1", {"response_format": '{"type": "json_object"}'}, {}, json.dumps({"a": 1})),
        ("G2", {"response_format": rf_schema(G2_SCHEMA)}, {}, G2_INSTANCE),
        ("G3", {"response_format": rf_schema(G3_SCHEMA)}, {}, json.dumps({"p": {"x": 1}, "v": None})),
        ("G4", {"regex": json.dumps(r"\d{4}-\d{2}-\d{2}")}, {}, "2024-01-02"),
        ("G5", {"ebnf": json.dumps(G5_EBNF)}, {}, "1+2*3"),
        ("G6", {"tools_json": two, "tool_choice": '"required"'}, {}, call("g", [("q", "hi")])),
        ("G7", {"tools_json": json.dumps([F_TOOL, G_STRICT]),
                "tool_choice": '{"type": "function", "function": {"name": "g"}}'}, {}, call("g", [("q", "hi")])),
        ("G8", {"response_format": rf_schema(G2_SCHEMA)}, {"stream": True}, G2_INSTANCE),
        ("G9", {"tools_json": json.dumps([G_STRICT])}, {}, call("g", [("q", "hi")])),
    ]


def j(value) -> str:
    return json.dumps(value)


OBJ_A = {"type": "object", "properties": {"a": {"type": "integer"}}}
# (keyword, schema, conforming instances, violating instances); instances are JSON texts.
KEYWORD_VARIANTS = [
    ("type", {"type": "integer"}, ["3"], ['"x"', "3.5"]),
    ("type", {"type": "string"}, ['"x"'], ["3", "null"]),
    ("properties", OBJ_A, [j({"a": 1})], [j({"a": "x"})]),
    ("required", dict(OBJ_A, required=["a"]), [j({"a": 1})], [j({})]),
    ("additionalProperties", dict(OBJ_A, additionalProperties=False), [j({"a": 1})], [j({"a": 1, "b": 2})]),
    ("additionalProperties", {"type": "object", "additionalProperties": {"type": "integer"}}, [j({"b": 1})],
     [j({"b": "x"})]),
    ("items", {"type": "array", "items": {"type": "integer"}}, [j([1, 2])], [j([1, "x"])]),
    ("prefixItems", {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "items": False},
     [j([1, "a"])], [j(["a", 1]), j([1, "a", 2])]),
    ("enum", {"enum": ["a", "b", 1]}, [j("a"), "1"], [j("c"), "2"]),
    ("const", {"const": "x"}, [j("x")], [j("y")]),
    ("anyOf", {"anyOf": [{"type": "integer"}, {"type": "string"}]}, ["1", j("s")], ["true"]),
    ("allOf", {"allOf": [dict(OBJ_A, required=["a"]),
                         {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}]},
     [j({"a": 1, "b": "x"})], [j({"a": 1}), j({"b": "x"})]),
    ("$ref", {"$defs": {"p": {"type": "integer"}}, "type": "object", "properties": {"a": {"$ref": "#/$defs/p"}},
              "required": ["a"]}, [j({"a": 1})], [j({"a": "x"})]),
    ("$defs", {"$defs": {"s": {"type": "array", "items": {"$ref": "#/$defs/s"}, "maxItems": 1}},
               "$ref": "#/$defs/s"}, [j([[]])], [j(["x"]), j([[], []])]),
    ("minItems", {"type": "array", "items": {"type": "integer"}, "minItems": 2}, [j([1, 2])], [j([1])]),
    ("maxItems", {"type": "array", "items": {"type": "integer"}, "maxItems": 2}, [j([1, 2])], [j([1, 2, 3])]),
    ("minLength", {"type": "string", "minLength": 2}, [j("ab")], [j("a")]),
    ("maxLength", {"type": "string", "maxLength": 2}, [j("ab")], [j("abc")]),
    ("pattern", {"type": "string", "pattern": "^[a-z]+$"}, [j("abc")], [j("aB"), j("")]),
    ("pattern", {"type": "string", "pattern": "[0-9]{3}"}, [j("123")], [j("12"), j("1234")]),
    ("minimum", {"type": "integer", "minimum": 3}, ["3", "10"], ["2", "-3"]),
    ("minimum", {"type": "integer", "minimum": 3, "maximum": 100}, ["3", "100"], ["2", "-3"]),
    ("minimum", {"type": "number", "minimum": 1.5}, ["1.5", "2"], ["1.4"]),
    ("maximum", {"type": "integer", "maximum": 3}, ["3", "-1"], ["4", "30"]),
    ("maximum", {"type": "integer", "minimum": -100, "maximum": 3}, ["3", "-100"], ["4", "-101"]),
    ("maximum", {"type": "number", "maximum": 1.5}, ["1.5", "1"], ["1.6"]),
    ("exclusiveMinimum", {"type": "integer", "exclusiveMinimum": 3}, ["4"], ["3", "2"]),
    ("exclusiveMinimum", {"type": "integer", "exclusiveMinimum": 3, "maximum": 100}, ["4"], ["3", "2"]),
    ("exclusiveMinimum", {"type": "number", "exclusiveMinimum": 1.5}, ["1.6"], ["1.5"]),
    ("exclusiveMaximum", {"type": "integer", "exclusiveMaximum": 3}, ["2"], ["3", "4"]),
    ("exclusiveMaximum", {"type": "integer", "minimum": 0, "exclusiveMaximum": 3}, ["2"], ["3", "4"]),
    ("exclusiveMaximum", {"type": "number", "exclusiveMaximum": 1.5}, ["1.4"], ["1.5"]),
    ("title", {"type": "integer", "title": "t"}, ["1"], [j("x")]),
    ("description", {"type": "integer", "description": "d"}, ["1"], [j("x")]),
    ("default", {"type": "integer", "default": 1}, ["1"], [j("x")]),
]
ANNOTATIONS = ("default", "description", "title")
CANDIDATE_KEYWORDS = ("type properties required additionalProperties items prefixItems enum const anyOf allOf $ref "
                      "$defs minItems maxItems minLength maxLength pattern format minimum maximum exclusiveMinimum "
                      "exclusiveMaximum").split()
# Every `format` value xgrammar 0.2.1 names (libxgrammar_bindings.so string table): conforming, violating.
FORMAT_VARIANTS = [
    ("date", ["2024-01-02"], ["x", "2024-1-2"]),
    ("date-time", ["2024-01-02T12:00:00Z"], ["2024-01-02", "x"]),
    ("duration", ["P1D", "PT1H"], ["1D", "x"]),
    ("email", ["a@b.co"], ["ab", "a b@c"]),
    ("hostname", ["example.com"], ["a b", "a!b"]),
    ("ipv4", ["192.168.0.1"], ["1.2.3", "a.b.c.d"]),
    ("ipv6", ["::1", "2001:db8::1"], ["xyz", "1.2.3.4.5"]),
    ("json-pointer", ["/a/b", ""], ["a", "/~2"]),
    ("relative-json-pointer", ["0/a", "1#"], ["/a", "x"]),
    ("time", ["12:00:00Z"], ["x", "12:00"]),
    ("uri", ["https://a.b/c"], ["x y", "/path"]),
    ("uri-reference", ["/a/b", "https://a.b"], ["a b", "%zz"]),
    ("uri-template", ["/a/{b}"], ["/a/{b", "a b"]),
    ("uuid", ["123e4567-e89b-12d3-a456-426614174000"], ["123", "g23e4567-e89b-12d3-a456-426614174000"]),
]


def enforcement_variants() -> list:
    """(name, keyword|None, format|None, schema, conforming, violating)."""
    out, seen = [], {}
    for kw, schema, ok, bad in KEYWORD_VARIANTS:
        seen[kw] = seen.get(kw, 0) + 1
        out.append((f"{kw}#{seen[kw]}", kw, None, schema, ok, bad))
    for fmt, ok, bad in FORMAT_VARIANTS:
        out.append((f"format={fmt}", None, fmt, {"type": "string", "format": fmt}, [j(s) for s in ok],
                    [j(s) for s in bad]))
    return out


def k_cases() -> list:
    """The 40-schema keyword matrix: the first 40 enforcement variants as json_schema requests."""
    variants = enforcement_variants()
    if len(variants) < 40:
        raise SystemExit("fewer than 40 keyword variants")
    return [(f"K{i + 1:02d}", {"response_format": rf_schema(v[3])}, {}, v[4][0]) for i, v in enumerate(variants[:40])]


BOUND_KEYWORDS = ("exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum")  # value-restricted (D5)
BOUND_TYPES = ("integer", "number")
# Every bound form: a sorted keyword tuple, and how a closed range [lo, hi] maps onto its schema values.
BOUND_FORMS = {
    ("maximum", "minimum"): lambda lo, hi: {"minimum": lo, "maximum": hi},
    ("maximum",): lambda lo, hi: {"maximum": hi},
    ("exclusiveMaximum",): lambda lo, hi: {"exclusiveMaximum": hi + 1},
    ("exclusiveMaximum", "exclusiveMinimum"): lambda lo, hi: {"exclusiveMinimum": lo - 1, "exclusiveMaximum": hi + 1},
    ("exclusiveMinimum", "maximum"): lambda lo, hi: {"exclusiveMinimum": lo - 1, "maximum": hi},
    ("exclusiveMaximum", "minimum"): lambda lo, hi: {"minimum": lo, "exclusiveMaximum": hi + 1},
    ("minimum",): lambda lo, hi: {"minimum": lo},
    ("exclusiveMinimum",): lambda lo, hi: {"exclusiveMinimum": lo - 1},
}
BOUND_RANGES = ((-12, 250), (0, 9), (100, 1000))


def bound_variants_of() -> list:
    """(name, type, form, schema, conforming, violating): each form over each range; the range ends and an inside
    value conform (one-sided forms also a far value on the open side), the integers just outside each bounded end
    (and a far one) violate. Number ranges add the half-steps just outside the ends."""
    out = []
    for typ in BOUND_TYPES:
        for form, values in BOUND_FORMS.items():
            for lo, hi in BOUND_RANGES:
                low = any(k in form for k in ("minimum", "exclusiveMinimum"))
                high = any(k in form for k in ("maximum", "exclusiveMaximum"))
                ok = [lo, (lo + hi) // 2, hi] + ([] if low else [lo - 1000]) + ([] if high else [hi + 100000])
                bad = ([lo - 1, lo - 100] if low else []) + ([hi + 1, hi + 100] if high else [])
                if typ == "number":
                    bad += ([lo - 0.5] if low else []) + ([hi + 0.5] if high else [])
                schema = {"type": typ, **values(lo, hi)}
                out.append((f"{typ}:{'+'.join(form)}:[{lo},{hi}]", typ, form, schema, [j(v) for v in ok],
                            [j(v) for v in bad]))
    return out


OBJ_ABC = {"type": "object", "properties": {k: {"type": "integer"} for k in "abc"}}
# A probe of every JSON kind: an unresolved `$ref` that compiles to AnySpec accepts them all.
NON_OBJECTS = ["1", j("x"), "[]", "null", "true"]
# (name, group, schema, conforming, violating). Group A: a Z2a claim that xgrammar drops a keyword in this
# combination (the claim is confirmed when the variant fails). Group B: a form the D5 walker accepts; it must pass.
COMBINATION_VARIANTS = [
    ("A:string:pattern+maxLength", "A", {"type": "string", "pattern": "^[a-z]+$", "maxLength": 5},
     [j("abc")], [j("abcdefg"), j("ABC")]),
    ("A:string:format+maxLength", "A", {"type": "string", "format": "email", "maxLength": 12},
     [j("a@b.co")], [j("abcdefghij@example.com"), j("x")]),
    ("A:string:format+pattern", "A", {"type": "string", "format": "date", "pattern": "^2024-"},
     [j("2024-01-02")], [j("2023-01-02"), j("x")]),
    ("A:$ref+required", "A",
     {"$defs": {"base": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}},
      "$ref": "#/$defs/base", "required": ["x"]},
     [j({"x": 1}), j({"x": 1, "y": 2})], [j({"y": 1}), j({})]),
    ("A:object+anyOf", "A", dict(OBJ_ABC, required=["a"], anyOf=[{"required": ["b"]}, {"required": ["c"]}]),
     [j({"a": 1, "b": 2}), j({"a": 1, "c": 2})], [j({"a": 1}), j({"b": 2}), "3"]),
    ("A:integer:maximum+anyOf", "A", {"type": "integer", "maximum": 5, "anyOf": [{"type": "integer"},
                                                                                 {"type": "string"}]},
     ["3", "5"], ["6", j("x")]),
    ("A:typeless:pattern", "A", {"pattern": "^[A-Z]{3}$"}, [j("ABC")], [j("abcd")]),
    ("A:typeless:pattern-in-properties", "A",
     {"type": "object", "properties": {"code": {"pattern": "^[A-Z]{3}$"}}, "required": ["code"]},
     [j({"code": "ABC"})], [j({"code": "abcd"})]),
    ("A:typeless:minLength", "A", {"minLength": 3}, [j("abc")], [j("ab")]),
    ("A:typeless:required", "A", {"required": ["a"]}, [j({"a": 1})], [j({})]),
    ("A:$ref:https", "A", {"$ref": "https://example.com/addr.json"}, ["{}"], NON_OBJECTS),
    ("A:$ref:other-file", "A", {"type": "object", "properties": {"a": {"$ref": "other.json#/x"}}, "required": ["a"]},
     [j({"a": {}})], [j({"a": v}) for v in (1, "x", [], None)]),
    ("A:$ref:#/default", "A",
     {"default": {"type": "string", "not": {"const": "x"}}, "type": "object",
      "properties": {"a": {"$ref": "#/default"}}, "required": ["a"]},
     [j({"a": "y"})], [j({"a": "x"})]),
    ("A:required-not-in-properties:no-properties", "A", {"type": "object", "required": ["a"]},
     [j({"a": 1})], [j({})]),
    ("A:required-not-in-properties:extra-name", "A",
     {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "c"]},
     [j({"a": 1, "c": 2})], [j({"a": 1}), j({"a": 1, "b": 2})]),
    ("B:string:format", "B", {"type": "string", "format": "email", "title": "t"}, [j("a@b.co")], [j("x"), j("a@")]),
    ("B:string:pattern", "B", {"type": "string", "pattern": "^[a-z]+$", "description": "d"},
     [j("abc")], [j("ABC"), j("a1"), j("")]),
    ("B:string:minLength+maxLength", "B", {"type": "string", "minLength": 2, "maxLength": 4},
     [j("ab"), j("abcd")], [j("a"), j("abcde")]),
    ("B:string:minLength", "B", {"type": "string", "minLength": 2}, [j("ab"), j("abcdefgh")], [j("a"), j("")]),
    ("B:string:maxLength", "B", {"type": "string", "maxLength": 3}, [j(""), j("abc")], [j("abcd")]),
    ("B:$ref+annotations", "B",
     {"$defs": {"p": {"type": "integer", "maximum": 5}}, "type": "object",
      "properties": {"a": {"$ref": "#/$defs/p", "title": "t", "description": "d", "default": 1}}, "required": ["a"]},
     [j({"a": 3}), j({"a": 5})], [j({"a": 6}), j({"a": "x"}), j({})]),
    ("B:$ref:#", "B", {"type": "object", "properties": {"c": {"$ref": "#"}}},
     [j({}), j({"c": {}}), j({"c": {"c": {}}})], [j({"c": 1}), j({"c": {"c": 1}})]),
    ("B:$ref:root-$defs", "B", {"$defs": {"x": {"type": "string", "pattern": "^[0-9]+$"}}, "$ref": "#/$defs/x"},
     [j("123")], [j("12a"), "123"]),
    ("B:$ref:chain", "B",
     {"$defs": {"x": {"$ref": "#/$defs/y"}, "y": {"type": "integer", "minimum": 0, "maximum": 9}},
      "type": "array", "items": {"$ref": "#/$defs/x"}},
     [j([]), j([0, 9])], [j([10]), j(["a"]), j({})]),
    ("B:const+annotations", "B", {"const": "x", "description": "d", "title": "t"}, [j("x")], [j("y"), "1"]),
    ("B:enum+annotations", "B", {"enum": [1, "a"], "title": "t", "default": 1}, ["1", j("a")], ["2", j("b")]),
    ("B:anyOf+annotations", "B",
     {"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 2}], "description": "d"},
     ["3", j("ab")], [j("abc"), "null", "3.5"]),
    ("B:required-subset", "B", dict(OBJ_ABC, required=["a"]),
     [j({"a": 1}), j({"a": 1, "b": 2, "c": 3})], [j({"b": 2}), j({}), j({"a": "x"})]),
    ("B:required-all", "B", dict(OBJ_ABC, required=["a", "b", "c"]),
     [j({"a": 1, "b": 2, "c": 3})], [j({"a": 1, "b": 2}), j({})]),
    ("B:type-array:integer+string", "B", {"type": ["integer", "string"]}, ["1", j("x")], ["null", "[]", "1.5"]),
    ("B:type-array:string+null", "B", {"type": ["string", "null"]}, [j("x"), "null"], ["1", "{}"]),
    ("B:type-array:object+array", "B", {"type": ["object", "array"]}, ["{}", "[]", j([1, "x"])],
     ["1", j("x"), "null"]),
    # O113 (owner 2026-10-17): `type` beside `enum`. A: a value contradicts the type, which xgrammar drops, so the
    # contradicting value is accepted. B: every value is of the type; the grammar equals the enum's (checked too).
    ("A:type+enum:string-with-integer", "A", {"type": "string", "enum": ["a", 1]}, [j("a")], ["1"]),
    ("A:type+enum:integer-with-fraction", "A", {"type": "integer", "enum": [1, 1.5]}, ["1"], ["1.5"]),
    ("B:type+enum:string", "B", {"type": "string", "enum": ["a", "b"], "description": "d"},
     [j("a"), j("b")], [j("c"), "1", "null"]),
    ("B:type+enum:integer", "B", {"type": "integer", "enum": [1, 2]}, ["1", "2"], ["3", j("1"), "1.5"]),
    ("B:type+enum:type-array", "B", {"type": ["string", "null"], "enum": ["a", None]}, [j("a"), "null"],
     [j("b"), "1"]),
    ("B:type+enum:in-properties", "B",
     {"type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["unit"]},
     [j({"unit": "c"}), j({"unit": "f"})], [j({"unit": "k"}), j({"unit": 1}), j({})]),
]
NUMBER_RANGE_TC64 = {"type": "number", "minimum": 0, "maximum": 10}  # FP4-Local tool-eval-bench TC-64
# (name, schema, conforming, violating). Conforming instances may use a fraction, an exponent and `-0`.
NUMBER_RANGE_VARIANTS = [
    ("number:[0,10]", NUMBER_RANGE_TC64, ["0", "10", "3.5", "10.0", "1e1", "0.0", "-0", "1E+1", "100e-1", "0.25e1"],
     ["-1", "10.5", "11", "1e2", "-0.1", "10.000001", "1e+2", j("5"), "null"]),
    ("number:[0.5,2.25]", {"type": "number", "minimum": 0.5, "maximum": 2.25},
     ["0.5", "2.25", "1", "2.2", "225e-2", "0.50", "5e-1"], ["0.49", "2.26", "2.251", "0.4999", "3", "0", "-1"]),
    ("number:[-7.25,-1.5]", {"type": "number", "minimum": -7.25, "maximum": -1.5},
     ["-7.25", "-1.5", "-3", "-1.50", "-725e-2"], ["-7.26", "-1.49", "0", "1.5", "-8", "-1", "7"]),
    ("number:[-3,4]", {"type": "number", "minimum": -3, "maximum": 4},
     ["-3", "0", "-0", "4", "-2.999", "3.99", "-3.0"], ["-3.01", "4.001", "5", "-4", "40"]),
    ("number:[100,1000]", {"type": "number", "minimum": 100, "maximum": 1000, "description": "d"},
     ["100", "1000", "1e3", "555.5", "1000.000"], ["99.99", "1000.1", "1e4", "10", "0"]),
    ("number:[1.5,1.5]", {"type": "number", "minimum": 1.5, "maximum": 1.5}, ["1.5", "15e-1", "1.50"],
     ["1.4", "1.6", "1", "2"]),
    ("object:properties", {"type": "object", "properties": {"x": NUMBER_RANGE_TC64}, "required": ["x"]},
     [j({"x": 3.5}), j({"x": 10})], [j({"x": 10.5}), j({"x": -1}), j({})]),
    ("array:items", {"type": "array", "items": NUMBER_RANGE_TC64}, [j([]), j([0, 10, 2.5])],
     [j([11]), j([1, -0.5]), j([10.5])]),
    ("$defs+$ref", {"$defs": {"r": NUMBER_RANGE_TC64}, "type": "object", "properties": {"a": {"$ref": "#/$defs/r"},
                                                                                         "b": {"$ref": "#/$defs/r"}}},
     [j({"a": 1, "b": 9.5})], [j({"a": 1, "b": 12}), j({"a": 20}), j({"a": 1, "b": 10.5})]),
    ("anyOf", {"anyOf": [NUMBER_RANGE_TC64, {"type": "string", "maxLength": 2}]}, ["7", j("ab")],
     ["11", j("abc"), "null", "10.5"]),
]
TYPED_ENUM = ":type+enum:"
KEYWORD_SECTIONS = ("variants", "bound_variants", "combination_variants", "number_range_variants")


# ------------------------------------------------------------------------------------------------------------------
# The sidecar process.


class Sidecar:
    def __init__(self, model: Path):
        self.proc = subprocess.Popen([sys.executable, str(SIDECAR), "--artifact", str(model), "--role", "grammar"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def ask(self, request: dict) -> dict:
        data = json.dumps(request).encode()
        self.proc.stdin.write(HEADER.pack(len(data)) + data)
        self.proc.stdin.flush()
        header = self.proc.stdout.read(HEADER.size)
        if len(header) != HEADER.size:
            raise SystemExit(f"the grammar sidecar exited ({self.proc.wait()})")
        (n,) = HEADER.unpack(header)
        return json.loads(self.proc.stdout.read(n))

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def mask_bytes(b64: str) -> bytes:
    data = base64.b64decode(b64, validate=True)
    if len(data) != MASK_WORDS * 4:
        raise SystemExit("a sidecar mask has the wrong length")
    return data


def allowed(mask: bytes, token: int) -> bool:
    return bool(mask[4 * (token >> 5) + ((token & 31) >> 3)] >> (token & 7) & 1)


# ------------------------------------------------------------------------------------------------------------------
# SGLang's side.


class Oracle:
    def __init__(self, model: Path):
        hf, self.tok = api.load_tokenizers(model)
        self.serving, _ = api.serving_chat(model, self.tok)
        config = ModelConfig(str(model))
        if config.vocab_size != VOCAB_SIZE:
            raise SystemExit("the model vocabulary is not 248320")
        self.eos = sorted(config.hf_eos_token_id)
        # managers/scheduler.py:855-869: think_end_ids from the reasoning parser's end token.
        parser = ReasoningParser(model_type="qwen3", stream_reasoning=False, tokenizer=self.tok)
        think_end_ids = self.tok.encode(parser.detector.think_end_token, add_special_tokens=False)
        server_args = api.Loose("server_args", grammar_backend="xgrammar", reasoning_parser="qwen3",
                                constrained_json_disable_any_whitespace=False, enable_strict_thinking=False)
        self.backend = create_grammar_backend(server_args, self.tok, VOCAB_SIZE, set(self.eos), think_end_ids)
        if type(self.backend).__name__ != "ReasonerGrammarBackend":
            raise SystemExit("the reference backend is not wrapped as the scheduler wraps it")
        info = xgrammar.TokenizerInfo.from_huggingface(self.tok, vocab_size=VOCAB_SIZE, stop_token_ids=self.eos)
        self.stop = sorted(info.stop_token_ids)
        self.by_bytes: dict = {}
        for token, piece in enumerate(info.decoded_vocab):
            if piece:
                self.by_bytes.setdefault(piece, []).append(token)
        self.max_piece = max(len(p) for p in self.by_bytes)

    def key(self, fields: dict, extra: dict):
        """SGLang's (key_type, key_string, require_reasoning), or None when SGLang refuses the request."""
        body = {"model": "fixture", "messages": [{"role": "user", "content": "x"}], **extra}
        text = json.dumps(body)[:-1]
        for name, raw in fields.items():
            text += f', "{"tools" if name == "tools_json" else name}": {raw}'
        try:
            req = ChatCompletionRequest.model_validate(orjson.loads((text + "}").encode()))
        except Exception:  # noqa: BLE001 - pydantic refuses the body
            return None
        if self.serving._validate_request(req) is not None:
            return None
        try:
            processed = self.serving._process_messages(req, False)
            params = req.to_sampling_params(
                stop=processed.stop, model_generation_config=self.serving.default_sampling_params,
                tool_call_constraint=processed.tool_call_constraint,
                renderer_handles_response_format=self.serving.chat_encoding_spec == "kimi_k3")
        except Exception:  # noqa: BLE001 - SGLang answers 400
            return None
        for key_type, field in (("json", "json_schema"), ("regex", "regex"), ("ebnf", "ebnf"),
                                ("structural_tag", "structural_tag")):
            if params.get(field) is not None:
                return key_type, params[field], processed.require_reasoning
        return None

    def grammar(self, key_type: str, key: str, require_reasoning: bool):
        value, _ = self.backend.get_cached_or_future_value((key_type, key), require_reasoning)
        value = value.result() if hasattr(value, "result") else value
        return None if isinstance(value, InvalidGrammarObject) else value

    @staticmethod
    def mask(grammar) -> bytes:
        vocab_mask = grammar.allocate_vocab_mask(VOCAB_SIZE, 1, "cpu")
        grammar.fill_vocab_mask(vocab_mask, 0)
        return vocab_mask[0].numpy().tobytes()

    def next_token(self, mask: bytes, rest: bytes):
        """The longest allowed token whose bytes prefix `rest`; at the end the lowest allowed stop token."""
        if not rest:
            return next((t for t in self.stop if allowed(mask, t)), None)
        for n in range(min(len(rest), self.max_piece), 0, -1):
            for token in self.by_bytes.get(rest[:n], ()):
                if allowed(mask, token):
                    return token, n
        return None


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_case(sidecar: Sidecar, oracle: Oracle, name: str, fields: dict, extra: dict, instance: str,
             failures: list) -> dict:
    case = {"name": name, "request": fields}
    if extra:
        case["sglang_extra"] = extra
    reply = sidecar.ask({"op": "grammar_open", **fields})
    ref = oracle.key(fields, extra)
    ref_grammar = oracle.grammar(*ref) if ref is not None else None
    if not reply["ok"]:
        case.update(outcome="refused", error={"type": reply["error"]["type"], "param": reply["error"].get("param")})
        if ref_grammar is not None:
            failures.append(f"{name} (sidecar refused, SGLang compiled)")
        if ref is not None:
            case["key_type"], case["key_sha256"] = ref[0], sha(ref[1].encode())
        return case
    handle = reply["grammar"]
    case.update(outcome="ok", key_type=reply["key_type"], key_sha256=reply["key_sha256"])
    try:
        if ref is None or (ref[0], sha(ref[1].encode())) != (reply["key_type"], reply["key_sha256"]):
            failures.append(f"{name} (key)")
            return case
        if ref[2]:
            failures.append(f"{name} (SGLang requires reasoning)")
            return case
        if ref_grammar is None:
            failures.append(f"{name} (sidecar compiled, SGLang refused)")
            return case
        mask = mask_bytes(reply["bitmask"])
        if mask != oracle.mask(ref_grammar):
            failures.append(f"{name} (mask 0)")
            return case
        masks = [mask]
        path = walk(sidecar, oracle, handle, ref_grammar, instance, masks, name, failures)
        if path is None:
            return case
        case.update(instance_sha256=sha(instance.encode("utf-8")), path_len=len(path),
                    path_sha256=sha(struct.pack(f"<{len(path)}I", *path)), stop_token=path[-1],
                    masks=len(masks), open_mask_sha256=sha(masks[0]), masks_sha256=sha(b"".join(masks)))
        return case
    finally:
        sidecar.ask({"op": "grammar_close", "grammar": handle})


def walk(sidecar: Sidecar, oracle: Oracle, handle: int, ref_grammar, instance: str, masks: list, name: str,
         failures: list):
    """The oracle path of `instance` from the grammar's current mask `masks[-1]`, stepped through both sides: the
    tokens, or None after recording a failure. Every non-final step's mask is appended to `masks`."""
    mask, rest, path, prefix = masks[-1], instance.encode("utf-8"), [], len(masks) - 1
    while True:
        pick = oracle.next_token(mask, rest)
        if pick is None or len(path) > len(instance) + 1:
            failures.append(f"{name} (no oracle path at step {prefix + len(path)})")
            return None
        token, used = pick if isinstance(pick, tuple) else (pick, 0)
        rest = rest[used:]
        path.append(token)
        step = sidecar.ask({"op": "grammar_step", "grammar": handle, "token": token})
        ref_grammar.accept_token(token)
        terminated = bool(ref_grammar.is_terminated())
        if not step["accepted"] or step["terminated"] != terminated:
            failures.append(f"{name} (step {prefix + len(path)} accepted/terminated)")
            return None
        if used == 0 and not terminated:
            failures.append(f"{name} (stop token did not terminate)")
            return None
        if terminated:
            if step["bitmask"] is not None or rest:
                failures.append(f"{name} (termination at step {prefix + len(path)})")
                return None
            return path
        mask = mask_bytes(step["bitmask"])
        masks.append(mask)
        if mask != oracle.mask(ref_grammar):
            failures.append(f"{name} (mask {prefix + len(path)})")
            return None


# The D15 thinking-deferred mask (ledger CR20): G2's schema opened with `enable_thinking`, against SGLang's
# `ReasonerGrammarObject` built for a reasoning request. The reasoning tokens are this synthetic text's encoding, then
# a grammar-legal first token and the lowest model EOS (neither may reach the grammar or terminate it), then the
# think-end ids; G2's instance follows.
THINK_TEXT = "synthetic reasoning about the schema"
THINK_EXTRA = {"chat_template_kwargs": {"enable_thinking": True}}


def run_thinking_case(sidecar: Sidecar, oracle: Oracle, failures: list) -> dict:
    name, fields = "T1", {"response_format": rf_schema(G2_SCHEMA)}
    case = {"name": name, "request": {**fields, "enable_thinking": True}, "sglang_extra": THINK_EXTRA}
    reply = sidecar.ask({"op": "grammar_open", **case["request"]})
    ref = oracle.key(fields, THINK_EXTRA)
    if not reply["ok"]:
        failures.append(f"{name} (sidecar refused)")
        return case
    handle = reply["grammar"]
    case.update(outcome="ok", key_type=reply["key_type"], key_sha256=reply["key_sha256"])
    try:
        if ref is None or (ref[0], sha(ref[1].encode())) != (reply["key_type"], reply["key_sha256"]):
            failures.append(f"{name} (key)")
            return case
        if not ref[2]:
            failures.append(f"{name} (SGLang does not require reasoning)")
            return case
        ref_grammar = oracle.grammar(*ref)
        if ref_grammar is None:
            failures.append(f"{name} (sidecar compiled, SGLang refused)")
            return case
        mask = mask_bytes(reply["bitmask"])
        if mask != oracle.mask(ref_grammar):
            failures.append(f"{name} (mask 0)")
            return case
        masks = [mask]
        first = oracle.next_token(oracle.mask(oracle.grammar(*ref[:2], False)), G2_INSTANCE.encode("utf-8"))[0]
        think = (oracle.tok.encode(THINK_TEXT, add_special_tokens=False) + [first, oracle.eos[0]]
                 + list(oracle.backend.think_end_ids))
        for i, token in enumerate(think, 1):
            step = sidecar.ask({"op": "grammar_step", "grammar": handle, "token": token})
            ref_grammar.accept_token(token)
            if not step["accepted"] or step["terminated"] != bool(ref_grammar.is_terminated()) or step["terminated"]:
                failures.append(f"{name} (step {i} accepted/terminated)")
                return case
            masks.append(mask_bytes(step["bitmask"]))
            if masks[-1] != oracle.mask(ref_grammar):
                failures.append(f"{name} (mask {i})")
                return case
        reasoning_masks = len(masks) - 1  # the open mask and every reasoning step's but the think-end one
        path = walk(sidecar, oracle, handle, ref_grammar, G2_INSTANCE, masks, name, failures)
        if path is None:
            return case
        all_ones = b"\xff" * (MASK_WORDS * 4)
        if masks[:reasoning_masks] != [all_ones] * reasoning_masks or masks[reasoning_masks] == all_ones:
            failures.append(f"{name} (reasoning masks)")
            return case
        steps = think + path
        case.update(think_len=len(think), think_sha256=sha(struct.pack(f"<{len(think)}I", *think)),
                    path_len=len(path), path_sha256=sha(struct.pack(f"<{len(path)}I", *path)),
                    stop_token=path[-1], steps_sha256=sha(struct.pack(f"<{len(steps)}I", *steps)),
                    masks=len(masks), all_ones_masks=reasoning_masks, open_mask_sha256=sha(masks[0]),
                    think_end_mask_sha256=sha(masks[reasoning_masks]), masks_sha256=sha(b"".join(masks)))
        return case
    finally:
        sidecar.ask({"op": "grammar_close", "grammar": handle})


def sidecar_accepts(sidecar: Sidecar, oracle: Oracle, schema, instance: str) -> bool | None:
    """Whether the sidecar's grammar for `schema` (a json_schema response_format) accepts every token of the
    instance's encoding and then the lowest model EOS, terminating; None when the open is refused."""
    reply = sidecar.ask({"op": "grammar_open", "response_format": rf_schema(schema)})
    if not reply["ok"]:
        return None
    handle = reply["grammar"]
    try:
        for token in oracle.tok.encode(instance, add_special_tokens=False) + [oracle.eos[0]]:
            step = sidecar.ask({"op": "grammar_step", "grammar": handle, "token": token})
            if not step["accepted"]:
                return False
        return step["terminated"]
    finally:
        sidecar.ask({"op": "grammar_close", "grammar": handle})


def number_range_variants(sidecar: Sidecar, oracle: Oracle, failures: list) -> list:
    out = []
    for name, schema, ok, bad in NUMBER_RANGE_VARIANTS:
        accepted = [sidecar_accepts(sidecar, oracle, schema, s) for s in ok]
        rejected = [sidecar_accepts(sidecar, oracle, schema, s) for s in bad]
        compiled = None not in accepted + rejected
        v = {"name": name, "schema": schema, "conforming_sha256": [sha(s.encode()) for s in ok],
             "violating_sha256": [sha(s.encode()) for s in bad], "compiled": compiled,
             "accepted": [a is True for a in accepted], "rejected": [r is False for r in rejected],
             "stock_pass": check_variant(schema, ok, bad)["pass"]}
        v["pass"] = compiled and all(v["accepted"]) and all(v["rejected"])
        if not v["pass"]:
            failures.append(f"number range {name}")
        out.append(v)
    return out


def parity(model: Path, failures: list) -> tuple:
    sidecar = Sidecar(model)
    try:
        info = sidecar.ask({"op": "info"})
        oracle = Oracle(model)
        want = {"features": ["grammar", "grammar_drafts", "message_boundary", "thinking"],
                "xgrammar": importlib.metadata.version("xgrammar"), "torch": torch.__version__,
                "stop_token_ids": oracle.eos, "vocab_size": VOCAB_SIZE}
        for k, v in want.items():
            if info.get(k) != v:
                failures.append(f"info.{k}")
        cases = [run_case(sidecar, oracle, *c, failures) for c in g_cases() + k_cases()]
        cases.append(run_thinking_case(sidecar, oracle, failures))
        ranges = number_range_variants(sidecar, oracle, failures)
    finally:
        sidecar.close()
    by = {c["name"]: c for c in cases}
    if (by["G8"].get("key_sha256"), by["G8"].get("masks_sha256")) != (by["G2"].get("key_sha256"),
                                                                       by["G2"].get("masks_sha256")):
        failures.append("G8 (differs from G2)")
    # After reasoning T1 is G2: its think-end mask is G2's open mask and the same instance path follows.
    if (by["T1"].get("think_end_mask_sha256"), by["T1"].get("path_sha256")) != (by["G2"].get("open_mask_sha256"),
                                                                               by["G2"].get("path_sha256")):
        failures.append("T1 (differs from G2 after reasoning)")
    return cases, oracle.eos, ranges


# ------------------------------------------------------------------------------------------------------------------
# Enforcement.


def compile_captured(schema) -> tuple:
    """(grammar or None, warned): the compile with fd 2 and Python warnings captured."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as sink, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        os.dup2(sink.fileno(), 2)
        try:
            grammar = xgrammar.Grammar.from_json_schema(json.dumps(schema), any_whitespace=True)
        except Exception:  # noqa: BLE001 - a compile failure fails the variant
            grammar = None
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        sink.seek(0)
        warned = bool(sink.read()) or bool(caught)
    return grammar, warned


def check_variant(schema, ok: list, bad: list) -> dict:
    grammar, warned = compile_captured(schema)
    accepted = [grammar is not None and _is_grammar_accept_string(grammar, s) for s in ok]
    rejected = [grammar is not None and not _is_grammar_accept_string(grammar, s) for s in bad]
    return {"schema": schema, "conforming_sha256": [sha(s.encode()) for s in ok],
            "violating_sha256": [sha(s.encode()) for s in bad], "compiled": grammar is not None, "warned": warned,
            "accepted": accepted, "rejected": rejected,
            "pass": grammar is not None and not warned and all(accepted) and all(rejected) and bool(bad)}


def without_enum_type(schema):
    """`schema` with `type` removed from every object that has an `enum` beside it."""
    if isinstance(schema, dict):
        return {k: without_enum_type(v) for k, v in schema.items() if not (k == "type" and "enum" in schema)}
    if isinstance(schema, list):
        return [without_enum_type(v) for v in schema]
    return schema


def same_grammar_without_type(schema) -> bool:
    """O113: the compiled grammar text equals that of the schema without the `type` beside each `enum`, i.e.
    xgrammar compiles exactly the enum constraint (the `type` is dropped, so it adds nothing)."""
    grammars = [compile_captured(s)[0] for s in (schema, without_enum_type(schema))]
    return None not in grammars and str(grammars[0]) == str(grammars[1])


def enforcement() -> dict:
    variants, results = [], {}
    for name, kw, fmt, schema, ok, bad in enforcement_variants():
        v = {"name": name, "keyword": kw or "format", "format": fmt, **check_variant(schema, ok, bad)}
        variants.append(v)
        key = ("format", fmt) if fmt else ("keyword", kw)
        results[key] = results.get(key, True) and v["pass"]
    bound_variants, forms = [], {}
    for name, typ, form, schema, ok, bad in bound_variants_of():
        v = {"name": name, "type": typ, "form": list(form), **check_variant(schema, ok, bad)}
        bound_variants.append(v)
        forms[(typ, form)] = forms.get((typ, form), True) and v["pass"]
    bound_forms = {typ: sorted(list(f) for (t, f), p in forms.items() if t == typ and p) for typ in BOUND_TYPES}
    formats = sorted(f for (kind, f), p in results.items() if kind == "format" and p)
    keywords = sorted(k for (kind, k), p in results.items()
                      if kind == "keyword" and p and k not in ANNOTATIONS and k not in BOUND_KEYWORDS)
    if formats:
        keywords = sorted(keywords + ["format"])
    restricted = {kw: {typ: [f for f in fs if kw in f] for typ, fs in bound_forms.items() if any(kw in f for f in fs)}
                  for kw in BOUND_KEYWORDS}
    combination_variants = []
    for name, group, schema, ok, bad in COMBINATION_VARIANTS:
        v = {"name": name, "group": group, **check_variant(schema, ok, bad)}
        if TYPED_ENUM in name:
            v["same_grammar_without_type"] = same_grammar_without_type(schema)
        combination_variants.append(v)
    return {"dialect": DIALECT, "candidates": sorted(CANDIDATE_KEYWORDS), "keywords": keywords,
            "annotations": sorted(a for a in ANNOTATIONS if results[("keyword", a)]),
            "formats": formats, "bound_forms": bound_forms,
            "value_restricted_keywords": {k: v for k, v in sorted(restricted.items()) if v},
            "failed_keywords": sorted(k for k in CANDIDATE_KEYWORDS if k not in keywords),
            "failed_formats": sorted(f for (kind, f), p in results.items() if kind == "format" and not p),
            "combinations_enforced": sorted(v["name"] for v in combination_variants if v["pass"]),
            "combinations_unenforced": sorted(v["name"] for v in combination_variants if not v["pass"]),
            "variants": variants, "bound_variants": bound_variants, "combination_variants": combination_variants}


# ------------------------------------------------------------------------------------------------------------------
# Documents.


def header(stem: str, args, model: Path, eos: list) -> dict:
    head = api.common_header(stem, args, model)
    head["generator"] = GENERATOR
    head.update(xgrammar_version=importlib.metadata.version("xgrammar"), torch_version=torch.__version__,
                vocab_size=VOCAB_SIZE, mask_words=MASK_WORDS, stop_token_ids=eos)
    return head


def documents(args, failures: list) -> dict:
    model = args.model.resolve()
    cases, eos, ranges = parity(model, failures)
    enforce = {**enforcement(), "number_range_variants": ranges}
    sections = {s: enforce.pop(s) for s in KEYWORD_SECTIONS}
    return {
        CASES_FILE: api.doc_text(header("grammar_cases", args, model, eos), {"cases": cases}),
        KEYWORDS_FILE: api.doc_text({**header("grammar_keywords", args, model, eos), **enforce}, sections),
    }


def image_digest(image: str, model: Path) -> str:
    lock = json.loads((ROOT / "configs/deployments/qwen38-flash-next.lock.json").read_text())
    digest = image.split("@", 1)[1]
    if digest != "sha256:" + lock["backend_build_sha256"]:
        raise SystemExit("--image is not the deployment lock's backend build")
    return digest


def generate(args) -> int:
    failures: list = []
    docs = documents(args, failures)
    if failures:
        print("GRAMMAR FIXTURES GENERATE FAIL " + ", ".join(failures))
        return 1
    for name, text in docs.items():
        (args.out / name).write_text(text, encoding="utf-8")
    keywords = json.loads(docs[KEYWORDS_FILE])
    print(f"wrote {CASES_FILE} ({len(json.loads(docs[CASES_FILE])['cases'])} cases) and {KEYWORDS_FILE}; keywords "
          f"{' '.join(keywords['keywords'])}; formats {' '.join(keywords['formats'])}; failed "
          f"{' '.join(keywords['failed_keywords'] + keywords['failed_formats']) or 'none'}")
    return 0


def selfcheck(args) -> int:
    out = ROOT / "tests/fixtures/native"
    committed = {name: json.loads((out / name).read_text(encoding="utf-8")) for name in (CASES_FILE, KEYWORDS_FILE)}
    args.image = committed[CASES_FILE]["image_digest"]
    failures: list = []
    docs = {name: json.loads(text) for name, text in documents(args, failures).items()}
    for name, doc in docs.items():
        old = committed[name]
        sections = ("cases",) if name == CASES_FILE else KEYWORD_SECTIONS
        for key in doc:
            if key not in sections and doc[key] != old.get(key):
                failures.append(f"{name} {key}")
        for section in sections:
            olds = {c["name"]: c for c in old.get(section, [])}
            for c in doc[section]:
                if olds.get(c["name"]) != c:
                    failures.append(f"{name} {c['name']}")
            if len(olds) != len(doc[section]):
                failures.append(f"{name} {section} count")
    if failures:
        print("GRAMMAR FIXTURES SELFCHECK FAIL " + ", ".join(dict.fromkeys(failures)))
        return 1
    print(f"GRAMMAR FIXTURES SELFCHECK PASS {len(docs[CASES_FILE]['cases'])} parity cases, "
          f"{len(docs[KEYWORDS_FILE]['variants'])} enforcement variants, "
          f"{len(docs[KEYWORDS_FILE]['bound_variants'])} bound variants, "
          f"{len(docs[KEYWORDS_FILE]['combination_variants'])} combination variants, "
          f"{len(docs[KEYWORDS_FILE]['number_range_variants'])} number range variants")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="prepared native artifact (read only)")
    p.add_argument("--out", type=Path, help="tests/fixtures/native (generate mode)")
    p.add_argument("--image", help="the pinned image reference with digest (generate mode, recorded)")
    p.add_argument("--selfcheck", action="store_true", help="re-run every case against the committed fixtures")
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    if args.selfcheck:
        return selfcheck(args)
    if not (args.out and args.image and "@sha256:" in args.image):
        p.error("generate mode needs --out and --image NAME@sha256:DIGEST")
    args.image = image_digest(args.image, args.model)
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
