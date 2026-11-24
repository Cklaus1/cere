"""Tokenizer sidecar of the native backend (tasks/m22-spec.md Q9, B4). Container only; spawned by `cered`.

    python3 scripts/native/tokenizer_sidecar.py --artifact DIR [--role tokenizer|grammar] [--feature image]
        [--max-request-bytes N] [--thinking-kwarg false|unset]

Loads the tokenizer and chat template from the prepared artifact DIR only (`AutoTokenizer.from_pretrained(DIR,
local_files_only=True)`, Hugging Face offline), with the image's `transformers`/`tokenizers` (the libraries SGLang
uses), and answers requests on stdin/stdout until EOF.

Protocol `cere-tokenizer-sidecar-v3` (tasks/m24-spec.md D8, D9; tasks/native-abi-ledger.md S0: every v2 frame
unchanged plus `info.features`, the sorted list of named protocol features this sidecar serves: `message_boundary`
and `thinking` always, plus each `--feature` argument). Feature `thinking` (tasks/m212-spec.md D4, D10, D13):
`encode_chat.enable_thinking` (bool, default false; with `--thinking-kwarg unset` a request without it renders with
the template kwarg unset, not false: SelfTune ST4 S3d, DICE-8B's S2 reference renders its prompts without
`chat_template_kwargs`, and its template reads an unset `enable_thinking` as thinking on), `reasoning_effort` (a
string or a number, a template kwarg only when sent) and `preserve_thinking` (bool, a template kwarg only when sent;
`cered` sends only false, tasks/api-lenient-spec.md AL-2), `info.min_tokens_suppress_ids` (sorted ids). Feature
`message_boundary` (tasks/m210-spec.md D4 (c), ledger S/W 132): every `encode_chat` reply carries
`last_message_start`, the token index where the last message's rendering begins under the same template kwargs and
tools: the length of the common prefix of `ids` and the ids of `messages[:-1]` rendered without the generation prompt
(so it is always a prefix of `ids`, also where the template re-renders history, e.g. a tool message followed by
another); 0 for a single message, for an image request (image sessions never capture, m210 D15') and when that
history render fails (the hint is dropped, never the request).
A frame is a 4-byte little-endian unsigned length followed by that many
bytes of UTF-8 JSON (one object). Requests larger than `--max-request-bytes` (default 4 MiB; `cered` passes
`api.max_request_body_bytes` + 64 KiB, tasks/m213-spec.md D18) are refused (their bytes are read and dropped, so the
stream stays in sync). Every request frame gets exactly one reply frame, in order (plus one attachment frame after an
image reply, below):

    {"op": "info"}
      -> {"ok": true, "op": "info", "protocol": "cere-tokenizer-sidecar-v3", "tokenizer_sha256": hex,
          "template_sha256": hex, "transformers": version, "tokenizers": version, "features": [name, ...],
          "min_tokens_suppress_ids": [int, ...]}
    {"op": "encode_chat", "messages": [message, ...], "tools_json"?: str, "render_tools"?: bool, "stop"?: [str],
     "enable_thinking"?: bool, "reasoning_effort"?: str | number, "preserve_thinking"?: bool}
      -> {"ok": true, "op": "encode_chat", "ids": [int, ...], "stop_token_lens": [int per stop],
          "tools_prepared": [tool, ...] | null, "last_message_start": int}
    {"op": "tool_param", "tool_json": tool | null, "func": str | null, "name": str, "raw": str}
      -> {"ok": true, "op": "tool_param", "fragment": str}
    {"op": "tool_args", "tool_json": tool | null, "func": str | null, "params": [[name, raw], ...]}
      -> {"ok": true, "op": "tool_args", "arguments": str}
    any failure
      -> {"ok": false, "error": {"type": T, "message": fixed text}}, plus "param" for T = invalid_value
         T: frame_too_large | bad_json | unknown_op | bad_request | template_error | encode_error |
            invalid_value | convert_error | reference_unavailable | unsupported_op | invalid_request | overloaded

Role `grammar` (tasks/m28-spec.md D1, D7; default `tokenizer`, whose `features` stay ["message_boundary", "thinking"]
and which never imports `xgrammar`, `torch` or SGLang's grammar code) is the grammar process `cered` spawns next to
the tokenizer one. It serves every frame above plus features `grammar` and `grammar_drafts`; in the tokenizer role the
four grammar ops are unsupported_op:

    {"op": "info"} adds "grammar" and "grammar_drafts" to "features", "xgrammar": version, "torch": version,
          "stop_token_ids": [int, ...] (the model EOS, sorted), "vocab_size": 248320
    {"op": "grammar_open", "response_format"?: str, "regex"?: str, "ebnf"?: str, "tools_json"?: str,
     "tool_choice"?: str, "parallel_tool_calls"?: bool, "enable_thinking"?: bool}
                                                         (str fields: the request's raw JSON value text)
      -> {"ok": true, "op": "grammar_open", "grammar": int, "key_type": json|regex|ebnf|structural_tag,
          "key_sha256": hex, "bitmask": base64}
    {"op": "grammar_step", "grammar": int, "token": int}
      -> {"ok": true, "op": "grammar_step", "accepted": bool, "terminated": bool, "bitmask": base64 | null}
    {"op": "grammar_drafts", "grammar": int, "drafts": [int, ...]}     (1 to 3 ids in [0, 248320))
      -> {"ok": true, "op": "grammar_drafts", "accepted": int, "bitmasks": base64}
    {"op": "grammar_close", "grammar": int} -> {"ok": true, "op": "grammar_close"}

`grammar_open` builds SGLang's grammar key from a request carrying exactly those fields (`orjson.loads` of each raw
value, `ChatCompletionRequest.model_validate`): the tool checks of `OpenAIServingChat._validate_request`
(`serving_chat.py:834-870`: invalid_value `tool_choice`, `tools`, `tools[i].function.parameters`; a json_schema
response_format without a schema is invalid_value `response_format`), the tool constraint of `_process_messages`
(:1078-1121: `FunctionCallParser(tools, "qwen3_coder").get_structure_constraint(tool_choice, parallel_tool_calls,
thinking_mode=False)` - the bench server runs a reasoning parser - else `get_json_schema_constraint` for
required/named), `ChatCompletionRequest.to_sampling_params`, then the key priority of `grammar_manager.py:144-151`
(json_schema, regex, ebnf, structural_tag); no constraint is bad_request. `key_sha256` is the SHA-256 of the key
string's UTF-8. The key compiles as `XGrammarGrammarBackend` does (`xgrammar_backend.py:208-397`: `TokenizerInfo.
from_huggingface(tok, vocab_size=248320, stop_token_ids=EOS)`, json `any_whitespace=True`, the structural-tag
sanitizers) on one `GrammarCompiler(cache_limit_bytes=256 MiB)` built at start, except that a json key with a
`type: "number"` schema bounded by an inclusive `minimum` and `maximum` compiles to an EBNF whose number rule is the
exact range (tasks/m28-spec.md B7, `number_range_ebnf` below); a failure is invalid_value with
`param` = the field the key came from (`response_format`, `regex`, `ebnf`, or `tool_choice` for a tool constraint).
Each handle owns a `GrammarMatcher(max_rollback_tokens=200)`; at most 4 are open (a 5th is overloaded) and an
unknown handle is invalid_request. `bitmask` is the next-token mask as 7,760 little-endian u32 words (token j is
word j >> 5, bit j & 31); `grammar_step` accepts one token: a rejected token leaves the state unchanged (accepted
false), a terminated grammar (accepted EOS, or a step after it) carries no mask. The model EOS is SGLang's
`hf_eos_token_id` (`model_config.py:1640-1658`): config.json's `eos_token_id` united with generation_config.json's.

`grammar_drafts` (tasks/grammar-spec-spec.md; cered's speculative rounds on a constrained request) checks a round's
drafts against the grammar without moving it: `accepted` is the number of leading drafts the grammar accepts without
terminating (a rejected draft, or one after which the grammar is terminated, ends the chain uncounted) and `bitmasks`
the masks after each of them concatenated (accepted * 7,760 words; "" for none). The matcher is rolled back by every
token it accepted (a terminating one included), so the handle answers afterwards exactly as before the op.

`enable_thinking` (tasks/m28-spec.md D15, ledger CR20; bool, default false, sent by `cered` only when the request
enables thinking; a non-bool is invalid_value `enable_thinking`) defers the grammar past the reasoning as SGLang's
`ReasonerGrammarBackend` wraps every grammar when a reasoning parser runs (`base_grammar_backend.py:387-403`,
`reasoner_grammar_backend.py:35-241`, `require_reasoning` = the request's thinking mode, `serving_chat.py:1068,1151`):
the open mask and every step mask in reasoning are all-ones (`fill_vocab_mask` leaves xgrammar's freshly allocated
row, every bit set, :150-153), every step there is accepted and not terminated (:113-130) and the token never reaches
the matcher; the step whose token completes the think-end ids (the tokenizer's encoding of `</think>`, matched as a
sequence like `TokenSequenceMatcher`, :83-95) ends reasoning and returns the grammar's mask (its initial one); later
steps behave as without the field. In reasoning `grammar_drafts` simulates those steps without leaving reasoning:
drafts there are accepted with the all-ones mask, the draft completing the think-end ids carries the grammar's mask,
later drafts in the call reach the matcher (and are rolled back). The grammar role refuses to start (exit 2) when
`</think>` encodes to nothing (SGLang then serves ungated, `managers/scheduler.py:860-875`,
`reasoner_grammar_backend.py:253-261`).

Messages (the D3 shapes `cered` forwards): {role: system|user, content: str}; {role: assistant, content?: str|null,
tool_calls?: [{id: str, type: "function", function: {name: str, arguments: str}}], reasoning_content?: str}
(O113: passed to the template, which renders it in the turn's `<think>` block); {role: tool, content: str,
tool_call_id: str}. A v1 frame (`op` and `messages` only, system/user/assistant strings) encodes exactly as v1 did:
`render_tools` defaults to false, `stop` to [].

`encode_chat` is SGLang's chat path (`serving_chat.py`): content None -> "" (:1281-1283), assistant `arguments`
orjson-loaded to an object (:109-146), `apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
tools=..., return_dict=False, enable_thinking=False)` and, if that fails with tools, once more with the flat
function-only tools (:1334-1372), then `encode(rendered)` with `add_special_tokens=False` only when the tokenizer adds
specials to an empty string. `tools_json` (the request's raw `tools` bytes) is prepared whenever it is sent, rendered
or not: `orjson.loads` -> `Tool.model_validate` -> `normalize_json_schema_types` -> `Draft202012Validator.check_schema`
-> `model_dump()` (:856-873, :1084-1089); a parse or model failure is invalid_value with param `tools`, a schema
failure invalid_value with param `tools[i].function.parameters`. Tools are rendered iff `render_tools` and the prepared
list is non-empty. `stop_token_lens[i]` = `len(encode(stop[i], add_special_tokens=False))` (`sampling_params.py:218-228`).

Added-token text (tasks/m214-spec.md N5 = b) escape, owner decision 2026-10-15; native path only). Client text
never becomes an added token's id. Covered strings: system, user and tool `content` (a string or each text part) and
every key and string value of the rendered tools; assistant `content` and `tool_calls` are not covered (they reproduce
model output, so history re-encodes to the generated ids). A covered string that contains the text of any entry of
the tokenizer's `added_tokens_decoder` (special or not: `<|im_start|>`, `<think>`, `<tool_call>`, the vision markers)
is escaped. A request without one is rendered and encoded exactly as above (one render, one `encode`). Otherwise:
render as above; render again with each escaped string's core (the string without its leading and trailing whitespace,
which a template may trim) between two code points absent from the first rendering; that second rendering minus the
two code points must equal the first, else template_error, so the template's structure and branches and every real
image part are today's. The marked rendering is cut at the code points: text outside a marked span is encoded with
`encode` as above (template tokens and image placeholders keep their ids); each marked span, as the template rendered
it (JSON-escaped inside a tool), is encoded alone by the tokenizer's own pipeline without its added tokens
(`tokenizers.Tokenizer.from_str` of the backend's JSON with `added_tokens` emptied), so it is ordinary BPE text and its
ids decode to exactly that span (an added id there is encode_error). Boundary rule: each marked span is a hard token
boundary on both sides; no BPE merge crosses it. The rule reads the strings only, so a message re-encodes to the same
ids on every turn, and `last_message_start` renders the history under it. `stop` strings are not content and keep
the rule above. The SGLang path (`bench-g0`) tokenizes on its own and still maps such text to the added ids: the two
paths give different ids for these requests only.

`tool_param`/`tool_args` run `Qwen3CoderDetector`'s own conversion on the prepared tool (`qwen3_coder_detector.py:63-175`):
`tool_param` returns the streaming fragment `json.dumps(name) + ": " + json.dumps(value, ensure_ascii=False)`
(:356-365, without the ", " separator), `tool_args` the unary `json.dumps({name: value, ...}, ensure_ascii=False)`
(:227); any exception is convert_error, and so is a result that is not valid UTF-8 (a lone surrogate the conversion's
`json.loads` decoded from an escape), which no reply may carry. SGLang's modules (`orjson`, `jsonschema`, `sglang`) are
imported lazily, on the first request that needs them; tool-free requests never import `sglang` (D8). `tools_prepared`
cannot carry one: `orjson` refuses lone surrogates.
`info` hashes use the deployment lock's definitions (docs/CONFIGURATION.md "Deployment identity"): `tokenizer_sha256`
is the SHA-256 of the `sha256sum` lines of merges.txt, tokenizer.json, tokenizer_config.json and vocab.json in bytewise
name order, `template_sha256` the SHA-256 of chat_template.jinja, both over the files in DIR.

Feature `image` (`--feature image`; tasks/m213-spec.md D2, D3, D8, D16-D18, ledger table S): at startup the sidecar
imports torch, torchvision, PIL and SGLang's `load_image`, builds `AutoImageProcessor.from_pretrained(DIR,
local_files_only=True)`, reads `image_token_id` from DIR/config.json and opens the CUDA device `cuda:0` (any failure
exits 2); `info` gains `preprocessor_sha256` (SHA-256 of DIR/preprocessor_config.json). A `user` or `tool` message's
content may then be an array of 1..=64 parts `{type: "text", text}` / `{type: "image_url", image_url: {url, detail?:
"auto"}}`, at most 32 images per request; the URL is a strict base64 `data:image/png` or `data:image/jpeg` URL. Each
image is decoded as SGLang does (`load_image` with GPU decode: nvJPEG for JPEG, else PIL; a PIL image not in mode RGB
is converted to RGB), the template renders `{"type": "image"}` for the part (in a tool message as in a user message,
chat_template.jinja `render_content`), one processor call with `device="cuda:0"` covers all images, and each
`<|image_pad|>` id expands to t*h*w/4 copies. An image reply adds
`"images": [{"grid_thw": [t, h, w], "token_offset", "pixel_bytes", "pad_id"}, ...]` and `"attachment_bytes"`, and is
followed by one raw frame of `attachment_bytes` bytes: each image's `pixel_values` rows cast to bfloat16, little-endian,
concatenated in `images` order. `pad_id` = 1,000,000 + (SGLang `tensor_hash` mod 2^30), the hash being SHA-256[:8]
(big-endian) of the image's processor rows after they move to the CPU, as SGLang's item `pad_value` with CPU feature
transport. Without the feature an array content is invalid_value (param `messages[i].content`). Bad parts or more
images are bad_request, a bad URL or an undecodable image invalid_value (param `messages[i].content[j].image_url.url`),
a processor failure encode_error; an error reply has no attachment frame. `--image-device` picks the processor's
device: `cuda:0` (the default, all of the above) or `cpu` (VisionLong, scripts/live/vision_cpu_processor.py): no CUDA
check and no CUDA context, the processor call with `device="cpu"`, `load_image` without GPU decode (PIL, SGLang's
path without CUDA) and the processor's resize by `CudaResize`, torch's CUDA bicubic antialias kernel reproduced bit
for bit (torchvision's CPU uint8 kernel and torch's CPU float kernel both differ from it on non-identity resizes), so
a PNG's pixel_values are bit-identical to `cuda:0`'s. JPEG is the one named decoder exception (tasks/m213l-spec.md
D-L9, VL-16): cuda:0 decodes it with nvJPEG, cpu with PIL (open, load, RGB conversion iff the mode is not RGB, D18),
deterministic but not nvJPEG's pixels (max abs 0.36 of the normalized values at an identity size); the processor
arithmetic after the decode is the same, and a JPEG is never refused. Everything else is unchanged.
`--image-threads N` (default unset: torch's own thread count) calls `torch.set_num_threads(N)` at startup.
`--image-cache-mib M` (default 0: no cache) memoizes (grid_thw, pad_id, bfloat16 rows) per image under SHA-256(the
base64-decoded image bytes), the device and `preprocessor_sha256`, least recently used first out once the cached rows
exceed M MiB (an image larger than M MiB is never cached); a hit skips decode, processor, hash and cast, the misses
still share one processor call, and the reply and attachment are byte-identical to the uncached ones (the processor's
rows of an image do not depend on the other images of the call: vision_cpu_processor.py `parity`).

Text never leaves the process except as token IDs: before any library loads, file descriptors 1 and 2 point at
/dev/null (library prints and warnings are dropped) and replies go to a private duplicate of the original stdout;
error messages are fixed strings that never quote input. A startup failure prints one line naming the file or
library to (a private duplicate of) stderr and exits 2; a truncated frame at EOF exits 1; EOF on a frame boundary
exits 0. Python stdlib plus the image's transformers/tokenizers (and, lazily, orjson/jsonschema/sglang; with feature
`image`, eagerly torch/torchvision/PIL/sglang; the grammar role also xgrammar, torch and SGLang's grammar modules,
imported at start); nothing from the repository is imported. The grammar role runs with `CUDA_VISIBLE_DEVICES` empty
(set before any import): its masks are CPU work, and the CUDA probes in those imports (tvm_ffi, SGLang's platform,
transformers' flash-attention check) otherwise initialise the driver beside cered's load of the same card, about a
second of the grammar start (tasks/beat-sglang-spec.md L2). The fixtures' oracle (`gen_grammar_fixtures.py`) has
always run that way.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib.metadata
import itertools
import io
import json
import math
import os
import re
import struct
import sys
import warnings
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path

PROTOCOL = "cere-tokenizer-sidecar-v3"
# Always served; ledger table S names (grammar, image, message_boundary, thinking).
FEATURES = ("message_boundary", "thinking")
KNOWN_FEATURES = ("image",)  # ledger table S names `--feature` enables (and advertises), each on request
# SGLang's `_ADDITIONAL_STOP_TOKEN_TEXTS` (`utils/hf_transformers/common.py:626-632`): with the tokenizer's
# `eos_token_id` they are the ids the `min_new_tokens` penalizer suppresses (`penaltylib/min_new_tokens.py:28-36`).
ADDITIONAL_STOP_TOKEN_TEXTS = ("<|eom_id|>", "<|content_model_end_sampling|>")
MAX_FRAME_BYTES = 4 << 20
HEADER = struct.Struct("<I")
TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
TEMPLATE_FILE = "chat_template.jinja"
PREPROCESSOR_FILE = "preprocessor_config.json"
ENCODE_KEYS = frozenset(("op", "messages", "tools_json", "render_tools", "stop", "enable_thinking",
                         "reasoning_effort", "preserve_thinking"))
DRAIN_CHUNK = 1 << 16
GRAMMAR_OPS = frozenset(("grammar_open", "grammar_step", "grammar_drafts", "grammar_close"))
GRAMMAR_FEATURES = ("grammar", "grammar_drafts")  # the grammar role's features beyond FEATURES
OPEN_FIELDS = ("response_format", "regex", "ebnf", "tools_json", "tool_choice")  # raw JSON value text
OPEN_KEYS = frozenset(("op", "parallel_tool_calls", "enable_thinking") + OPEN_FIELDS)
VOCAB_SIZE = 248320  # the model vocabulary SGLang sizes TokenizerInfo and the mask by (m28 design note D7)
# The unfilled row of xgrammar's `allocate_token_bitmask` (every bit set; 248320 = 7760 * 32, no padding bits).
ALL_ONES_ROW = b"\xff" * (VOCAB_SIZE // 8)
ALL_ONES_MASK = base64.b64encode(ALL_ONES_ROW).decode("ascii")
THINK_END_TOKEN = "</think>"  # the bench server's --reasoning-parser qwen3 detector's think_end_token
MAX_GRAMMARS = 4
# `grammar_drafts` checks at most this many drafts: CERE_MAX_CONSTRAINED_DRAFTS (executor/include/cere_executor.h),
# the drafts a constrained ticket's speculative round carries. The matcher's rollback window must hold them.
MAX_DRAFTS = 3
COMPILER_CACHE_BYTES = 256 << 20
TOOL_CALL_PARSER = "qwen3_coder"  # the bench server's --tool-call-parser
DEVICE = "cuda:0"  # D17: SGLang's fast processor runs on cuda:<base_gpu_id>; GPU0 is cuda:0 in the container
DATA_URL_PREFIXES = ("data:image/png;base64,", "data:image/jpeg;base64,")  # D3
MAX_IMAGES = 32  # D4 / VisionLong C2, whole history
MAX_PARTS = 64  # D2
MM_PAD_SHIFT = 1_000_000  # SGLang `schedule_batch.py` MM_PAD_SHIFT_VALUE: pad = shift + (hash mod 2^30)
# N5 (m214 design note, owner decision 2026-10-15): roles whose content is client text, escaped as ordinary text.
COVERED_ROLES = ("system", "user", "tool")


class StartupError(Exception):
    """The artifact or the tokenizer libraries are unusable; exit 2."""


class Truncated(Exception):
    """EOF inside a frame; exit 1."""


class RequestError(Exception):
    """A typed, text-free error reply."""

    def __init__(self, kind: str, message: str, param: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.param = param

    def reply(self) -> dict:
        error = {"type": self.kind, "message": self.message}
        if self.param is not None:
            error["param"] = self.param
        return {"ok": False, "error": error}


def lock_digest(directory: Path) -> tuple[str, str]:
    """(tokenizer_sha256, template_sha256) of `directory` by the deployment lock's definitions: the tokenizer digest
    covers the present members of the tokenizer set (Inkling 2 Mini and Gemma ship no merges.txt/vocab.json)."""
    lines = []
    for name in sorted(TOKENIZER_FILES, key=str.encode):
        if (directory / name).exists():
            lines.append(f"{hashlib.sha256(read_file(directory, name)).hexdigest()}  {name}\n")
    if not lines:
        raise StartupError(f"the artifact has none of {', '.join(TOKENIZER_FILES)}")
    tokenizer = hashlib.sha256("".join(lines).encode()).hexdigest()
    return tokenizer, hashlib.sha256(read_file(directory, TEMPLATE_FILE)).hexdigest()


def read_file(directory: Path, name: str) -> bytes:
    try:
        return (directory / name).read_bytes()
    except OSError as e:
        raise StartupError(f"cannot read {name} in the artifact ({e.strerror})") from None


class Tokenizer:
    """The artifact's tokenizer with SGLang's chat-encoding rules."""

    def __init__(self, directory: Path, thinking_unset: bool = False):
        self.thinking_unset = thinking_unset
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import tokenizers
            import transformers
        except ImportError as e:
            raise StartupError(f"cannot import {e.name}") from None
        self.versions = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
        try:
            self.tok = transformers.AutoTokenizer.from_pretrained(
                str(directory), local_files_only=True, trust_remote_code=False,
                clean_up_tokenization_spaces=False, use_fast=True)
            adds_specials = len(self.tok.encode("")) > 0
            added = {int(i): token.content for i, token in self.tok.added_tokens_decoder.items()}
            # N5: the tokenizer's own pipeline without its added tokens encodes escaped text (module docstring).
            spec = json.loads(self.tok.backend_tokenizer.to_str())
            spec.update(added_tokens=[], truncation=None, padding=None)
            self.plain = tokenizers.Tokenizer.from_str(json.dumps(spec))
        except Exception as e:  # noqa: BLE001 - any loader failure stops the sidecar before it serves
            raise StartupError(f"cannot load the tokenizer from the artifact ({type(e).__name__})") from None
        self.encode_kwargs = {"add_special_tokens": False} if adds_specials else {}
        self.added_ids = frozenset(added)
        texts = sorted({t for t in added.values() if t}, key=len, reverse=True)
        self.added_text = re.compile("|".join(map(re.escape, texts))) if texts else None

    def min_tokens_suppress_ids(self) -> list:
        """The sorted ids SGLang's `min_new_tokens` penalizer suppresses besides the request's `stop_token_ids`
        (tasks/m212-spec.md D10): the tokenizer's `eos_token_id` plus its additional stop tokens."""
        try:
            added = self.tok.get_added_vocab()
            ids = {added[text] for text in ADDITIONAL_STOP_TOKEN_TEXTS if text in added}
            if self.tok.eos_token_id is not None:
                ids.add(self.tok.eos_token_id)
        except Exception as e:  # noqa: BLE001 - any failure stops the sidecar before it serves
            raise StartupError(f"cannot read the tokenizer's stop tokens ({type(e).__name__})") from None
        if not all(type(t) is int and t >= 0 for t in ids):
            raise StartupError("the tokenizer's stop tokens are not token IDs")
        return sorted(ids)

    def render(self, messages: list, tools: list | None, kwargs: dict,
               generation: bool = True) -> tuple[str, list | None]:
        """`kwargs`: `enable_thinking` (always), `reasoning_effort` and `preserve_thinking` (only when sent), SGLang's
        template kwargs (`serving_chat.py:1295-1315`, tasks/m212-spec.md D4, D13, tasks/api-lenient-spec.md AL-2).
        Returns the text and the tools it rendered (`tools` or their flat form); `generation` is
        `add_generation_prompt`."""
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation,
                                                tools=tools, return_dict=False, **kwargs), tools
        except Exception:  # noqa: BLE001 - SGLang retries once with flat function-only tools (serving_chat.py:1346)
            if not tools:
                raise
        flat = [t["function"] if "function" in t else t for t in tools]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, tools=flat,
                                            return_dict=False, **kwargs), flat

    def encode_chat(self, messages: list, tools: list | None, kwargs: dict) -> tuple[list, list | None]:
        """The prompt ids and the tools the template rendered them with."""
        return self.encode_prompt(messages, tools, kwargs, generation=True)

    def encode_prompt(self, messages: list, tools: list | None, kwargs: dict,
                      generation: bool) -> tuple[list, list | None]:
        """Render and encode; covered strings holding added-token text are escaped (N5, module docstring)."""
        try:
            rendered, used = self.render(messages, tools, kwargs, generation)
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("template_error", "the chat template failed to render") from None
        if not isinstance(rendered, str):
            raise RequestError("template_error", "the chat template did not render a string")
        if self.added_text is not None and any(self.added_text.search(s) for s in covered_strings(messages, used)):
            return self.escaped(rendered, messages, used, kwargs, generation), used
        try:
            ids = self.tok.encode(rendered, **self.encode_kwargs)
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("encode_error", "encoding the rendered prompt failed") from None
        return token_ids(ids), used

    def escaped(self, rendered: str, messages: list, tools: list | None, kwargs: dict, generation: bool) -> list:
        """N5: the ids of `rendered` (today's text) with every covered string that holds added-token text encoded as
        ordinary text. A second render marks each such string's core with two code points absent from `rendered`;
        it must reproduce `rendered` once they are removed. Outside the marks: `encode` as today; each marked span
        alone through `plain` (no added tokens), a hard token boundary on both sides."""
        taken = set(rendered)
        free = list(itertools.islice((chr(c) for c in range(0xE000, 0x110000) if chr(c) not in taken), 2))
        if len(free) < 2:
            raise RequestError("template_error", "no code points are left to mark escaped text")
        start, end = free

        def wrap(text: str) -> str:
            if not self.added_text.search(text):
                return text
            head = len(text) - len(text.lstrip())
            core = text.strip()
            return text[:head] + start + core + end + text[head + len(core):]

        try:
            marked = self.tok.apply_chat_template(
                [marked_message(m, wrap) for m in messages], tokenize=False, add_generation_prompt=generation,
                tools=None if tools is None else marked_json(tools, wrap), return_dict=False, **kwargs)
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("template_error", "the chat template failed to render") from None
        pieces = marked.split(start) if isinstance(marked, str) else []
        outside, spans = pieces[:1], []
        for piece in pieces[1:]:
            span, sep, rest = piece.partition(end)
            spans.append(span)
            outside.append(rest if sep else None)
        if (not pieces or None in outside or any(end in text for text in outside)
                or "".join(text + span for text, span in zip(outside, spans + [""])) != rendered):
            raise RequestError("template_error", "the chat template rendered escaped text inconsistently")
        ids = []
        try:
            for i, text in enumerate(outside):
                if text:
                    ids += token_ids(self.tok.encode(text, **self.encode_kwargs))
                if i < len(spans):
                    plain = token_ids(self.plain.encode(spans[i], add_special_tokens=False).ids)
                    if self.added_ids.intersection(plain):
                        raise RequestError("encode_error", "escaped text encoded to an added token")
                    ids += plain
        except RequestError:
            raise
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("encode_error", "encoding the rendered prompt failed") from None
        return ids

    def last_message_start(self, messages: list, tools: list | None, kwargs: dict, ids: list) -> int:
        """Feature `message_boundary` (tasks/m210-spec.md D4 (c)): the common prefix length of `ids` and the ids of
        `messages[:-1]` under the same kwargs, `tools` (those `ids` were rendered with) and N5 escape rule, without
        the generation prompt; 0 for one message or when the history does not render or encode (e.g. a template
        that refuses a history without a user query)."""
        if len(messages) < 2:
            return 0
        try:
            history, _ = self.encode_prompt(messages[:-1], tools, kwargs, generation=False)
        except Exception:  # noqa: BLE001 - a hint only: no boundary, the request is unaffected
            return 0
        common = 0
        for a, b in zip(history, ids):
            if a != b:
                break
            common += 1
        return common

    def stop_token_lens(self, stop: list) -> list:
        """SGLang's per-stop token count (`sampling_params.py:218-228`)."""
        try:
            return [len(token_ids(self.tok.encode(s, add_special_tokens=False))) for s in stop]
        except RequestError:
            raise
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("encode_error", "encoding a stop string failed") from None


class InklingProcessor:
    """Feature `image` on an Inkling 2 artifact (tasks/inkling-spec.md 3.6, S7): transformers 5.24.0's
    `Inkling2ImageProcessor` (image_processing_inkling2.py, TorchvisionBackend) ported to torch on the CPU, bit for
    bit (scripts/native/inkling_reference/run_vision.py checks it against the reference's processor). No resize: a PIL
    image keeps its alpha (`convert("RGBA")` when it has transparency data, else `convert("RGB")`), an RGBA image is
    composited onto the 8-px checkerboard (204 / 153) in integers, the image is padded at the bottom and right to whole
    40-px cells with -1 on the 0-255 scale, then (x - 127.5) / 127.5 in FP32 (the fused rescale + normalize:
    float32(0.5) * (1 / rescale_factor) for mean and std, then `sub_`, `div_`). One cell is one image token (rows *
    cols placeholders). The rows it returns are the vision tower's patch-embedding input: each cell's 2 x 2 native
    20-px patches (block order), each patch 20 x 20 x 3 (y, x, channel) = 1,200 values, i.e. the processor's
    `pixel_values [N, 40, 40, 3]` reshaped as `Inkling2VisionEmbeddings.forward` does, and the grid is counted in
    patches: [1, 2 * rows, 2 * cols], so t * h * w / merge^2 = rows * cols tokens as for every image processor."""

    PROCESSOR_TYPE = "Inkling2ImageProcessor"
    CONFIG = {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5], "rescale_factor": 0.00392156862745098, "size": {"height": 40, "width": 40}}
    CELL = 40
    PATCH = 20
    merge_size = 2

    def __init__(self, torch, config: dict):
        if {k: config.get(k) for k in self.CONFIG} != self.CONFIG:
            raise StartupError("the artifact's Inkling2ImageProcessor config is not the pinned one")
        self.torch = torch
        mean = torch.tensor(config["image_mean"]) * (1.0 / config["rescale_factor"])
        std = torch.tensor(config["image_std"]) * (1.0 / config["rescale_factor"])
        self.mean, self.std = mean[:, None, None], std[:, None, None]

    def rgb(self, image):
        """uint8 [3, H, W] of a PIL image: `process_image` + `convert_to_rgb` of image_processing_inkling2.py."""
        torch = self.torch
        mode = "RGBA" if image.has_transparency_data else "RGB"
        image = image.convert(mode)
        width, height = image.size
        x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, len(mode))
        x = x.permute(2, 0, 1)
        if len(mode) == 3:
            return x.contiguous()
        rows = torch.arange(height)[:, None] // 8
        columns = torch.arange(width)[None, :] // 8
        background = torch.where((rows + columns) % 2 == 0, 204, 153)
        rgba = x.to(torch.int32)
        alpha = rgba[3:4]
        return ((rgba[:3] * alpha + background * (255 - alpha) + 127) // 255).to(torch.uint8)

    def cells(self, image):
        """(FP32 [rows * cols, 40, 40, 3] = the reference's `pixel_values` of one image, rows, cols)."""
        torch, cell = self.torch, self.CELL
        x = self.rgb(image)
        channels, height, width = x.shape
        rows, cols = -(-height // cell), -(-width // cell)
        padded = torch.full((channels, rows * cell, cols * cell), -1.0, dtype=torch.float32)
        padded[:, :height, :width] = x
        padded = padded.sub_(self.mean).div_(self.std)
        return padded.reshape(channels, rows, cell, cols, cell).permute(1, 3, 2, 4, 0).flatten(0, 1), rows, cols

    def patch_rows(self, cells):
        """FP32 [4 N, 1,200]: the patch-embedding rows of [N, 40, 40, 3] cells (Inkling2VisionEmbeddings.forward)."""
        merge, patch = self.merge_size, self.PATCH
        return cells.reshape(-1, merge, patch, merge, patch, 3).transpose(2, 3).reshape(-1, 3 * patch * patch)

    def __call__(self, images, return_tensors="pt", device="cpu"):
        del return_tensors, device  # CPU arithmetic only; the result does not depend on a device
        rows, grids = [], []
        for image in images:
            cells, r, c = self.cells(image)
            rows.append(self.patch_rows(cells))
            grids.append([1, self.merge_size * r, self.merge_size * c])
        return {"pixel_values": self.torch.cat(rows), "image_grid_thw": self.torch.tensor(grids)}


def inkling_processor_config(directory: Path) -> dict | None:
    """The `image_processor` section of DIR/processor_config.json when it names Inkling2ImageProcessor, else None."""
    path = directory / "processor_config.json"
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_bytes())["image_processor"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(config, dict) or config.get("image_processor_type") != InklingProcessor.PROCESSOR_TYPE:
        return None
    return config


def inkling_image_token_id(directory: Path) -> int:
    """Inkling2Processor's placeholder id: `tokenizer.convert_tokens_to_ids(image_token)` of processor_config.json
    (`<|unused_200054|>`, an added token of tokenizer.json)."""
    try:
        token = json.loads(read_file(directory, "processor_config.json"))["image_token"]
        added = json.loads(read_file(directory, "tokenizer.json"))["added_tokens"]
        ids = [t["id"] for t in added if t["content"] == token]
    except (ValueError, KeyError, TypeError):
        ids = []
    if len(ids) != 1 or type(ids[0]) is not int:
        raise StartupError("the artifact's tokenizer.json lacks the processor's image token")
    return ids[0]


class CudaResize:
    """`--image-device cpu`: torchvision's `resize` of a uint8 image with BICUBIC and antialias on CUDA, bit for bit on
    the CPU. There torchvision casts to float32, runs torch's `upsample_gen2d_aa_out_frame` (UpSampleBilinear2d.cu:
    per output a horizontal pass over its input rows, then a vertical pass over those; each pass `t0 * w0`, then one
    fused multiply-add per tap), then `clamp_(0, 255)`, `round_()` and the cast back. The weights are UpSample.cuh's
    `_compute_weights_span` and `_compute_weights` in float32 as nvcc compiles them: `scale * (i + 0.5)` fused into
    each use (`center -/+ support`, `xmin - center`), the filter's multiply-adds fused. A dimension that keeps its
    size is exact in the kernel (weights -0, 1, -0, 0) and skipped here.

    Each output comes from a float32 pass with those weights in any rounding order; an output within the error bound
    of a rounding boundary (n + 0.5) is then recomputed in the kernel's operation order, its float32 fmas emulated in
    float64 (the product is exact) with a correction where the float64 sum lands on a float32 tie. The bound: k taps
    of sequential float32 operations are within gamma_k = k u / (1 - k u) (u = 2^-24) times sum |t w| of the exact
    sum, for the kernel and the approximation alike; inputs in [0, 255], S the largest sum |w| of a pass, so
    |approximation - kernel| <= 2 * 255 * Sw * Sh * (gamma_kw + gamma_kh) plus second-order terms; twice that is the
    margin."""

    U = 2.0 ** -24
    LOW = (1 << 29) - 1  # the float64 mantissa bits below float32's
    TIE = 1 << 28
    PLANS = 256  # cached (size, out) weight plans; the cache restarts when full

    def __init__(self, torch):
        self.torch = torch
        self.plans = {}

    def fma(self, a, b, c):
        """float32 fma(a, b, c) elementwise (broadcasting; c a tensor or a float)."""
        torch = self.torch
        p = a.double() * b.double()
        c = c.double() if torch.is_tensor(c) else torch.tensor(c, dtype=torch.float64)
        s = p + c
        r = s.float()
        tie = (s.view(torch.int64) & self.LOW) == self.TIE
        if bool(tie.any()):  # the float64 sum rounded onto a float32 tie: round by the sign of its error
            bb = s - p
            err = (p - (s - bb)) + (c - bb)
            rd = r.double()
            r = torch.where(tie & (err > 0) & (rd < s), torch.nextafter(r, torch.full_like(r, math.inf)), r)
            r = torch.where(tie & (err < 0) & (rd > s), torch.nextafter(r, torch.full_like(r, -math.inf)), r)
        return r

    def weights(self, size: int, out: int) -> tuple:
        """(input index [out, k], float32 weights [out, k]; taps past an output's xsize have weight 0)."""
        torch = self.torch
        f = torch.float32
        scale = torch.tensor(float(size), dtype=f) / torch.tensor(float(out), dtype=f)  # area_pixel_compute_scale
        up = float(scale) < 1.0
        support = torch.tensor(2.0, dtype=f) * (1.0 if up else scale)  # (interp_size * 0.5) * scale
        invscale = torch.tensor(1.0 if up else 1.0 / float(scale), dtype=torch.float64).float()
        sc = scale.expand(out)
        half = torch.arange(out, dtype=f) + 0.5
        xmin = torch.clamp((self.fma(sc, half, (-support).expand(out)) + 0.5).trunc().long(), min=0)
        xsize = torch.clamp((self.fma(sc, half, support.expand(out)) + 0.5).trunc().long(), max=size) - xmin
        xmin_m_center = self.fma(-sc, half, xmin.to(f))
        k = int(xsize.max())
        x = ((torch.arange(k, dtype=f)[None, :] + xmin_m_center[:, None]) + 0.5) * invscale
        x = x.abs()
        near = self.fma(self.fma(torch.full_like(x, 1.5), x, -2.5) * x, x, 1.0)  # ((a + 2) x - (a + 3)) x x + 1
        far = self.fma(self.fma(x - 5.0, x, 8.0), x, -4.0) * -0.5  # (((x - 5) x + 8) x - 4) a, a = -0.5
        valid = torch.arange(k)[None, :] < xsize[:, None]
        w = torch.where(valid & (x < 1), near, torch.where(valid & (x < 2), far, torch.zeros_like(x)))
        total = torch.zeros(out, dtype=f)
        for j in range(k):  # total_w += w, in tap order
            total = torch.where(valid[:, j], total + w[:, j], total)
        w = torch.where(valid & (total != 0)[:, None], w / total[:, None], w)
        return torch.clamp(xmin[:, None] + torch.arange(k)[None, :], max=size - 1), w

    def plan(self, size: int, out: int) -> tuple:
        """(index, weights, gamma_k, largest sum |w|) of one dimension; the identity is (arange, 1, 0, 1)."""
        torch = self.torch
        if size == out:
            return torch.arange(size)[:, None], torch.ones(size, 1), 0.0, 1.0
        plan = self.plans.get((size, out))
        if plan is None:
            if len(self.plans) >= self.PLANS:
                self.plans.clear()
            index, w = self.weights(size, out)
            k = w.shape[1]
            plan = self.plans[(size, out)] = (index, w, k * self.U / (1 - k * self.U), float(w.abs().sum(1).max()))
        return plan

    def __call__(self, image, height: int, width: int):
        """`image` [..., C, H, W] uint8 resized to height x width, as torchvision on CUDA."""
        torch = self.torch
        shape = image.shape
        h, w = shape[-2:]
        if (h, w) == (height, width):
            return image
        image = image.reshape(-1, *shape[-3:])
        iw, ww, gw, sw = self.plan(w, width)
        ih, wh, gh, sh = self.plan(h, height)
        x = image.to(torch.float32)
        if w != width:
            x = self.approximate(x.transpose(-1, -2).contiguous(), iw, ww).transpose(-1, -2).contiguous()
        if h != height:
            x = self.approximate(x, ih, wh)
        x = x.clamp_(0, 255)
        near = ((x - x.floor()) - 0.5).abs_() <= 4 * 255 * sw * sh * (gw + gh)
        out = x.round_().to(torch.uint8)
        flagged = near.nonzero()
        if flagged.shape[0]:
            b, c, oy, ox = flagged.unbind(1)
            rows, cols, wx, wy = ih[oy], iw[ox], ww[ox], wh[oy]
            src = image[b[:, None, None], c[:, None, None], rows[:, :, None], cols[:, None, :]].to(torch.float32)
            line = src[..., 0] * wx[:, None, 0]  # the horizontal pass at each of the output's input rows
            for j in range(1, cols.shape[1]):
                line = self.fma(src[..., j], wx[:, None, j], line)
            y = line[:, 0] * wy[:, 0]
            for j in range(1, rows.shape[1]):
                y = self.fma(line[:, j], wy[:, j], y)
            out[b, c, oy, ox] = y.clamp_(0, 255).round_().to(torch.uint8)
        return out.reshape(*shape[:-2], height, width)

    @staticmethod
    def approximate(x, index, w):
        """sum_j x[..., index[:, j], :] * w[:, j] along dim -2, float32, any rounding."""
        acc = x.index_select(-2, index[:, 0]) * w[:, 0, None]
        for j in range(1, index.shape[1]):
            acc.addcmul_(x.index_select(-2, index[:, j]), w[:, j, None])
        return acc


class Images:
    """Feature `image` (tasks/m213-spec.md D8, D16-D18): SGLang's own decoder and the artifact's image processor on
    `device` (`cuda:0`, or `cpu` with `--image-device cpu`), all imported and built at startup so the synchronous
    sidecar never stalls on a first image request."""

    def __init__(self, directory: Path, device: str = DEVICE, threads: int = 0, cache_mib: int = 0):
        inkling = inkling_processor_config(directory)
        self.inkling = inkling is not None
        try:
            import PIL.Image  # noqa: F401 - SGLang's loader opens PNGs with it
            import torch
            if not self.inkling:  # InklingProcessor needs only PIL and torch
                import torchvision  # noqa: F401 - the fast processor's resize and the nvJPEG decoder
                import transformers
                from sglang.srt.utils.common import load_image
                self.load_image = load_image
        except ImportError as e:
            raise StartupError(f"cannot import {e.name}") from None
        self.torch = torch
        self.pil = PIL.Image
        config_file = "processor_config.json" if self.inkling else PREPROCESSOR_FILE
        self.preprocessor_sha256 = hashlib.sha256(read_file(directory, config_file)).hexdigest()
        try:
            image_token_id = json.loads(read_file(directory, "config.json")).get("image_token_id")
        except (ValueError, AttributeError):
            image_token_id = None
        if image_token_id is None and self.inkling:
            image_token_id = inkling_image_token_id(directory)
        if type(image_token_id) is not int or image_token_id < 0:
            raise StartupError("config.json in the artifact has no image_token_id")
        self.image_token_id = image_token_id
        self.device = device
        if threads:
            torch.set_num_threads(threads)
        self.cache_bytes = cache_mib << 20
        self.cache = OrderedDict()  # (raw SHA-256, device, preprocessor_sha256) -> (grid, pad_id, bf16 view)
        self.cached_bytes = 0
        self.processor_calls = 0
        if self.inkling:
            # CPU arithmetic, no resize: the same rows on any --image-device, no CUDA context needed.
            self.processor = InklingProcessor(torch, inkling)
            self.merge = InklingProcessor.merge_size
            return
        if device != "cpu" and not torch.cuda.is_available():
            raise StartupError(f"feature image needs the CUDA device {device}")
        try:
            self.processor = transformers.AutoImageProcessor.from_pretrained(str(directory), local_files_only=True)
            self.merge = int(self.processor.merge_size)
        except Exception as e:  # noqa: BLE001 - any loader failure stops the sidecar before it serves
            raise StartupError(f"cannot load the image processor from the artifact ({type(e).__name__})") from None
        if device == "cpu":
            self.install_cuda_resize(torch)
        else:
            try:
                torch.zeros(1, device=device)  # the CUDA context, before the first request
            except Exception as e:  # noqa: BLE001
                raise StartupError(f"cannot initialise the CUDA device {device} ({type(e).__name__})") from None

    def install_cuda_resize(self, torch) -> None:
        """`--image-device cpu`: the processor resizes a uint8 image with BICUBIC and antialias to a height x width by
        `CudaResize` (any other call keeps torchvision's); a processor that does not resample BICUBIC refuses."""
        def bicubic(resample) -> bool:
            return getattr(resample, "value", resample) in (3, "bicubic")  # PILImageResampling, InterpolationMode

        if not bicubic(getattr(self.processor, "resample", None)):
            raise StartupError("--image-device cpu needs an image processor that resamples BICUBIC")
        exact, torchvision_resize = CudaResize(torch), self.processor.resize

        def resize(image, size, resample=None, antialias=True, **kwargs):
            if (antialias and bicubic(resample) and image.dtype == torch.uint8 and not size.shortest_edge
                    and not (size.max_height and size.max_width) and size.height and size.width):
                return exact(image, size.height, size.width)
            return torchvision_resize(image, size, resample, antialias, **kwargs)

        self.processor.resize = resize

    def decode(self, raw: bytes, param: str):
        """`base_processor.py` `_load_single_item`: SGLang's `load_image` (nvJPEG for JPEG on a CUDA device, else
        PIL), then a PIL image is loaded and converted to RGB iff its mode is not RGB (alpha dropped, not composited;
        D18). Inkling: PIL only and the mode kept (InklingProcessor composites the alpha as the reference does)."""
        try:
            if self.inkling:
                image = self.pil.open(io.BytesIO(raw))
                image.load()
                return image
            image, _ = self.load_image(raw, self.device != "cpu")
            if isinstance(image, self.torch.Tensor):
                return image
            if image.mode != "RGB":
                return image.convert("RGB")
            image.load()
            return image
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("invalid_value", "the image could not be decoded", param) from None

    def process(self, images: list) -> list:
        """[(grid_thw, pad_id, bf16 bytes)] per image, in order. One processor call over all images with SGLang's
        `device` (D17); the features move to the CPU as `base_processor.py:713-724` does for `mm_feature_transport`
        `cpu` (cere-ctl-g0 on one node), so the item hash is `mm_utils.tensor_hash`'s CPU SHA-256 path over that
        image's rows as the processor returned them (D16); the attachment is those rows `.to(torch.bfloat16)`, the
        cast `qwen3_vl.py` applies before the patch embedding (D8)."""
        torch = self.torch
        try:
            self.processor_calls += 1
            out = self.processor(images=images, return_tensors="pt", device=self.device)
            pixels = out["pixel_values"].to("cpu")
            grids = out["image_grid_thw"].tolist()
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("encode_error", "preprocessing the images failed") from None
        if len(grids) != len(images) or not all(len(g) == 3 and all(type(v) is int and v > 0 for v in g)
                                                for g in grids):
            raise RequestError("encode_error", "the image processor returned an unexpected grid")
        if sum(t * h * w for t, h, w in grids) != pixels.shape[0]:
            raise RequestError("encode_error", "the image processor returned an unexpected number of patches")
        result, start = [], 0
        for grid in grids:
            rows = pixels[start:start + grid[0] * grid[1] * grid[2]].contiguous()
            start += grid[0] * grid[1] * grid[2]
            digest = hashlib.sha256(memoryview(rows.reshape(-1).view(torch.uint8).numpy())).digest()
            pad_id = MM_PAD_SHIFT + int.from_bytes(digest[:8], "big") % (1 << 30)
            bf16 = rows.to(torch.bfloat16).contiguous().reshape(-1).view(torch.uint8).numpy()
            result.append((grid, pad_id, memoryview(bf16)))
        return result

    def prepare(self, raw: bytes, param: str) -> tuple:
        """(cache key or None, cached (grid, pad_id, bf16) or None, decoded image or None): a cache hit skips
        decode; without the cache every image decodes, as before."""
        if not self.cache_bytes:
            return None, None, self.decode(raw, param)
        key = (hashlib.sha256(raw).digest(), self.device, self.preprocessor_sha256)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            return key, hit, None
        return key, None, self.decode(raw, param)

    def process_prepared(self, prepared: list) -> list:
        """`process` over the prepared images that missed the cache (one call), cached entries in their place."""
        misses = [image for _, hit, image in prepared if hit is None]
        fresh = iter(self.process(misses) if misses else ())
        result = []
        for key, hit, _ in prepared:
            if hit is None:
                hit = next(fresh)
                if key is not None:
                    self.remember(key, hit)
            result.append(hit)
        return result

    def remember(self, key: tuple, entry: tuple) -> None:
        size = entry[2].nbytes
        if size > self.cache_bytes:
            return
        old = self.cache.pop(key, None)
        if old is not None:
            self.cached_bytes -= old[2].nbytes
        self.cache[key] = entry
        self.cached_bytes += size
        while self.cached_bytes > self.cache_bytes:
            _, (_, _, view) = self.cache.popitem(last=False)
            self.cached_bytes -= view.nbytes


def token_ids(ids) -> list:
    if not (isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids)):
        raise RequestError("encode_error", "the tokenizer returned something other than token IDs")
    return ids


def orjson_loads(text: str):
    """SGLang's request parser (`http_server.py:433-442`), imported on first use."""
    try:
        import orjson
    except ImportError:
        raise RequestError("reference_unavailable", "orjson cannot be imported") from None
    return orjson.loads(text)


def arguments_of(text: str) -> dict:
    """`serving_chat.py:109-123`: an assistant history call's `arguments` string as a JSON object."""
    try:
        value = orjson_loads(text)
    except RequestError:
        raise
    except Exception:  # noqa: BLE001 - the message would quote the input
        raise RequestError("bad_request", "tool call arguments must be valid JSON") from None
    if not isinstance(value, dict):
        raise RequestError("bad_request", "tool call arguments must be a JSON object")
    return value


class Reference:
    """SGLang's own tool code, imported on the first request that needs it (D8: never for tool-free requests)."""

    _loaded = None

    @classmethod
    def get(cls) -> "Reference":
        if cls._loaded is None:
            try:
                cls._loaded = cls()
            except Exception:  # noqa: BLE001 - a missing or broken library is a typed error, the sidecar lives on
                raise RequestError("reference_unavailable", "the SGLang tool modules cannot be imported") from None
        return cls._loaded

    def __init__(self):
        from jsonschema import Draft202012Validator, SchemaError
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
        from sglang.srt.function_call.utils import normalize_json_schema_types

        self.validator = Draft202012Validator
        self.schema_error = SchemaError
        self.tool = Tool
        self.normalize = normalize_json_schema_types
        self.detector = Qwen3CoderDetector()

    def prepare_tools(self, tools_json: str) -> list | None:
        """`serving_chat.py:856-873,1089`: the request's tools after normalize, as `model_dump()` dicts."""
        try:
            raw = orjson_loads(tools_json)
            if raw is None:
                return None
            if not isinstance(raw, list):
                raise ValueError("tools is not a list")
            tools = [self.tool.model_validate(t) for t in raw]
        except RequestError:
            raise
        except Exception:  # noqa: BLE001 - orjson and pydantic failures; the message would quote the input
            raise RequestError("invalid_value", "tools is not a valid list of function tools", "tools") from None
        for i, tool in enumerate(tools):
            if tool.function.parameters is None:
                continue
            try:
                self.normalize(tool.function.parameters)
                self.validator.check_schema(tool.function.parameters)
            except (self.schema_error, RecursionError):
                raise RequestError("invalid_value", "a tool's parameters are not a valid JSON schema",
                                   f"tools[{i}].function.parameters") from None
        return [tool.model_dump() for tool in tools]

    def config(self, tool_json, func) -> dict:
        """`_get_arguments_config` over the one prepared tool `cered` matched (none -> no schema, raw strings)."""
        if tool_json is None:
            return {}
        try:
            tool = self.tool.model_validate(tool_json)
        except Exception:  # noqa: BLE001 - the message would quote the input
            raise RequestError("bad_request", "tool_json is not a prepared tool") from None
        return self.detector._get_arguments_config(func, [tool])

    def convert(self, name: str, raw: str, config: dict, func):
        return self.detector._convert_param_value(raw, name, config, func)

    def tool_param(self, tool_json, func, name: str, raw: str) -> str:
        """The streaming fragment of one closed parameter (`qwen3_coder_detector.py:356-365`)."""
        config = self.config(tool_json, func)
        try:
            return utf8(f"{json.dumps(name)}: {json.dumps(self.convert(name, raw, config, func), ensure_ascii=False)}")
        except Exception:  # noqa: BLE001 - SGLang's stream fails on any conversion error
            raise RequestError("convert_error", "converting the parameter failed") from None

    def tool_args(self, tool_json, func, params: list) -> str:
        """The unary `arguments` of one call (`qwen3_coder_detector.py:212-229`)."""
        config = self.config(tool_json, func)
        try:
            parsed = {}
            for name, raw in params:
                parsed[name] = self.convert(name, raw, config, func)
            return utf8(json.dumps(parsed, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - SGLang's unary parse falls back to the whole text on any error
            raise RequestError("convert_error", "converting the call's parameters failed") from None


def utf8(text: str) -> str:
    """`text` if it is valid UTF-8; a lone surrogate (the reference's `json.loads` of an escape such as `"\\ud800"`)
    raises UnicodeEncodeError, so the reply never carries an escape `cered`'s JSON parser refuses."""
    text.encode("utf-8")
    return text


def model_eos(directory: Path) -> list:
    """SGLang's `hf_eos_token_id` (`model_config.py:1640-1658`): config.json's `eos_token_id` united with
    generation_config.json's (when that file exists), sorted."""
    ids = set()
    for name, required in (("config.json", True), ("generation_config.json", False)):
        if not required and not (directory / name).exists():
            continue
        try:
            value = json.loads(read_file(directory, name)).get("eos_token_id")
        except (ValueError, AttributeError):
            raise StartupError(f"{name} in the artifact is not a JSON object") from None
        values = value if isinstance(value, list) else [] if value is None else [value]
        if not all(type(v) is int and 0 <= v < VOCAB_SIZE for v in values):
            raise StartupError(f"{name} in the artifact has an invalid eos_token_id")
        ids.update(values)
    if not ids:
        raise StartupError("the artifact names no EOS token")
    return sorted(ids)


# --- number ranges (m28 design note B7 [REVISED: owner 2026-10-18 number bounds]) ---
# xgrammar 0.2.1 compiles `{"type": "number", "minimum": 0, "maximum": 10}` to its unbounded number rule (it accepts
# 11 and 10.5; `grammar_keywords.json` `number_range_variants` `stock_pass`). A `type: "number"` schema whose only
# non-annotation keywords are an inclusive `minimum` and `maximum` therefore compiles to a range rule of its own:
# exactly the JSON numbers whose decimal value lies in [minimum, maximum], with any fraction and, if written with
# one, an exponent whose value is in [-NUMBER_EXPONENT_MAX, NUMBER_EXPONENT_MAX] (`1e1`, `1E+01`, `100e-1`; not
# `0e10`, although its value is 0). Such a pair with minimum > maximum, a non-finite value or a bound of more than
# NUMBER_BOUND_DIGITS digits before or after its decimal point is invalid_value. Every other number bound form
# compiles as SGLang compiles it: cered's walker refuses those (`constraint.rs` `number_range`).
NUMBER_BOUND_DIGITS = 20  # digits a bound may have before, and after, its decimal point
NUMBER_EXPONENT_MAX = 9
NUMBER_RANGE_KEYS = frozenset(("type", "minimum", "maximum", "title", "description", "default"))
DIGITS = "0123456789"


def decimal_digits(value: Decimal) -> tuple:
    """|value| exactly: (integer digits without leading zeros, "0" for none; fraction digits without trailing
    zeros)."""
    _, digits, exp = value.copy_abs().as_tuple()
    text = "".join(map(str, digits))
    if exp >= 0:
        whole, frac = text + "0" * exp, ""
    else:
        text = text.rjust(1 - exp, "0")
        whole, frac = text[:exp], text[exp:]
    return whole.lstrip("0") or "0", frac.rstrip("0")


def shift(value: Decimal, places: int) -> Decimal:
    """value * 10**places exactly (Decimal arithmetic rounds to the context precision)."""
    sign, digits, exp = value.as_tuple()
    return Decimal((sign, digits, exp + places))


# A literal's comparison against one bound (digits as `decimal_digits`): "L"/"G" decided (the literal is below/above
# it), ("i", k, c) after k integer digits whose order against the bound's first k is c (-1, 0, 1), ("f", p) with the
# integer part equal and the first p fraction digits equal (p capped at the bound's fraction length).
def compare_int_digit(state, bound: tuple, d: str):
    if state == "G":
        return "G"
    _, k, c = state
    if k == len(bound[0]):
        return "G"  # more integer digits than the bound, none of them a leading zero
    return "i", k + 1, c or (d > bound[0][k]) - (d < bound[0][k])


def compare_int_end(state, bound: tuple):
    if state == "G":
        return "G"
    _, k, c = state
    if k < len(bound[0]) or c < 0:
        return "L"
    return "G" if c > 0 else ("f", 0)


def compare_frac_digit(state, bound: tuple, d: str):
    if state in ("L", "G"):
        return state
    p, frac = state[1], bound[1]
    b = frac[p] if p < len(frac) else "0"
    if d != b:
        return "L" if d < b else "G"
    return "f", min(p + 1, len(frac))


def compare_final(state, bound: tuple) -> str:
    if state in ("L", "G"):
        return state
    return "L" if state[1] < len(bound[1]) else "E"  # the bound's remaining fraction digits end in a non-zero one


def mantissa_dfa(lo: Decimal, hi: Decimal) -> dict:
    """The DFA of JSON mantissas `-?(0|[1-9][0-9]*)(\\.[0-9]+)?` whose value lies in [lo, hi]: state -> (accepting,
    {char: state}) from start state "T", keeping only the states from which an accepting one is reachable. A state
    is (sign, syntax, lower comparison, upper comparison) with syntax S start, Z integer part "0", I a longer integer
    part, D after ".", F in the fraction; "-" compares the magnitude against [max(-hi, 0), -lo] (so "-0" is 0)."""
    zero, ranges = Decimal(0), {}
    if hi >= 0:
        ranges["+"] = (max(lo, zero), hi)
    if lo <= 0:
        ranges["-"] = (max(hi.copy_negate(), zero), lo.copy_negate())
    bounds = {sign: (decimal_digits(a), decimal_digits(b)) for sign, (a, b) in ranges.items()}
    first = ("i", 0, 0)

    def edges(state) -> dict:
        if state == "T":
            out = edges(("+", "S", first, first)) if "+" in bounds else {}
            if "-" in bounds:
                out["-"] = ("-", "S", first, first)
            return out
        sign, syntax, a, b = state
        low, high = bounds[sign]
        out = {}
        if syntax in ("S", "I"):
            for d in DIGITS[syntax == "S":]:
                out[d] = (sign, "I", compare_int_digit(a, low, d), compare_int_digit(b, high, d))
        if syntax == "S":
            out["0"] = (sign, "Z", compare_int_digit(a, low, "0"), compare_int_digit(b, high, "0"))
        if syntax in ("Z", "I"):
            out["."] = (sign, "D", compare_int_end(a, low), compare_int_end(b, high))
        if syntax in ("D", "F"):
            for d in DIGITS:
                out[d] = (sign, "F", compare_frac_digit(a, low, d), compare_frac_digit(b, high, d))
        return out

    def accepting(state) -> bool:
        if state == "T" or state[1] in ("S", "D"):
            return False
        sign, syntax, a, b = state
        low, high = bounds[sign]
        if syntax != "F":
            a, b = compare_int_end(a, low), compare_int_end(b, high)
        return compare_final(a, low) != "L" and compare_final(b, high) != "G"

    dfa, todo = {}, ["T"]
    while todo:
        state = todo.pop()
        if state not in dfa:
            dfa[state] = (accepting(state), edges(state))
            todo.extend(dfa[state][1].values())
    live = {s for s, (acc, _) in dfa.items() if acc}
    grew = True
    while grew:
        grew = False
        for s, (_, out) in dfa.items():
            if s not in live and any(t in live for t in out.values()):
                live.add(s)
                grew = True
    return {s: (acc, {c: t for c, t in out.items() if t in live}) for s, (acc, out) in dfa.items() if s in live}


def exponent_ebnf(e: int) -> str:
    """An exponent part whose value is `e` (leading zeros allowed, as JSON allows them)."""
    if e == 0:
        return '[eE] [+\\-]? "0"+'
    return '[eE] ' + ('"+"?' if e > 0 else '"-"') + f' "0"* "{abs(e)}"'


def char_class(chars: list) -> str:
    if chars in (["."], ["-"]):
        return f'"{chars[0]}"'
    ds, runs, i = sorted(chars), [], 0
    while i < len(ds):
        j = i
        while j + 1 < len(ds) and ord(ds[j + 1]) == ord(ds[j]) + 1:
            j += 1
        runs.append(ds[i] if i == j else f"{ds[i]}-{ds[j]}")
        i = j + 1
    return "[" + "".join(runs) + "]"


def number_range_rules(lo: Decimal, hi: Decimal, name: str) -> list:
    """EBNF rules: `name` matches exactly the JSON numbers whose decimal value lies in [lo, hi] (lo <= hi) and whose
    exponent, if any, has a value in [-NUMBER_EXPONENT_MAX, NUMBER_EXPONENT_MAX]. A number m·10^e is the mantissa DFA
    of [lo·10^-e, hi·10^-e] followed by an exponent of value e; the DFAs of every e are minimized together (Moore) and
    each state becomes one rule `<name>_<n>`."""
    graph, exponents = {}, []
    for e in range(-NUMBER_EXPONENT_MAX, NUMBER_EXPONENT_MAX + 1):
        dfa = mantissa_dfa(shift(lo, -e), shift(hi, -e))
        if "T" in dfa:
            graph.update({(e, s): (acc, {c: (e, t) for c, t in out.items()}) for s, (acc, out) in dfa.items()})
            exponents.append(e)
    block = {s: int(acc) for s, (acc, _) in graph.items()}
    while True:
        signature = {s: (block[s], tuple(sorted((c, block[t]) for c, t in out.items())))
                     for s, (_, out) in graph.items()}
        ids: dict = {}
        refined = {s: ids.setdefault(k, len(ids)) for s, k in sorted(signature.items(), key=lambda kv: repr(kv[1]))}
        if len(ids) == len(set(block.values())):
            break
        block = refined
    rules, emitted = [], set()
    for s in sorted(graph, key=repr):
        if block[s] in emitted:
            continue
        emitted.add(block[s])
        acc, out = graph[s]
        targets: dict = {}
        for c, t in out.items():
            targets.setdefault(block[t], []).append(c)
        alternatives = ['""'] * acc + [f"{char_class(chars)} {name}_{t}" for t, chars in sorted(targets.items())]
        rules.append(f"{name}_{block[s]} ::= (" + " | ".join(f"({a})" for a in alternatives) + ")")
    top = [f'{name}_{block[(e, "T")]} ' + (f'("" | {exponent_ebnf(0)})' if e == 0 else exponent_ebnf(e))
           for e in exponents]
    return [f"{name} ::= (" + " | ".join(f"({a})" for a in top) + ")"] + rules


def number_range(schema: dict):
    """(minimum, maximum) as exact decimals when `schema` is a `type: "number"` schema bounded by exactly an
    inclusive `minimum` and `maximum` (plus annotations), else None."""
    if schema.get("type") != "number" or not ("minimum" in schema and "maximum" in schema):
        return None
    if not schema.keys() <= NUMBER_RANGE_KEYS:
        return None
    values = [schema["minimum"], schema["maximum"]]
    if not all(type(v) is int or (type(v) is float and math.isfinite(v)) for v in values):
        raise RequestError("invalid_value", "number bounds must be finite numbers")
    lo, hi = (Decimal(repr(v)) if type(v) is float else Decimal(v) for v in values)
    if lo > hi or any(len(part) > NUMBER_BOUND_DIGITS for v in (lo, hi) for part in decimal_digits(v)):
        raise RequestError("invalid_value", f"number bounds need minimum <= maximum, each with at most "
                                            f"{NUMBER_BOUND_DIGITS} digits before and after the decimal point")
    return lo, hi


def number_range_ebnf(key: str, from_json_schema):
    """The EBNF of json_schema key `key` with every bounded-number schema (in the positions cered's walker visits)
    compiled to its range rule, or None when the key has none, so it compiles as SGLang's backend compiles it. Each
    such schema becomes a `const` sentinel string (`cere_number_range_<key digest>_<i>`, absent from the key) for
    `from_json_schema` (xgrammar's converter, `any_whitespace=True`); every sentinel literal in its EBNF is then
    replaced by the range rule's name."""
    tag = "cere_number_range_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    ranges: list = []

    def visit(schema):
        if not isinstance(schema, dict):
            return schema
        bounds = number_range(schema)
        if bounds is not None:
            ranges.append(bounds)
            return {"const": f"{tag}_{len(ranges) - 1}"}
        out = dict(schema)
        for k in ("properties", "$defs"):
            if isinstance(out.get(k), dict):
                out[k] = {n: visit(s) for n, s in out[k].items()}
        for k in ("items", "additionalProperties"):
            if k in out:
                out[k] = visit(out[k])
        for k in ("prefixItems", "anyOf", "allOf"):
            if isinstance(out.get(k), list):
                out[k] = [visit(s) for s in out[k]]
        return out

    schema = visit(json.loads(key))
    if not ranges:
        return None
    if tag in key:
        raise RequestError("invalid_value", "the schema contains the number range sentinel")
    ebnf = str(from_json_schema(json.dumps(schema), any_whitespace=True))
    rules = []
    for i, (lo, hi) in enumerate(ranges):
        name = f"{tag}_{i}"
        literal = f'"\\"{name}\\""'
        uses = ebnf.count(literal)
        if uses != len(re.findall(rf"\b{name}\b", ebnf)):
            raise RequestError("invalid_value", "a bounded number schema did not compile")
        if uses:  # an unreferenced `$defs` entry is not in the grammar
            ebnf = ebnf.replace(literal, name)
            rules += number_range_rules(lo, hi, name)
    return ebnf.rstrip("\n") + "\n" + "\n".join(rules) + "\n"


class Grammar:
    """The grammar role: SGLang's grammar key and xgrammar's masks (tasks/m28-spec.md D1, D7)."""

    def __init__(self, directory: Path, tokenizer: Tokenizer):
        if sys.byteorder != "little":
            raise StartupError("the bitmask encoding needs a little-endian host")
        self.eos = model_eos(directory)
        try:
            import torch
            import xgrammar
            from jsonschema import Draft202012Validator, SchemaError
            from sglang.srt.constrained.utils import is_legacy_structural_tag
            from sglang.srt.constrained.xgrammar_backend import MAX_ROLLBACK_TOKENS, XGrammarGrammarBackend
            from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
            from sglang.srt.function_call.function_call_parser import FunctionCallParser
            from sglang.srt.function_call.utils import get_json_schema_constraint, normalize_json_schema_types
        except ImportError as e:
            raise StartupError(f"cannot import {e.name}") from None
        try:
            xgrammar_version = importlib.metadata.version("xgrammar")
        except importlib.metadata.PackageNotFoundError:
            raise StartupError("cannot read the xgrammar version") from None
        self.versions = {"xgrammar": xgrammar_version, "torch": torch.__version__}
        self.xgr = xgrammar
        self.request = ChatCompletionRequest
        self.parser = FunctionCallParser
        self.json_constraint = get_json_schema_constraint
        self.normalize = normalize_json_schema_types
        self.validator = Draft202012Validator
        self.schema_error = SchemaError
        self.is_legacy = is_legacy_structural_tag
        self.backend = XGrammarGrammarBackend
        self.rollback = MAX_ROLLBACK_TOKENS
        if self.rollback < MAX_DRAFTS:
            raise StartupError(f"grammar_drafts needs max_rollback_tokens >= {MAX_DRAFTS}")
        self.tok = tokenizer.tok
        try:
            info = xgrammar.TokenizerInfo.from_huggingface(self.tok, vocab_size=VOCAB_SIZE, stop_token_ids=self.eos)
            self.compiler = xgrammar.GrammarCompiler(tokenizer_info=info, cache_limit_bytes=COMPILER_CACHE_BYTES)
            self.bitmask = xgrammar.allocate_token_bitmask(1, VOCAB_SIZE)
        except Exception as e:  # noqa: BLE001 - any setup failure stops the sidecar before it serves
            raise StartupError(f"cannot build the xgrammar tokenizer info ({type(e).__name__})") from None
        # managers/scheduler.py:860-875 and reasoner_grammar_backend.py:253-261: the think-end ids are the tokenizer's
        # encoding of the reasoning parser's end token. SGLang only warns and serves ungated when it encodes to
        # nothing; a grammar role that cannot defer a thinking request's grammar refuses to start instead.
        try:
            self.think_end = tuple(token_ids(self.tok.encode(THINK_END_TOKEN, add_special_tokens=False)))
        except Exception:  # noqa: BLE001 - any tokenizer failure here stops the sidecar before it serves
            self.think_end = ()
        if not self.think_end:
            raise StartupError(f"the tokenizer cannot encode {THINK_END_TOKEN}")
        # utils/token_sequence_matcher.py: the KMP prefix lengths of the think-end ids.
        prefix, matched = [0] * len(self.think_end), 0
        for i in range(1, len(self.think_end)):
            while matched and self.think_end[i] != self.think_end[matched]:
                matched = prefix[matched - 1]
            if self.think_end[i] == self.think_end[matched]:
                matched += 1
            prefix[i] = matched
        self.think_prefix = tuple(prefix)
        self.matchers: dict = {}
        self.reasoning: dict = {}  # handle -> think-end ids matched so far, while the handle is in reasoning
        self.next_handle = 1

    # --- key (SGLang's request mapping) ---

    def chat_request(self, request: dict):
        """The request SGLang would parse: the raw fields through orjson, then `ChatCompletionRequest`."""
        body = {"model": "cere", "messages": [{"role": "user", "content": "x"}]}
        present = []
        for field in OPEN_FIELDS:
            if field not in request:
                continue
            name = "tools" if field == "tools_json" else field
            present.append(name)
            try:
                body[name] = orjson_loads(request[field])
            except RequestError:
                raise
            except Exception:  # noqa: BLE001 - the message would quote the input
                raise RequestError("invalid_value", f"{name} is not valid JSON", name) from None
        if "parallel_tool_calls" in request:
            body["parallel_tool_calls"] = request["parallel_tool_calls"]
            present.append("parallel_tool_calls")
        try:
            return self.request.model_validate(body), present
        except Exception as e:  # noqa: BLE001 - pydantic errors quote the input
            raise RequestError("invalid_value", "the request field has an invalid value",
                               invalid_field(e, present)) from None

    def validate(self, req) -> None:
        """The grammar-relevant checks of `OpenAIServingChat._validate_request` (`serving_chat.py:834-870,887-890`)."""
        tools = list(req.tools or [])
        choice = req.tool_choice
        if isinstance(choice, str) and choice.lower() == "required" and not tools:
            raise RequestError("invalid_value", "tool_choice required needs tools", "tool_choice")
        if choice is not None and not isinstance(choice, str):
            if not any(t.function.name == choice.function.name for t in tools):
                raise RequestError("invalid_value", "tool_choice names a function that is not in tools", "tool_choice")
        for i, tool in enumerate(tools):
            if tool.function.parameters is None:
                continue
            try:
                self.normalize(tool.function.parameters)
                self.validator.check_schema(tool.function.parameters)
            except (self.schema_error, RecursionError):
                raise RequestError("invalid_value", "a tool's parameters are not a valid JSON schema",
                                   f"tools[{i}].function.parameters") from None
        rf = req.response_format
        if rf is not None and rf.type == "json_schema" and getattr(rf.json_schema, "schema_", None) is None:
            raise RequestError("invalid_value", "a json_schema response_format needs a schema", "response_format")

    def tool_constraint(self, req):
        """`serving_chat.py:1078-1121` with the bench server's parsers (thinking_mode False: a reasoning parser runs)."""
        tools = list(req.tools or [])
        if not tools or req.tool_choice == "none":
            return None
        parser = self.parser(tools, TOOL_CALL_PARSER, tokenizer=self.tok)
        constraint = parser.get_structure_constraint(req.tool_choice, parallel_tool_calls=req.parallel_tool_calls,
                                                     thinking_mode=False)
        if (constraint is None and not parser.detector.parses_required_natively()
                and (req.tool_choice == "required" or not isinstance(req.tool_choice, str))):
            constraint = ("json_schema", self.json_constraint(tools, req.tool_choice,
                                                              parallel_tool_calls=req.parallel_tool_calls))
        return constraint

    def key(self, request: dict) -> tuple:
        """(key_type, key_string, param): `to_sampling_params`, then `grammar_manager.py:144-151`."""
        req, present = self.chat_request(request)
        self.validate(req)
        try:
            constraint = self.tool_constraint(req)
            params = req.to_sampling_params(stop=[], model_generation_config={}, tool_call_constraint=constraint)
        except Exception:  # noqa: BLE001 - SGLang's messages quote the input
            raise RequestError("invalid_value", "the request has no valid grammar constraint",
                               "tool_choice" if "tools" in present else present[0]) from None
        rf = req.response_format
        from_rf = rf is not None and rf.type in ("json_schema", "json_object", "structural_tag")
        for key_type, field in (("json", "json_schema"), ("regex", "regex"), ("ebnf", "ebnf"),
                                ("structural_tag", "structural_tag")):
            value = params.get(field)
            if value is None:
                continue
            param = key_type if key_type in ("regex", "ebnf") else "response_format" if from_rf else "tool_choice"
            if not isinstance(value, str):
                raise RequestError("invalid_value", "the request has no valid grammar constraint", param)
            return key_type, value, param
        raise bad("grammar_open needs a request that carries a grammar constraint")

    # --- compile (XGrammarGrammarBackend.dispatch_*, xgrammar_backend.py:335-397) ---

    def compile(self, key_type: str, key: str):
        compiler = self.compiler
        if key_type == "json":
            if key == "$$ANY$$":
                return compiler.compile_builtin_json_grammar()
            ebnf = number_range_ebnf(key, self.xgr.Grammar.from_json_schema)
            if ebnf is not None:
                return compiler.compile_grammar(ebnf)
            return compiler.compile_json_schema(schema=key, any_whitespace=True)
        if key_type == "regex":
            return compiler.compile_regex(key)
        if key_type == "ebnf":
            return compiler.compile_grammar(key)
        tag = json.loads(key)
        if self.is_legacy(tag):
            self.backend._sanitize_structural_tag_structures(tag)
            items = [self.xgr.StructuralTagItem(begin=s["begin"], schema=json.dumps(s["schema"]), end=s["end"])
                     for s in tag["structures"]]
            new_tag = self.xgr.StructuralTag.from_legacy_structural_tag(items, tag["triggers"])
            new_tag.format.at_least_one = tag.get("at_least_one", False)
            return compiler.compile_structural_tag(new_tag)
        fmt = tag.get("format")
        if isinstance(fmt, dict):
            self.backend._sanitize_structural_format(fmt)
            tag["format"] = fmt
            key = json.dumps(tag)
        return compiler.compile_structural_tag(key)

    def mask_bytes(self, matcher) -> bytes:
        """The next-token mask: 7,760 little-endian u32 words."""
        matcher.fill_next_token_bitmask(self.bitmask, 0)
        return self.bitmask[0].numpy().tobytes()

    def mask(self, matcher) -> str:
        """The next-token mask, base64."""
        return base64.b64encode(self.mask_bytes(matcher)).decode("ascii")

    def advance_think_end(self, matched: int, token: int) -> int:
        """`TokenSequenceMatcher.advance`: the think-end ids matched after `token`."""
        while matched and token != self.think_end[matched]:
            matched = self.think_prefix[matched - 1]
        return matched + 1 if token == self.think_end[matched] else matched

    # --- ops ---

    def open(self, request: dict) -> dict:
        if not request.keys() <= OPEN_KEYS:
            raise bad("grammar_open takes op and optionally " + ", ".join(sorted(OPEN_KEYS - {"op"})))
        if not all(is_str(request[f]) for f in OPEN_FIELDS if f in request):
            raise bad("the grammar_open request fields are raw JSON text strings")
        if type(request.get("parallel_tool_calls", True)) is not bool:
            raise bad("parallel_tool_calls is a boolean")
        thinking = request.get("enable_thinking", False)
        if type(thinking) is not bool:
            raise RequestError("invalid_value", "enable_thinking is a boolean", "enable_thinking")
        key_type, key, param = self.key(request)
        if len(self.matchers) >= MAX_GRAMMARS:
            raise RequestError("overloaded", f"at most {MAX_GRAMMARS} grammars are open")
        try:
            compiled = self.compile(key_type, key)
            matcher = self.xgr.GrammarMatcher(compiled, max_rollback_tokens=self.rollback, override_stop_tokens=None)
        except RequestError as e:
            raise RequestError(e.kind, e.message, param) from None
        except Exception:  # noqa: BLE001 - xgrammar's messages quote the input
            raise RequestError("invalid_value", "the grammar failed to compile", param) from None
        handle = self.next_handle
        self.next_handle += 1
        self.matchers[handle] = matcher
        if thinking:
            self.reasoning[handle] = 0
        return {"ok": True, "op": "grammar_open", "grammar": handle, "key_type": key_type,
                "key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "bitmask": ALL_ONES_MASK if thinking else self.mask(matcher)}

    def matcher_of(self, request: dict, keys: set):
        if set(request) != keys:
            raise bad(f"{request['op']} takes exactly the keys {', '.join(sorted(keys))}")
        handle = request["grammar"]
        if type(handle) is not int:
            raise bad("grammar is an integer handle")
        if handle not in self.matchers:
            raise RequestError("invalid_request", "no open grammar has this handle")
        return handle, self.matchers[handle]

    def step(self, request: dict) -> dict:
        token = request.get("token")
        if "token" in request and not (type(token) is int and 0 <= token < VOCAB_SIZE):
            raise bad(f"token is an integer in [0, {VOCAB_SIZE})")
        handle, matcher = self.matcher_of(request, {"op", "grammar", "token"})
        if handle in self.reasoning:
            # ReasonerGrammarObject (reasoner_grammar_backend.py:83-153): in reasoning every token is accepted, the
            # matcher never sees it and the mask is left unfilled; the token completing the think-end ids ends
            # reasoning, and the mask is then the grammar's own (its initial one).
            matched = self.advance_think_end(self.reasoning[handle], token)
            if matched < len(self.think_end):
                self.reasoning[handle] = matched
                return {"ok": True, "op": "grammar_step", "accepted": True, "terminated": False,
                        "bitmask": ALL_ONES_MASK}
            del self.reasoning[handle]
            return {"ok": True, "op": "grammar_step", "accepted": True, "terminated": False,
                    "bitmask": self.mask(matcher)}
        # XGrammarGrammar.accept_token ignores tokens after termination (xgrammar_backend.py:93-95).
        accepted = not matcher.is_terminated() and bool(matcher.accept_token(token))
        terminated = bool(matcher.is_terminated())
        return {"ok": True, "op": "grammar_step", "accepted": accepted, "terminated": terminated,
                "bitmask": self.mask(matcher) if accepted and not terminated else None}

    def drafts(self, request: dict) -> dict:
        drafts = request.get("drafts")
        if "drafts" in request and not (type(drafts) is list and 0 < len(drafts) <= MAX_DRAFTS
                                        and all(type(t) is int and 0 <= t < VOCAB_SIZE for t in drafts)):
            raise bad(f"drafts is a list of 1 to {MAX_DRAFTS} integers in [0, {VOCAB_SIZE})")
        handle, matcher = self.matcher_of(request, {"op", "grammar", "drafts"})
        # `step` per draft on a copy of the handle's state: the reasoning progress is a local and the matcher is
        # rolled back by every token it accepted, so the handle is left exactly as it was.
        matched = self.reasoning.get(handle)  # None: out of reasoning, the matcher sees each draft
        masks, fed = [], 0
        try:
            for token in drafts:
                if matched is not None:
                    matched = self.advance_think_end(matched, token)
                    if matched < len(self.think_end):
                        masks.append(ALL_ONES_ROW)
                    else:
                        matched = None
                        masks.append(self.mask_bytes(matcher))
                    continue
                if matcher.is_terminated() or not matcher.accept_token(token):
                    break
                fed += 1
                if matcher.is_terminated():
                    break
                masks.append(self.mask_bytes(matcher))
        finally:
            if fed:
                matcher.rollback(fed)
        return {"ok": True, "op": "grammar_drafts", "accepted": len(masks),
                "bitmasks": base64.b64encode(b"".join(masks)).decode("ascii")}

    def close(self, request: dict) -> dict:
        handle, _ = self.matcher_of(request, {"op", "grammar"})
        del self.matchers[handle]
        self.reasoning.pop(handle, None)
        return {"ok": True, "op": "grammar_close"}

    def handle(self, request: dict) -> dict:
        op = request["op"]
        if op == "grammar_open":
            return self.open(request)
        if op == "grammar_step":
            return self.step(request)
        if op == "grammar_drafts":
            return self.drafts(request)
        return self.close(request)


def invalid_field(error, present: list) -> str:
    """The request field a validation error names (pydantic `loc`), else the first field the request carries."""
    try:
        loc = error.errors()[0]["loc"][0]
    except Exception:  # noqa: BLE001 - not a pydantic error
        loc = None
    if loc in present:
        return loc
    return present[0] if present else "response_format"


def read_exact(stream, n: int) -> bytes | None:
    """n bytes, or None at EOF before the first byte; Truncated at EOF after it."""
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            if data:
                raise Truncated()
            return None
        data += chunk
    return data


def drain(stream, n: int) -> None:
    while n:
        chunk = stream.read(min(n, DRAIN_CHUNK))
        if not chunk:
            raise Truncated()
        n -= len(chunk)


def read_request(stream, limit: int) -> dict | RequestError | None:
    """The next request object, a RequestError for a frame that is not one, or None at EOF."""
    header = read_exact(stream, HEADER.size)
    if header is None:
        return None
    (length,) = HEADER.unpack(header)
    if length > limit:
        drain(stream, length)
        return RequestError("frame_too_large", f"request frames are limited to {limit} bytes")
    payload = read_exact(stream, length) if length else b""
    if payload is None:
        raise Truncated()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return RequestError("bad_json", "the frame is not UTF-8 JSON")
    if not isinstance(value, dict):
        return RequestError("bad_json", "the frame is not a JSON object")
    return value


def is_str(value) -> bool:
    return isinstance(value, str)


def bad(message: str):
    return RequestError("bad_request", message)


def tool_call_of(call) -> dict:
    """A D3 history call; `arguments` becomes the object SGLang hands the template (`serving_chat.py:126-145`)."""
    if not (isinstance(call, dict) and set(call) == {"id", "type", "function"} and is_str(call["id"])
            and call["type"] == "function" and isinstance(call["function"], dict)
            and set(call["function"]) == {"name", "arguments"} and is_str(call["function"]["name"])
            and is_str(call["function"]["arguments"])):
        raise bad("a tool call is {id, type: function, function: {name, arguments: string}} and nothing else")
    function = call["function"]
    return {"id": call["id"], "type": "function",
            "function": {"name": function["name"], "arguments": arguments_of(function["arguments"])}}


def parts_of(content: list, index: int, images: list) -> list:
    """D2 parts of a user message as SGLang's openai content format hands them to the template
    (`jinja_template_utils.py:123-215`): an image part becomes `{"type": "image"}` and its URL is appended to
    `images` as (url, param); a text part stays `{"type": "text", "text"}`."""
    if not 0 < len(content) <= MAX_PARTS:
        raise bad(f"a content array has 1 to {MAX_PARTS} parts")
    out = []
    for j, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and set(part) == {"type", "text"} and is_str(part["text"]):
            out.append({"type": "text", "text": part["text"]})
            continue
        image = part.get("image_url") if kind == "image_url" and set(part) == {"type", "image_url"} else None
        if not (isinstance(image, dict) and "url" in image and image.keys() <= {"url", "detail"}
                and is_str(image["url"]) and image.get("detail", "auto") == "auto"):
            raise bad("a content part is {type: text, text: string} or "
                      "{type: image_url, image_url: {url: string, detail?: auto}}")
        if len(images) == MAX_IMAGES:
            raise bad(f"a request carries at most {MAX_IMAGES} images")
        images.append((image["url"], f"messages[{index}].content[{j}].image_url.url"))
        out.append({"type": "image"})
    return out


def image_bytes(url: str, param: str) -> bytes:
    """D3: only PNG/JPEG base64 data URLs, strict RFC 4648 (no whitespace, correct padding)."""
    prefix = next((p for p in DATA_URL_PREFIXES if url.startswith(p)), None)
    if prefix is None:
        raise RequestError("invalid_value", "an image URL is a data:image/png or data:image/jpeg base64 URL", param)
    try:
        return base64.b64decode(url[len(prefix):], validate=True)
    except (binascii.Error, ValueError):
        raise RequestError("invalid_value", "the image URL is not strict base64", param) from None


def message_of(m, index: int, images: list | None) -> dict:
    """A D3 message as SGLang's template sees it: `model_dump()` minus None values, content None -> "". A user or
    tool message's content may be a D2 parts array when the sidecar serves feature `image` (`images` is a list)."""
    if not isinstance(m, dict):
        raise bad("each message is an object")
    role = m.get("role")
    keys = set(m)
    if role == "user" and keys == {"role", "content"} and isinstance(m["content"], list):
        if images is None:
            raise RequestError("invalid_value", "this sidecar does not serve feature image",
                               f"messages[{index}].content")
        return {"role": role, "content": parts_of(m["content"], index, images)}
    if role in ("system", "user"):
        if keys != {"role", "content"} or not is_str(m["content"]):
            raise bad("a system or user message is {role, content: string} and nothing else")
        return {"role": role, "content": m["content"]}
    if role == "assistant":
        content = m.get("content")
        calls = m.get("tool_calls")
        reasoning = m.get("reasoning_content")
        if not (keys <= {"role", "content", "tool_calls", "reasoning_content"}
                and (content is None or is_str(content)) and (calls is None or isinstance(calls, list))
                and ("reasoning_content" not in keys or is_str(reasoning))):
            raise bad("an assistant message is {role, content?: string|null, tool_calls?: [call, ...], "
                      "reasoning_content?: string}")
        if content is None and not calls:
            raise bad("an assistant message without content needs tool_calls")
        out = {"role": role, "content": "" if content is None else content}
        if "reasoning_content" in keys:
            out["reasoning_content"] = reasoning
        if calls is not None:
            out["tool_calls"] = [tool_call_of(c) for c in calls]
        return out
    if role == "tool":
        if keys != {"role", "content", "tool_call_id"} or not is_str(m["tool_call_id"]):
            raise bad("a tool message is {role, content: string|parts, tool_call_id: string} and nothing else")
        if isinstance(m["content"], list):
            if images is None:
                raise RequestError("invalid_value", "this sidecar does not serve feature image",
                                   f"messages[{index}].content")
            content = parts_of(m["content"], index, images)
        elif is_str(m["content"]):
            content = m["content"]
        else:
            raise bad("a tool message is {role, content: string|parts, tool_call_id: string} and nothing else")
        return {"role": role, "content": content, "tool_call_id": m["tool_call_id"]}
    raise bad("a message role is system, user, assistant or tool")


def covered_strings(messages: list, tools: list | None):
    """N5's covered text, as the template receives it: system, user and tool `content` (a string, or the text of
    each text part) and every key and string value of the rendered tools. Not assistant content or `tool_calls`."""
    for m in messages:
        if m["role"] in COVERED_ROLES:
            content = m["content"]
            if isinstance(content, str):
                yield content
            else:
                yield from (part["text"] for part in content if part.get("type") == "text")
    stack = [] if tools is None else [tools]
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                stack += [key, item]
        elif isinstance(value, list):
            stack += value


def marked_message(m: dict, wrap) -> dict:
    """`m` with `wrap` applied to its covered strings (`covered_strings`)."""
    if m["role"] not in COVERED_ROLES:
        return m
    content = m["content"]
    if isinstance(content, str):
        return dict(m, content=wrap(content))
    return dict(m, content=[dict(p, text=wrap(p["text"])) if p.get("type") == "text" else p for p in content])


def marked_json(value, wrap):
    """A copy of the tools JSON with `wrap` applied to every key and string value."""
    if isinstance(value, str):
        return wrap(value)
    if isinstance(value, dict):
        return {wrap(key): marked_json(item, wrap) for key, item in value.items()}
    if isinstance(value, list):
        return [marked_json(item, wrap) for item in value]
    return value


def encode_chat(request: dict, tokenizer: Tokenizer, vision: Images | None) -> tuple[dict, list]:
    if not (request.keys() <= ENCODE_KEYS and "messages" in request):
        raise bad("encode_chat takes op, messages and optionally tools_json, render_tools, stop, enable_thinking, "
                  "reasoning_effort, preserve_thinking")
    messages = request["messages"]
    tools_json = request.get("tools_json")
    render_tools = request.get("render_tools", False)
    stop = request.get("stop", [])
    if not (isinstance(messages, list) and messages):
        raise bad("messages must be a non-empty list")
    if not (tools_json is None or is_str(tools_json)) or "tools_json" in request and tools_json is None:
        raise bad("tools_json is a string")
    if type(render_tools) is not bool or render_tools and tools_json is None:
        raise bad("render_tools is a boolean and needs tools_json")
    if not (isinstance(stop, list) and all(is_str(s) for s in stop)):
        raise bad("stop is a list of strings")
    kwargs = {}
    if "enable_thinking" in request or not tokenizer.thinking_unset:
        kwargs["enable_thinking"] = request.get("enable_thinking", False)
        if type(kwargs["enable_thinking"]) is not bool:
            raise bad("enable_thinking is a boolean")
    if "reasoning_effort" in request:
        effort = request["reasoning_effort"]
        if not (is_str(effort) or type(effort) in (int, float)):
            raise bad("reasoning_effort is a string or a number")
        kwargs["reasoning_effort"] = effort
    if "preserve_thinking" in request:
        if type(request["preserve_thinking"]) is not bool:
            raise bad("preserve_thinking is a boolean")
        kwargs["preserve_thinking"] = request["preserve_thinking"]
    urls = None if vision is None else []
    rendered = [message_of(m, i, urls) for i, m in enumerate(messages)]
    images = [vision.prepare(image_bytes(url, param), param) for url, param in urls or ()]
    prepared = None if tools_json is None else Reference.get().prepare_tools(tools_json)
    ids, used = tokenizer.encode_chat(rendered, prepared if render_tools and prepared else None, kwargs)
    reply = {"ok": True, "op": "encode_chat", "ids": ids, "stop_token_lens": tokenizer.stop_token_lens(stop),
             "tools_prepared": prepared, "last_message_start": 0}
    if not images:
        reply["last_message_start"] = tokenizer.last_message_start(rendered, used, kwargs, ids)
        return reply, []
    return expand_images(reply, vision, vision.process_prepared(images))


def expand_images(reply: dict, vision: Images, processed: list) -> tuple[dict, list]:
    """Each `<|image_pad|>` of the rendered prompt becomes t*h*w / merge^2 copies of the image token, as
    `base_processor.py` `_expand_input_ids` does; `token_offset` is the first copy's index (D20)."""
    ids = reply["ids"]
    if sum(1 for t in ids if t == vision.image_token_id) != len(processed):
        raise RequestError("encode_error", "the rendered prompt's image placeholders do not match the images")
    expanded, images, chunks, next_image = [], [], [], iter(processed)
    for token in ids:
        if token != vision.image_token_id:
            expanded.append(token)
            continue
        grid, pad_id, pixels = next(next_image)
        images.append({"grid_thw": grid, "token_offset": len(expanded), "pixel_bytes": pixels.nbytes,
                       "pad_id": pad_id})
        expanded.extend([token] * (grid[0] * grid[1] * grid[2] // (vision.merge * vision.merge)))
        chunks.append(pixels)
    reply = dict(reply, ids=expanded, images=images, attachment_bytes=sum(c.nbytes for c in chunks))
    return reply, chunks


def tool_target(request: dict, keys: set) -> tuple:
    if set(request) != keys:
        raise bad(f"{request['op']} takes exactly the keys {', '.join(sorted(keys))}")
    tool_json, func = request["tool_json"], request["func"]
    if not ((tool_json is None or isinstance(tool_json, dict)) and (func is None or is_str(func))):
        raise bad("tool_json is an object or null and func a string or null")
    return tool_json, func


def tool_param(request: dict) -> dict:
    tool_json, func = tool_target(request, {"op", "tool_json", "func", "name", "raw"})
    if not (is_str(request["name"]) and is_str(request["raw"])):
        raise bad("name and raw are strings")
    fragment = Reference.get().tool_param(tool_json, func, request["name"], request["raw"])
    return {"ok": True, "op": "tool_param", "fragment": fragment}


def tool_args(request: dict) -> dict:
    tool_json, func = tool_target(request, {"op", "tool_json", "func", "params"})
    params = request["params"]
    if not (isinstance(params, list)
            and all(isinstance(p, list) and len(p) == 2 and is_str(p[0]) and is_str(p[1]) for p in params)):
        raise bad("params is a list of [name, raw] string pairs")
    return {"ok": True, "op": "tool_args", "arguments": Reference.get().tool_args(tool_json, func, params)}


def handle(request: dict, tokenizer: Tokenizer, vision: Images | None, info: dict,
           grammar: Grammar | None) -> tuple[dict, list]:
    """The reply object and the attachment chunks that follow it as one raw frame (none: no frame)."""
    op = request.get("op")
    if op == "info":
        if set(request) != {"op"}:
            raise bad("info takes only the key op")
        return info, []
    if op == "encode_chat":
        return encode_chat(request, tokenizer, vision)
    if op == "tool_param":
        return tool_param(request), []
    if op == "tool_args":
        return tool_args(request), []
    if op in GRAMMAR_OPS:
        if grammar is None:
            raise RequestError("unsupported_op", "the grammar ops are served by the grammar role only")
        return grammar.handle(request), []
    raise RequestError("unknown_op", "op must be info, encode_chat, tool_param, tool_args or a grammar op")


def write_frame(out, obj: dict, attachment: list) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode()
    out.write(HEADER.pack(len(data)) + data)
    if attachment:
        out.write(HEADER.pack(sum(chunk.nbytes for chunk in attachment)))
        for chunk in attachment:
            out.write(chunk)
    out.flush()


def silence_stdio():
    """A private copy of stdout for frames; fds 1 and 2 (and sys.stdout/stderr) go to /dev/null."""
    out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    warnings.simplefilter("ignore")
    return out


def serve(stream, out, tokenizer: Tokenizer, vision: Images | None, info: dict, limit: int,
          grammar: Grammar | None = None) -> int:
    while True:
        try:
            request = read_request(stream, limit)
        except Truncated:
            return 1
        if request is None:
            return 0
        try:
            if isinstance(request, RequestError):
                raise request
            reply, attachment = handle(request, tokenizer, vision, info, grammar)
        except RequestError as e:
            reply, attachment = e.reply(), []
        write_frame(out, reply, attachment)


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("not an integer") from None
    if not 0 < value < 1 << 32:
        raise argparse.ArgumentTypeError("must be between 1 and 2^32 - 1")
    return value


def cache_mib(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("not an integer") from None
    if not 0 <= value < 1 << 20:
        raise argparse.ArgumentTypeError("must be between 0 and 2^20 - 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, type=Path, help="prepared artifact directory (read only)")
    parser.add_argument("--role", choices=("tokenizer", "grammar"), default="tokenizer",
                        help="grammar: also serve the grammar ops (feature grammar, tasks/m28-spec.md D7)")
    parser.add_argument("--feature", action="append", choices=KNOWN_FEATURES, default=[],
                        help="serve and advertise this protocol feature (repeatable)")
    parser.add_argument("--max-request-bytes", type=positive_int, default=MAX_FRAME_BYTES,
                        help=f"request frame limit in bytes (default {MAX_FRAME_BYTES})")
    parser.add_argument("--image-device", choices=(DEVICE, "cpu"), default=DEVICE,
                        help=f"feature image: the image processor's device (default {DEVICE}; cpu: no CUDA context, "
                             "PNG pixels bit-identical to cuda:0's, JPEG decoded by PIL)")
    parser.add_argument("--image-threads", type=positive_int, default=0,
                        help="feature image: torch.set_num_threads(N) at startup (default: torch's own)")
    parser.add_argument("--image-cache-mib", type=cache_mib, default=0,
                        help="feature image: per-image result cache bound in MiB, LRU (default 0: no cache)")
    parser.add_argument("--thinking-kwarg", choices=("false", "unset"), default="false",
                        help="encode_chat without enable_thinking: render with the template kwarg false (default, "
                             "SGLang's) or unset (DICE-8B: its S2 reference renders without chat_template_kwargs)")
    args = parser.parse_args(argv)
    if args.role == "grammar":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no CUDA in the grammar role (docstring; BeatSglang L2)
    directory = args.artifact.resolve()
    err = os.fdopen(os.dup(2), "w")  # the one startup-failure line; library output during loading is dropped
    out = silence_stdio()
    try:
        if not directory.is_dir():
            raise StartupError("the artifact directory does not exist")
        tokenizer_sha256, template_sha256 = lock_digest(directory)
        tokenizer = Tokenizer(directory, thinking_unset=args.thinking_kwarg == "unset")
        vision = (Images(directory, args.image_device, args.image_threads, args.image_cache_mib)
                  if "image" in args.feature and args.role == "tokenizer" else None)
        grammar = Grammar(directory, tokenizer) if args.role == "grammar" else None
    except StartupError as e:
        print(f"tokenizer_sidecar: {e}", file=err, flush=True)
        return 2
    err.close()
    features = set(FEATURES) | (set(args.feature) if vision is not None else set())
    info = {"ok": True, "op": "info", "protocol": PROTOCOL, "tokenizer_sha256": tokenizer_sha256,
            "template_sha256": template_sha256, **tokenizer.versions,
            "min_tokens_suppress_ids": tokenizer.min_tokens_suppress_ids()}
    if vision is not None:
        info["preprocessor_sha256"] = vision.preprocessor_sha256
    if grammar is not None:
        features.update(GRAMMAR_FEATURES)
        info.update(**grammar.versions, stop_token_ids=grammar.eos, vocab_size=VOCAB_SIZE)
    info["features"] = sorted(features)
    return serve(sys.stdin.buffer, out, tokenizer, vision, info, args.max_request_bytes, grammar)


if __name__ == "__main__":
    sys.exit(main())
