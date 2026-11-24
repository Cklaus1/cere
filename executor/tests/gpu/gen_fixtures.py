"""Generates the native tokenizer fixtures of tasks/m22-spec.md B4 from synthetic strings, in the pinned image (no GPU).

    docker run --rm --network none --user UID:GID -e HOME=/tmp -e HF_HUB_OFFLINE=1 --entrypoint python3 \
        -v REPO:/src:ro -v MODEL:/model:ro -v REPO/tests/fixtures/native:/out IMAGE \
        /src/executor/tests/gpu/gen_fixtures.py --model /model --out /out

Writes three files consumed by R5's Rust detokenizer and Qwen3 reasoning split (crates/cere-runtime/tests):

* `tokenizer_mini.json`: a valid `tokenizers` JSON derived from the model's `tokenizer.json`: the real normalizer,
  pre-tokenizer, post-processor, decoder and all added tokens; the BPE vocabulary restricted to the 256 byte-level
  symbols plus every token the cases use, with their real IDs, plus every added token at its real ID (`tokenizers`
  numbers an added token absent from the model vocabulary after `len(vocab)`, which would renumber it in a sparse
  vocabulary); the merges whose parts and result are in that vocabulary, in the real order. It decodes the cases
  exactly like the full tokenizer (checked here).
* `detok_cases.json`: `{"schema": "cere-detok-cases-v1", "tokenizer": "tokenizer_mini.json", "tokenizer_sha256",
  "source_tokenizer_sha256", "skip_special_tokens": true, "cases": [{"name", "ids", "text", "text_sha256"}]}`.
  `text` is `transformers` `decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)` with the
  model's tokenizer (what SGLang's detokenizer calls); the mini tokenizer must give the same text for every case.
  `source_tokenizer_sha256` is the deployment lock's `tokenizer_sha256` definition over the model's files.
* `reasoning_cases.json`: `{"schema": "cere-reasoning-cases-v1", "parser", "force_reasoning", "cases": [{"name",
  "text", "non_stream": {"reasoning_text", "normal_text"}, "streams": [{"chunks", "deltas": [{"reasoning_text",
  "normal_text"}], "finish": {"reasoning_text", "normal_text"}}]}]}`: the raw outputs of SGLang's
  `ReasoningParser(parser, stream_reasoning=False, force_reasoning).parse_non_stream(text)` and, per chunking, of
  `ReasoningParser(parser, stream_reasoning=True, force_reasoning)` `parse_stream_chunk` per chunk then
  `parse_stream_end`. `parser` and `force_reasoning` are what the bench server (`--reasoning-parser auto`,
  `--default-chat-template-kwargs {"enable_thinking":false}`, bench_backend.py) resolves for the model's chat
  template: SGLang's own template detection plus `OpenAIServingChat._get_reasoning_from_request`. The response maps
  them as SGLang does (serving_chat.py): `content` = normal_text or "", `reasoning_content` = reasoning_text or null.

Every string below is synthetic (written for this file); no prompt, response or parity text is involved. Random
ID sequences use a fixed seed, so the output is a pure function of the model's tokenizer files and the image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.template_detection import detect_reasoning_parser, detect_reasoning_pattern
from tokenizers import Tokenizer
from transformers import AutoTokenizer

TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
SEED = 20261006
RANDOM_CASES = 64
RANDOM_MAX_LEN = 48

SYNTHETIC = (
    ("ascii_words", "Purple lanterns drift over the quiet harbor tonight."),
    ("leading_space", " leading space and trailing space "),
    ("multiple_spaces", "gap    of    four   and\ttab"),
    ("newlines", "line one\nline two\n\nline four\r\nline five"),
    ("digits", "Order 12345 costs 678.90, ref 0007 and 1e-9."),
    ("punctuation", "Wait... what?! (yes) [no] {maybe} <ok> \"quoted\" 'single' -- dash; colon:"),
    ("code", "def frob(x):\n    return [y * 2 for y in x if y % 3]\n"),
    ("json_like", '{"key": [1, 2, {"nested": null}], "flag": true}'),
    ("latin_accents", "Cr\u00e8me br\u00fbl\u00e9e, na\u00efve fa\u00e7ade, \u00e5ngstr\u00f6m"),
    ("decomposed_accents", "Cre\u0300me bru\u0302le\u0301e"),
    ("cjk", "\u732b\u5728\u5c4b\u9876\u4e0a\u7761\u89c9\u3002\u4eca\u65e5\u306f\u6674\u308c\u3002"),
    ("hangul", "\ud558\ub298\uc774 \ub9d1\ub2e4"),
    ("cyrillic_greek", "\u041f\u0440\u0438\u0432\u0435\u0442 \u03b1\u03b2\u03b3 \u03a9"),
    ("arabic_hebrew", "\u0645\u0631\u062d\u0628\u0627 \u05e9\u05dc\u05d5\u05dd"),
    ("emoji", "Rocket \U0001f680 then family \U0001f468\u200d\U0001f469\u200d\U0001f467 and flag \U0001f1fa\U0001f1f3"),
    ("rare_symbols", "\u2603 \u2042 \u29bf \U0001d11e \U0001f9ec \u00a7\u00b6"),
    ("math", "\u2200x \u2208 \u211d: x\u00b2 \u2265 0 \u21d2 \u221a(x\u00b2) = |x|"),
    ("zero_width", "a\u200bb\u200cc\u200dd\ufeffe"),
    ("think_block", "<think>\nweigh option A against B\n</think>\n\nChoose B."),
    ("chat_specials", "<|im_start|>assistant\nhello there<|im_end|>\n<|endoftext|>"),
    ("tool_markup", "<tool_call>\n{\"name\": \"lookup\"}\n</tool_call>"),
    ("long_repeat", "nananananananananananananananananananananananananananananananananananananananananana batman"),
    ("mixed_scripts", "abc\u4e2d\u6587def\U0001f600ghi\u00e9jkl"),
    ("single_char", "x"),
    ("only_space", " "),
    ("only_newline", "\n"),
)

REASONING = (
    ("plain", "The answer is forty-two."),
    ("empty", ""),
    ("full_block", "<think>\nCount the beads twice.\n</think>\n\nThere are nine beads."),
    ("block_no_newlines", "<think>short</think>tail"),
    ("close_without_open", "silent reasoning</think>\n\nvisible answer"),
    ("open_unterminated", "<think>\nthis reasoning was cut off by the length lim"),
    ("double_open", "<think><think>nested start</think>after"),
    ("empty_block", "<think>\n\n</think>\n\nOnly content."),
    ("late_block", "preface text <think>late thought</think> trailing words"),
    ("tool_in_reasoning", "<think>plan a call<tool_call>\n{\"name\": \"f\"}\n</tool_call>"),
    ("tool_without_think", "Call: <tool_call>{}</tool_call>"),
    ("two_closes", "<think>a</think>b</think>c"),
    ("unicode", "<think>\u601d\u8003 \u00e9 \U0001f600</think>\u7b54\u3048 \u00fc"),
    ("angle_brackets", "<thin> is not a tag, nor </thinking> or <think"),
    ("close_only", "</think>"),
    ("open_only", "<think>"),
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lock_tokenizer_sha256(model: Path) -> str:
    lines = "".join(f"{sha256_hex((model / n).read_bytes())}  {n}\n" for n in sorted(TOKENIZER_FILES, key=str.encode))
    return sha256_hex(lines.encode())


def merge_parts(merge) -> tuple[str, str]:
    if isinstance(merge, str):
        a, b = merge.split(" ", 1)
        return a, b
    a, b = merge
    return a, b


def detok_cases(hf, raw: Tokenizer) -> list[dict]:
    """(name, ids) of every case: synthetic strings, byte splits, special-token mixes and seeded random IDs."""
    cases = []
    for name, text in SYNTHETIC:
        cases.append((name, hf.encode(text, add_special_tokens=False)))
    # Byte-level splits of multi-byte characters (a sequence that ends inside a UTF-8 character).
    byte_ids = [raw.token_to_id(symbol) for symbol in byte_symbols()]
    for name, text in (("euro_bytes", "\u20ac"), ("emoji_bytes", "\U0001f680"), ("cjk_bytes", "\u732b")):
        data = text.encode()
        cases.append((name, [byte_ids[b] for b in data]))
        cases.append((name + "_truncated", [byte_ids[b] for b in data[:-1]]))
        cases.append((name + "_then_ascii", [byte_ids[b] for b in data[:-1]] + hf.encode("ok",
                                                                                        add_special_tokens=False)))
    added = sorted(raw.get_added_tokens_decoder().items())
    special = [i for i, t in added if t.special]
    plain_added = [i for i, t in added if not t.special]
    words = hf.encode(" alpha beta", add_special_tokens=False)
    cases.append(("all_special_tokens", special))
    cases.append(("all_non_special_added_tokens", plain_added))
    cases.append(("specials_between_words", [special[0]] + words + special[1:3] + words + [special[-1]]))
    cases.append(("added_between_words", words + plain_added[:4] + words))
    rng = random.Random(SEED)
    size = raw.get_vocab_size(with_added_tokens=True)
    for k in range(RANDOM_CASES):
        cases.append((f"random_{k:02d}", [rng.randrange(size) for _ in range(rng.randint(1, RANDOM_MAX_LEN))]))
    return cases


def byte_symbols() -> list[str]:
    """GPT-2 bytes_to_unicode: the ByteLevel symbol of each byte value 0..255."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    codes = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            codes.append(256 + n)
            n += 1
    table = dict(zip(printable, codes))
    return [chr(table[b]) for b in range(256)]


def mini_tokenizer(source: dict, raw: Tokenizer, used: set[int]) -> dict:
    added = {t["content"]: t["id"] for t in source["added_tokens"]}
    keep = {raw.token_to_id(s) for s in byte_symbols()} | (used - set(added.values()))
    vocab = {token: i for token, i in source["model"]["vocab"].items() if i in keep}
    if len(vocab) != len(keep):
        raise SystemExit(f"{len(keep) - len(vocab)} kept IDs have no vocabulary entry")
    vocab.update(added)
    merges = []
    for merge in source["model"]["merges"]:
        a, b = merge_parts(merge)
        if a in vocab and b in vocab and a + b in vocab:
            merges.append(merge)
    model = dict(source["model"], vocab=dict(sorted(vocab.items(), key=lambda kv: kv[1])), merges=merges)
    return dict(source, model=model)


def build_detok(model: Path, out: Path) -> None:
    hf = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False,
                                       clean_up_tokenization_spaces=False, use_fast=True)
    raw = Tokenizer.from_file(str(model / "tokenizer.json"))
    named = detok_cases(hf, raw)
    used = {i for _, ids in named for i in ids}
    source = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))
    mini = mini_tokenizer(source, raw, used)
    mini_bytes = (json.dumps(mini, ensure_ascii=False, indent=1) + "\n").encode()
    mini_tok = Tokenizer.from_str(mini_bytes.decode())
    cases = []
    for name, ids in named:
        text = hf.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        mini_text = mini_tok.decode(ids, skip_special_tokens=True)
        if mini_text != text:
            raise SystemExit(f"case {name}: the mini tokenizer decodes differently from the model's")
        cases.append({"name": name, "ids": ids, "text": text, "text_sha256": sha256_hex(text.encode())})
    (out / "tokenizer_mini.json").write_bytes(mini_bytes)
    doc = {"schema": "cere-detok-cases-v1", "tokenizer": "tokenizer_mini.json",
           "tokenizer_sha256": sha256_hex(mini_bytes), "source_tokenizer_sha256": lock_tokenizer_sha256(model),
           "skip_special_tokens": True, "cases": cases}
    write_json(out / "detok_cases.json", doc)


def chunkings(text: str) -> list[list[str]]:
    """Whole text, one code point per chunk, 3-code-point chunks, and splits inside every tag."""
    out = [[text], list(text), [text[i:i + 3] for i in range(0, len(text), 3)]]
    cuts = sorted({i + k for tag in ("<think>", "</think>", "<tool_call>") for i in find_all(text, tag)
                   for k in (2, 5) if 0 < i + k < len(text)})
    if cuts:
        bounds = [0] + cuts + [len(text)]
        out.append([text[a:b] for a, b in zip(bounds, bounds[1:])])
    return [c for c in out if c] or [[""]]


def find_all(text: str, tag: str) -> list[int]:
    out, i = [], text.find(tag)
    while i >= 0:
        out.append(i)
        i = text.find(tag, i + 1)
    return out


def pair(reasoning, normal) -> dict:
    return {"reasoning_text": reasoning or "", "normal_text": normal or ""}


# The bench server's `--default-chat-template-kwargs` (scripts/live/bench_backend.py SGLANG_ARGS).
BENCH_TEMPLATE_KWARGS = {"enable_thinking": False}


def reasoning_mode(model: Path, hf) -> tuple[str, bool]:
    """(parser, force_reasoning) of a bench request: auto-detected parser, template force or the request toggle."""
    template = (model / "chat_template.jinja").read_text(encoding="utf-8")
    template_force, config = detect_reasoning_pattern(template)
    parser = detect_reasoning_parser(template, hf, config, template_force)
    if parser is None:
        raise SystemExit("SGLang detects no reasoning parser for the model's chat template")
    serving = SimpleNamespace(reasoning_parser=parser, template_manager=SimpleNamespace(reasoning_config=config),
                              _reasoning_detector=ReasoningParser(parser, stream_reasoning=True, tokenizer=hf).detector)
    request = ChatCompletionRequest(model="fixture", messages=[{"role": "user", "content": "synthetic"}],
                                    chat_template_kwargs=dict(BENCH_TEMPLATE_KWARGS))
    return parser, bool(template_force or OpenAIServingChat._get_reasoning_from_request(serving, request))


def build_reasoning(model: Path, out: Path) -> None:
    hf = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False, use_fast=True)
    parser_name, force = reasoning_mode(model, hf)
    cases = []
    for name, text in REASONING:
        reasoning, normal = ReasoningParser(parser_name, stream_reasoning=False,
                                            force_reasoning=force).parse_non_stream(text)
        streams = []
        for chunks in chunkings(text):
            parser = ReasoningParser(parser_name, stream_reasoning=True, force_reasoning=force)
            deltas = [pair(*parser.parse_stream_chunk(c)) for c in chunks]
            streams.append({"chunks": chunks, "deltas": deltas, "finish": pair(*parser.parse_stream_end())})
        cases.append({"name": name, "text": text, "non_stream": pair(reasoning, normal), "streams": streams})
    write_json(out / "reasoning_cases.json", {"schema": "cere-reasoning-cases-v1", "parser": parser_name,
                                              "force_reasoning": force, "cases": cases})


def write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="model directory (tokenizer files, read only)")
    p.add_argument("--out", type=Path, required=True, help="tests/fixtures/native")
    args = p.parse_args()
    build_detok(args.model, args.out)
    build_reasoning(args.model, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
