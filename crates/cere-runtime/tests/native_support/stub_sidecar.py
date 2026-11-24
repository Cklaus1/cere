"""Stub tokenizer sidecar for the native integration tests: protocol `cere-tokenizer-sidecar-v3` of
scripts/native/tokenizer_sidecar.py (4-byte little-endian length + one UTF-8 JSON object per frame, one reply per
request, same reply and error shapes), without a tokenizer or SGLang. Python stdlib only; never prints text.

    python3 stub_sidecar.py --tokenizer-sha256 H --template-sha256 H [--exit-after N] [--features A,B]
        [--feature image --preprocessor-sha256 H] [--max-request-bytes N] [--log PATH] [--bad-image-offset]
        [--role grammar] [--xgrammar V] [--no-grammar-feature] [--no-grammar-drafts-feature] [--startup-sleep S]
        [--fail-start-if PATH] [--slow-start-if PATH S] [--slow-encode-if PATH S] [--hold-start-while PATH]
        [--die-at-start-role ROLE] [--die-first-start-role ROLE PATH]

    {"op": "info"}         -> the two hashes given on the command line and `features`: `--features` (default
                              `message_boundary,thinking`) plus `image` with `--feature image` (then also
                              `preprocessor_sha256`)
    {"op": "encode_chat"}  -> ids: [2] first when tools are rendered, then per message, in order: [1] followed by
                              the UTF-8 bytes of its content ("" for null) and of each call's name and arguments
                              (ids < 256); stop_token_lens: the UTF-8 length of each stop (a bound on any token
                              count); tools_prepared: the parsed `tools_json` as is. A tool whose `parameters` has
                              `"type": "invalid"` fails like a schema error (`invalid_value`, its path).
                              With `--feature image` (wire contract of M2.13 D8) a user message's content may be a
                              parts list: a text part adds its UTF-8 bytes, an image part (a PNG data URL; only its
                              IHDR is read) adds vision_start 248053, t*h*w/4 image pads 248056 and vision_end
                              248054, its grid from HF `smart_resize`. The reply then carries `images[]`
                              {grid_thw, token_offset (index of the first pad), pixel_bytes, pad_id} and
                              `attachment_bytes`, followed by one frame of synthetic BF16 patch bytes (byte k of
                              image i is (k + 31·i) mod 251). Without the feature list content is `invalid_value`.
                              With feature `message_boundary` (M2.10 D4 (c)) the reply carries
                              `last_message_start`: the ids before the last message (0 for one message or with
                              images, as the real sidecar).
    {"op": "tool_param"}   -> `json.dumps(name): json.dumps(value)`, {"op": "tool_args"} -> `json.dumps(params)`;
                              a value is an int when the prepared tool declares the parameter `integer` and it
                              parses, None for "null" (any case), else the raw string; the raw value "boom" fails
                              with `convert_error`, "unavailable" with `reference_unavailable`, and "die" exits
                              the stub without a reply.

Grammar role (`--role grammar`, spec m28 D7; the tokenizer role refuses these ops with `unsupported_op`):
    {"op": "info"}          -> also `xgrammar` (V, default 0.2.1), `torch`, `stop_token_ids`, `vocab_size` 248320 and
                               `features: ["grammar", "grammar_drafts"]` (no `grammar_drafts` with
                               --no-grammar-drafts-feature, `[]` with --no-grammar-feature)
    {"op": "grammar_open"}  -> handle, `key_type`, `key_sha256` (SHA-256 of the key) and the base64 mask (7,760 LE
                               u32 words) of position 0. One constraint field (raw JSON text) or `tools_json`. The
                               mask of position p allows exactly token (seed + 7p) mod 248320, seed from the key;
                               the third accepted token terminates. A regex "stall" sleeps an hour before replying,
                               "die" exits without a reply, "invalid" fails to compile (`invalid_value`, the field);
                               a 5th open handle is `overloaded`.
    {"op": "grammar_step"}  -> `accepted`, `terminated`, the next mask (null once terminated or for a rejected
                               token); an unknown handle is `invalid_request`; token 4294967295 exits without a reply.
    {"op": "grammar_close"} -> releases the handle.
    {"op": "grammar_drafts"} (GrammarSpec GS-03) -> `accepted`: how many leading `drafts` (1..3 ids) the handle
                               accepts without terminating, walked over a copy of its state as `grammar_step` would
                               (the reasoning deferral too); `bitmasks`: the base64 masks after each accepted draft,
                               concatenated ("" for none). The handle's state and its step count (the K of the
                               knobs) are unchanged; knobs never fire here.

--exit-after N exits 0 after N replies (a sidecar that dies). --startup-sleep S sleeps before serving (a slow warm-up).
--fail-start-if PATH exits 2 at startup when PATH exists (the real sidecar's startup failure), so a test can let the
first child start and its respawn fail. --slow-start-if PATH S sleeps S seconds before serving when PATH exists (a
respawn slower than the readiness probe interval, so `/readyz` sees it). --slow-encode-if PATH S sleeps S seconds
before each `encode_chat` reply while PATH exists (a stub descheduled on a loaded host). --hold-start-while PATH
waits before serving while PATH exists (a respawn a test holds not ready, with no wall-clock sleep).
--die-at-start-role ROLE exits 3 without serving, after the startup waits, when the stub's `--role` is ROLE (one of
the two processes dying while the native load runs, tasks/load-speed-spec.md LS-04). --die-first-start-role ROLE
PATH does the same only while PATH is absent, creating it, so the next start of that role serves (a process that
missed its startup beside the load and starts again alone, LS-04).
EOF on a frame boundary exits 0, inside a frame 1.
--max-request-bytes N is the request frame limit (default 4 MiB). --log PATH appends `max_request_bytes N` at
startup, then each request's op name (no content) as one line. --bad-image-offset reports every image's
token_offset one row late (a reply the executor must refuse).
The knobs are arguments and request values, not environment variables, so tests running in parallel threads never
share them.
"""
import argparse
import base64
import hashlib
import json
import os
import math
import struct
import sys
import time

PROTOCOL = "cere-tokenizer-sidecar-v3"
HEADER = struct.Struct("<I")
DEFAULT_MAX_FRAME_BYTES = 4 << 20
ROLES = ("system", "user", "assistant", "tool")
VOCAB_SIZE = 248320
MASK_WORDS = VOCAB_SIZE // 32
MAX_GRAMMARS = 4
GRAMMAR_OPS = ("grammar_open", "grammar_step", "grammar_close", "grammar_drafts")
MAX_DRAFTS = 3  # CERE_MAX_CONSTRAINED_DRAFTS
CONSTRAINTS = (("response_format", "json"), ("regex", "regex"), ("ebnf", "ebnf"))
VISION_START, IMAGE_PAD, VISION_END = 248053, 248056, 248054
PATCH, FACTOR, MIN_PIXELS, MAX_PIXELS = 16, 32, 65536, 16777216
PATCH_ROW_BYTES = 1536 * 2  # in_channels * temporal_patch * patch^2 BF16 values
PNG_PREFIX = "data:image/png;base64,"


def read_exact(stream, n):
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return None if not data else data
        data += chunk
    return data


def error(kind, message, param=None):
    reply = {"ok": False, "error": {"type": kind, "message": message}}
    if param is not None:
        reply["error"]["param"] = param
    return reply


def smart_resize(height, width):
    """HF `smart_resize` for factor 32 (Python `round` is ties-to-even, as cered's Rust port)."""
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def image_grid(part):
    """The [t, h, w] grid of an image part (a PNG data URL), None when it is not one."""
    url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
    if not isinstance(url, str) or not url.startswith(PNG_PREFIX):
        return None
    try:
        head = base64.b64decode(url[len(PNG_PREFIX):], validate=True)[:24]
    except ValueError:
        return None
    if len(head) < 24 or head[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", head[16:24])
    h_bar, w_bar = smart_resize(height, width)
    return [1, h_bar // PATCH, w_bar // PATCH]


def message_ids(m, at, ids, images, feature_image):
    """Appends message `m`'s ids (and its images) to `ids`; an error reply, or None."""
    malformed = error("bad_request", "each message needs a known role and a string content")
    if not isinstance(m, dict) or m.get("role") not in ROLES:
        return malformed
    content = m.get("content")
    if content is None and m["role"] == "assistant":
        content = ""
    ids.append(1)
    if isinstance(content, list):
        if not feature_image:
            return error("invalid_value", "this sidecar does not serve feature image", f"messages[{at}].content")
        if m["role"] not in ("user", "tool") or not content:
            return error("bad_request", "malformed content parts")
        for j, part in enumerate(content):
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text" and isinstance(part.get("text"), str):
                ids.extend(part["text"].encode("utf-8"))
                continue
            grid = image_grid(part) if kind == "image_url" else None
            if grid is None:
                return error("invalid_value", "the image could not be decoded",
                             f"messages[{at}].content[{j}].image_url.url")
            if len(images) == 32:
                return error("bad_request", "at most 32 images")
            t, h, w = grid
            ids.append(VISION_START)
            images.append({"grid_thw": grid, "token_offset": len(ids), "pixel_bytes": t * h * w * PATCH_ROW_BYTES})
            ids.extend([IMAGE_PAD] * (t * h * w // 4))
            ids.append(VISION_END)
        return None
    if not isinstance(content, str):
        return malformed
    ids.extend(content.encode("utf-8"))
    for call in m.get("tool_calls") or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name, arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            return malformed
        ids.extend((name + arguments).encode("utf-8"))
    return None


# The toy continues a 717-token prompt ending in 2235 with "ab" (`toy_next`), so a thinking answer is visible.
THINK_ID = 2235
# api-lenient design note AL-2: a `preserve_thinking: false` render carries one more token, `PRESERVE_FALSE_ID`,
# before the thinking token (the real template drops earlier think blocks; the stub only makes the kwarg visible).
PRESERVE_FALSE_ID = 3


def encode_chat(request, opts):
    """The reply, and the attachment frame's bytes (None without images)."""
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return error("bad_request", "messages must be a non-empty list"), None
    stop = request.get("stop", [])
    tools_json = request.get("tools_json")
    render_tools = request.get("render_tools", False)
    if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
        return error("bad_request", "stop must be a list of strings"), None
    if render_tools and tools_json is None:
        return error("bad_request", "render_tools needs tools_json"), None
    prepared = None
    if tools_json is not None:
        try:
            prepared = json.loads(tools_json)
        except ValueError:
            return error("invalid_value", "the tools are not valid", "tools"), None
        if not isinstance(prepared, list):
            return error("invalid_value", "the tools are not valid", "tools"), None
        for i, tool in enumerate(prepared):
            if (tool.get("function") or {}).get("parameters", {}).get("type") == "invalid":
                return error("invalid_value", "the tool parameters are not a valid JSON schema",
                             f"tools[{i}].function.parameters"), None
    ids = [2] if render_tools else []
    images = []
    last_message_start = 0
    for at, m in enumerate(messages):
        if at == len(messages) - 1 and at > 0:
            last_message_start = len(ids)
        failed = message_ids(m, at, ids, images, "image" in opts.feature)
        if failed is not None:
            return failed, None
    preserve = request.get("preserve_thinking", True)
    if type(preserve) is not bool:
        return error("bad_request", "preserve_thinking is a boolean"), None
    if preserve is False:
        ids.append(PRESERVE_FALSE_ID)
    # Feature `thinking` (M2.12 D4): an enabled thinking render ends with one more token, `THINK_ID`.
    if request.get("enable_thinking", False) is True:
        ids.append(THINK_ID)
    reply = {"ok": True, "op": "encode_chat", "ids": ids,
             "stop_token_lens": [len(s.encode("utf-8")) for s in stop], "tools_prepared": prepared}
    if "message_boundary" in opts.features.split(","):
        reply["last_message_start"] = 0 if images else last_message_start
    if not images:
        return reply, None
    attachment = bytearray()
    for i, image in enumerate(images):
        pixels = bytes((k + 31 * i) % 251 for k in range(image["pixel_bytes"]))
        digest = int.from_bytes(hashlib.sha256(pixels).digest()[:8], "big")
        image["pad_id"] = 1_000_000 + digest % (1 << 30)
        if opts.bad_image_offset:
            image["token_offset"] += 1
        attachment += pixels
    reply["images"] = images
    reply["attachment_bytes"] = len(attachment)
    return reply, bytes(attachment)


class ConvertError(Exception):
    pass


class Unavailable(Exception):
    pass


def convert(tool_json, name, raw):
    if raw == "boom":
        raise ConvertError()
    if raw == "unavailable":
        raise Unavailable()
    if raw == "die":
        sys.exit(0)
    if raw.lower() == "null":
        return None
    props = ((tool_json or {}).get("function") or {}).get("parameters", {}).get("properties", {})
    if props.get(name, {}).get("type") == "integer":
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def tool_op(request):
    op = request["op"]
    try:
        if op == "tool_param":
            value = convert(request.get("tool_json"), request["name"], request["raw"])
            fragment = f"{json.dumps(request['name'])}: {json.dumps(value, ensure_ascii=False)}"
            return {"ok": True, "op": op, "fragment": fragment}
        params = {}
        for name, raw in request["params"]:
            params[name] = convert(request.get("tool_json"), name, raw)
        return {"ok": True, "op": op, "arguments": json.dumps(params, ensure_ascii=False)}
    except ConvertError:
        return error("convert_error", "a tool value could not be converted")
    except Unavailable:
        return error("reference_unavailable", "the SGLang tool modules cannot be imported")
    except (KeyError, TypeError, ValueError):
        return error("bad_request", "malformed tool op")


class Grammars:
    """The grammar role's open handles: handle -> state (seed or script, accepted tokens, steps, thinking)."""

    def __init__(self, vocab_size, think_end_id):
        self.handles = {}
        self.next = 1
        self.vocab_size = vocab_size
        self.mask_words = (vocab_size + 31) // 32
        self.think_end_id = think_end_id

    def allowed(self, state, position):
        """The ids the mask of `position` allows."""
        script = state["script"]
        if script is None:
            return [(state["seed"] + 7 * position) % self.vocab_size]
        if script["allow"] is not None:
            return script["allow"]
        return [script["seq"][position]]

    def terminated(self, state, position):
        script = state["script"]
        if script is None:
            return position >= 3
        return script["allow"] is None and position >= len(script["seq"])

    def encode(self, words):
        return base64.b64encode(struct.pack(f"<{len(words)}I", *words)).decode("ascii")

    def words(self, state, position):
        if state["thinking"]:
            return [0xFFFFFFFF] * self.mask_words
        words = [0] * self.mask_words
        for token in self.allowed(state, position):
            words[token // 32] |= 1 << (token % 32)
        return words

    def mask(self, state, position):
        return self.encode(self.words(state, position))

    @staticmethod
    def script(value):
        """A regex "seq:A,B,..[;reject@K][;fail@K][;die@K][;sleep@K=S]" (one id per position, the last accepted
        one terminates) or "allow:A,B,.." (that set at every position, never terminating); K counts the handle's
        grammar_step calls from 0: `reject` answers accepted false, `fail` a typed error, `die` exits without a
        reply, `sleep` sleeps S seconds first. None for any other value."""
        if not isinstance(value, str) or not value.startswith(("seq:", "allow:")):
            return None
        head, *knobs = value.split(";")
        kind, ids = head.split(":", 1)
        tokens = [int(t) for t in ids.split(",") if t]
        script = {"seq": tokens, "allow": tokens if kind == "allow" else None, "knobs": {}}
        for knob in knobs:
            name, at = knob.split("@", 1)
            step, _, arg = at.partition("=")
            script["knobs"][int(step)] = (name, float(arg) if arg else None)
        return script

    def open(self, request):
        fields = [(name, kind) for name, kind in CONSTRAINTS if request.get(name) is not None]
        if len(fields) > 1 or (not fields and request.get("tools_json") is None):
            return error("bad_request", "exactly one constraint is needed")
        thinking = request.get("enable_thinking", False)
        if not isinstance(thinking, bool):
            return error("invalid_value", "enable_thinking is a boolean", "enable_thinking")
        script = None
        if fields:
            (name, key_type), raw = fields[0], request[fields[0][0]]
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                return error("invalid_value", "the constraint is not JSON text", name)
            if value == "stall":
                time.sleep(3600)
            if value == "die":
                sys.exit(0)
            if value == "invalid":
                return error("invalid_value", "Grammar compilation failed", name)
            script = self.script(value)
        else:
            key_type, raw = "structural_tag", request["tools_json"] + str(request.get("tool_choice"))
            try:
                script = self.script(json.loads(request["tools_json"])[0]["function"].get("description"))
            except (ValueError, LookupError, TypeError, AttributeError):
                script = None
        if len(self.handles) >= MAX_GRAMMARS:
            return error("overloaded", "too many open grammars")
        key = hashlib.sha256(f"{key_type}:{raw}".encode("utf-8")).hexdigest()
        state = {"seed": int(key[:8], 16) % self.vocab_size, "script": script, "position": 0, "steps": 0,
                 "thinking": thinking}
        handle, self.next = self.next, self.next + 1
        self.handles[handle] = state
        return {"ok": True, "op": "grammar_open", "grammar": handle, "key_type": key_type, "key_sha256": key,
                "bitmask": self.mask(state, 0)}

    def step(self, request):
        state = self.handles.get(request.get("grammar"))
        token = request.get("token")
        if state is None or not isinstance(token, int):
            return error("invalid_request", "unknown grammar handle")
        if token == 0xFFFFFFFF:
            sys.exit(0)
        step, state["steps"] = state["steps"], state["steps"] + 1
        knob, arg = (state["script"] or {"knobs": {}})["knobs"].get(step, (None, None))
        if knob == "sleep":
            time.sleep(arg)
        if knob == "die":
            sys.exit(0)
        if knob == "fail":
            return error("internal", "the matcher failed")
        if state["thinking"]:
            # The D15 contract: every token is accepted in reasoning; the think-end token ends it and its step
            # answers the grammar's initial mask (the matcher never sees the think tokens).
            state["thinking"] = token != self.think_end_id
            return {"ok": True, "op": "grammar_step", "accepted": True, "terminated": False,
                    "bitmask": self.mask(state, 0)}
        position = state["position"]
        accepted = knob != "reject" and token in self.allowed(state, position)
        if accepted:
            state["position"] = position = position + 1
        terminated = self.terminated(state, position)
        bitmask = self.mask(state, position) if accepted and not terminated else None
        return {"ok": True, "op": "grammar_step", "accepted": accepted, "terminated": terminated, "bitmask": bitmask}

    def drafts(self, request):
        state = self.handles.get(request.get("grammar"))
        drafts = request.get("drafts")
        if state is None:
            return error("invalid_request", "unknown grammar handle")
        if (not isinstance(drafts, list) or not 1 <= len(drafts) <= MAX_DRAFTS
                or not all(type(t) is int and 0 <= t < self.vocab_size for t in drafts)):
            return error("bad_request", "drafts is a list of 1 to 3 token ids")
        walk, rows = dict(state), []
        for token in drafts:
            if walk["thinking"]:
                walk["thinking"] = token != self.think_end_id
                rows.append(self.words(walk, 0))
                continue
            position = walk["position"]
            if self.terminated(walk, position) or token not in self.allowed(walk, position):
                break
            walk["position"] = position = position + 1
            if self.terminated(walk, position):
                break
            rows.append(self.words(walk, position))
        return {"ok": True, "op": "grammar_drafts", "accepted": len(rows),
                "bitmasks": self.encode([w for row in rows for w in row])}

    def close(self, request):
        if self.handles.pop(request.get("grammar"), None) is None:
            return error("invalid_request", "unknown grammar handle")
        return {"ok": True, "op": "grammar_close"}


def handle(frame, info, opts, grammars):
    """The reply to one request frame, and the attachment frame that follows it (None for most)."""
    try:
        request = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return error("bad_json", "the request is not UTF-8 JSON"), None
    if not isinstance(request, dict):
        return error("bad_request", "the request is not an object"), None
    op = request.get("op")
    if opts.log:
        with open(opts.log, "a", encoding="utf-8") as log:
            log.write(f"{op if isinstance(op, str) else '?'}\n")
    if op == "info":
        return {"ok": True, "op": "info", **info}, None
    if op == "encode_chat":
        if opts.slow_encode_if is not None and os.path.exists(opts.slow_encode_if[0]):
            time.sleep(float(opts.slow_encode_if[1]))
        return encode_chat(request, opts)
    if op in ("tool_param", "tool_args"):
        return tool_op(request), None
    if op in GRAMMAR_OPS:
        if grammars is None:
            return error("unsupported_op", "grammar ops need the grammar role"), None
        return {"grammar_open": grammars.open, "grammar_step": grammars.step,
                "grammar_close": grammars.close, "grammar_drafts": grammars.drafts}[op](request), None
    return error("unknown_op", "unknown op"), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--exit-after", type=int, default=-1)
    parser.add_argument("--features", default="message_boundary,thinking", help="info.features, comma-separated")
    parser.add_argument("--role", choices=("tokenizer", "grammar"), default="tokenizer")
    parser.add_argument("--xgrammar", default="0.2.1")
    parser.add_argument("--no-grammar-feature", action="store_true")
    parser.add_argument("--no-grammar-drafts-feature", action="store_true")
    parser.add_argument("--startup-sleep", type=float, default=0.0)
    parser.add_argument("--fail-start-if")
    parser.add_argument("--slow-start-if", nargs=2, metavar=("PATH", "S"))
    parser.add_argument("--slow-encode-if", nargs=2, metavar=("PATH", "S"))
    parser.add_argument("--hold-start-while")
    parser.add_argument("--die-at-start-role", choices=("tokenizer", "grammar"))
    parser.add_argument("--die-first-start-role", nargs=2, metavar=("ROLE", "PATH"))
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE, help="grammar info.vocab_size and mask length")
    parser.add_argument("--stop-token-ids", default="248044:248046",
                        help="grammar info.stop_token_ids, ':'-separated (the runtime config's sidecar item pattern)")
    parser.add_argument("--think-end-id", type=int, default=248069, help="the </think> id of enable_thinking")
    parser.add_argument("--feature", action="append", choices=["image"], default=[])
    parser.add_argument("--preprocessor-sha256")
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_FRAME_BYTES)
    parser.add_argument("--log")
    parser.add_argument("--bad-image-offset", action="store_true")
    args = parser.parse_args()
    if args.fail_start_if is not None and os.path.exists(args.fail_start_if):
        return 2
    info = {"protocol": PROTOCOL, "tokenizer_sha256": args.tokenizer_sha256,
            "template_sha256": args.template_sha256, "transformers": "stub", "tokenizers": "stub",
            "features": sorted({f for f in args.features.split(",") if f} | set(args.feature))}
    if "image" in args.feature:
        info["preprocessor_sha256"] = args.preprocessor_sha256
    if args.log:
        with open(args.log, "a", encoding="utf-8") as log:
            log.write(f"max_request_bytes {args.max_request_bytes}\n")
    grammars = None
    if args.role == "grammar":
        grammars = Grammars(args.vocab_size, args.think_end_id)
        if not args.no_grammar_feature:
            drafts = set() if args.no_grammar_drafts_feature else {"grammar_drafts"}
            info["features"] = sorted({*info["features"], "grammar", *drafts})
        info.update({"xgrammar": args.xgrammar, "torch": "stub",
                     "stop_token_ids": [int(t) for t in args.stop_token_ids.split(":") if t],
                     "vocab_size": args.vocab_size})
    time.sleep(args.startup_sleep)
    if args.slow_start_if is not None and os.path.exists(args.slow_start_if[0]):
        time.sleep(float(args.slow_start_if[1]))
    while args.hold_start_while is not None and os.path.exists(args.hold_start_while):
        time.sleep(0.01)
    if args.die_at_start_role == args.role:
        return 3
    if args.die_first_start_role is not None and args.die_first_start_role[0] == args.role:
        first = args.die_first_start_role[1]
        if not os.path.exists(first):
            with open(first, "w"):
                pass
            return 3
    exit_after = args.exit_after
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    replies = 0
    while replies != exit_after:
        header = read_exact(stdin, HEADER.size)
        if header is None:
            return 0
        if len(header) < HEADER.size:
            return 1
        (length,) = HEADER.unpack(header)
        frame = read_exact(stdin, length) if length else b""
        if frame is None or len(frame) < length:
            return 1
        reply, attachment = (error("frame_too_large", "the request frame is too large"), None) \
            if length > args.max_request_bytes else handle(frame, info, args, grammars)
        body = json.dumps(reply, separators=(",", ":")).encode("utf-8")
        stdout.write(HEADER.pack(len(body)) + body)
        if attachment is not None:
            stdout.write(HEADER.pack(len(attachment)) + attachment)
        stdout.flush()
        replies += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
