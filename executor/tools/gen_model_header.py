#!/usr/bin/env python3
"""SelfTune D1 / ST1a (tasks/selftune-spec.md, ST-R7): the constants header of one native model descriptor.

usage: python3 executor/tools/gen_model_header.py executor/models/<model_id>.json OUT.h

Reads the descriptor (schema schemas/model_descriptor.schema.json; `make precheck` validates it) and writes a C++17
header of `inline constexpr` constants in `namespace cere::gen`, one per descriptor fact, so the executor's dimension
constants read the descriptor instead of repeating it. Byte-deterministic: constants sorted by name, no timestamp, no
path, so the build stays reproducible (build/repro-build). Standard library only: executor/CMakeLists.txt runs it
with the build image's python3, the Makefile host build and executor/hip/build.sh with the host's.

What becomes a constant (the field path drops a leading `dimensions.` and is CamelCased: `vision.hidden_size` ->
kVisionHiddenSize):
- every `{value, sources}` fact outside `checkpoint`: an integer is uint32_t (int64_t when negative or wider), a
  number double, a boolean bool, a string `char[]`, a list of integers / numbers / strings an array of those;
- the plain strings `model_id`, `draft_head.kind` and every plain string field of `decode`;
- kHas<Section> for the optional sections graph, vision and decode (false when null or absent);
- per layer, one array per `layers.kinds` key (kLayerMixer, kLayerFfn, kLayerPle, ...);
- kFamily, the CERE_FAMILY_* value (include/cere_executor.h) of `model_type` (SelfTune ST4: a library is built for
  one model, and its family decides which descriptors it accepts), also as the macro CERE_GEN_FAMILY for the sources
  whose code exists for one family only (`#if CERE_GEN_FAMILY == CERE_FAMILY_...`). A model_type with no family is
  refused.
Not emitted: `schema_version`, `checkpoint` (it holds an absolute reference path) and `slots` (the registry's, ST2).
Anything else (a nested list, a non-ASCII string, a non-finite number, a mixed list) is refused by name.
"""
import json
import math
import os
import sys
from pathlib import Path

OPTIONAL_SECTIONS = ("decode", "graph", "vision")
SKIPPED = ("schema_version", "checkpoint", "slots", "layers")
U32_MAX = (1 << 32) - 1
# model_type -> CERE_FAMILY_* (include/cere_executor.h; tests/test_gen_model_header.py checks the two agree).
FAMILIES = {"qwen4_exp": 0, "sdar": 1, "inkling2_loop": 3}


class DescriptorError(ValueError):
    """The descriptor holds something the header cannot express."""


def constant_name(field: str) -> str:
    """`dimensions.num_hidden_layers` -> kNumHiddenLayers; `vision.hidden_size` -> kVisionHiddenSize."""
    field = field.removeprefix("dimensions.")
    return "k" + "".join(word[:1].upper() + word[1:] for part in field.split(".") for word in part.split("_"))


def _string(field: str, value: str) -> str:
    text = json.dumps(value, ensure_ascii=True)
    if "\\u" in text or "\\/" in text:
        raise DescriptorError(f"{field}: string {value!r} is not printable ASCII")
    return text


def _number(field: str, value) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DescriptorError(f"{field}: non-finite number {value!r}")
        return repr(value)
    return str(value)


def _scalar_type(field: str, values: list) -> str:
    if all(isinstance(v, bool) for v in values):
        return "bool"
    if any(isinstance(v, bool) for v in values):
        raise DescriptorError(f"{field}: booleans mixed with other values")
    if all(isinstance(v, int) for v in values):
        if all(0 <= v <= U32_MAX for v in values):
            return "uint32_t"
        if all(-(1 << 63) <= v < (1 << 63) for v in values):
            return "int64_t"
        raise DescriptorError(f"{field}: integer outside int64_t")
    if all(isinstance(v, (int, float)) for v in values):
        return "double"
    if all(isinstance(v, str) for v in values):
        return "const char *"
    raise DescriptorError(f"{field}: unsupported value {values!r}")


def _literal(field: str, kind: str, value) -> str:
    if kind == "bool":
        return "true" if value else "false"
    if kind == "const char *":
        return _string(field, value)
    if kind == "double":
        return _number(field, float(value))
    return _number(field, value)


def _declaration(field: str, name: str, value) -> str:
    if isinstance(value, list):
        if not value:
            raise DescriptorError(f"{field}: empty list")
        if any(isinstance(v, (list, dict)) or v is None for v in value):
            raise DescriptorError(f"{field}: nested or null list element")
        kind = _scalar_type(field, value)
        items = ", ".join(_literal(field, kind, v) for v in value)
        qualified = "const char *const" if kind == "const char *" else kind
        return f"inline constexpr {qualified} {name}[{len(value)}] = {{{items}}};"
    if isinstance(value, str):
        return f"inline constexpr char {name}[] = {_string(field, value)};"
    if isinstance(value, (dict, type(None))):
        raise DescriptorError(f"{field}: unsupported value {value!r}")
    kind = _scalar_type(field, [value])
    return f"inline constexpr {kind} {name} = {_literal(field, kind, value)};"


def _facts(node, path: str):
    """(dotted field, value, is_fact) for every fact and plain leaf below `node`, in document order."""
    if isinstance(node, dict):
        if set(node) == {"value", "sources"}:
            yield path, node["value"], True
            return
        for key, child in node.items():
            yield from _facts(child, f"{path}.{key}" if path else key)
    else:
        yield path, node, False


def constants(data: dict) -> dict[str, tuple[str, str]]:
    """Constant name -> (descriptor field, C++ declaration)."""
    if not isinstance(data, dict) or not isinstance(data.get("model_id"), str):
        raise DescriptorError("not a model descriptor (no model_id)")
    out: dict[str, tuple[str, str]] = {}

    def add(field: str, value) -> None:
        name = constant_name(field)
        if name in out:
            raise DescriptorError(f"{field}: constant {name} already comes from {out[name][0]}")
        out[name] = (field, _declaration(field, name, value))

    for key, node in data.items():
        if key in SKIPPED:
            continue
        if key in OPTIONAL_SECTIONS and node is None:
            continue
        for field, value, fact in _facts(node, key):
            if fact:
                add(field, value)
            elif field == "model_id" or field == "draft_head.kind" or (field.startswith("decode.")
                                                                         and isinstance(value, str)):
                add(field, value)
            else:
                raise DescriptorError(f"{field}: plain value {value!r} outside a {{value, sources}} fact")
    for section in OPTIONAL_SECTIONS:
        name = constant_name(f"has_{section}")
        out[name] = (section, f"inline constexpr bool {name} = {'true' if data.get(section) is not None else 'false'};")
    kinds = data.get("layers", {}).get("kinds")
    if not isinstance(kinds, list) or not kinds or not all(isinstance(k, dict) for k in kinds):
        raise DescriptorError("layers.kinds: not a non-empty list of objects")
    keys = sorted(kinds[0])
    if any(sorted(k) != keys for k in kinds):
        raise DescriptorError("layers.kinds: entries with different keys")
    for key in keys:
        add(f"layer.{key}", [k[key] for k in kinds])
    out["kFamily"] = ("family", f"inline constexpr uint32_t kFamily = {family(data)};")
    return out


def family(data: dict) -> int:
    """The CERE_FAMILY_* value of the descriptor's `model_type`."""
    node = data.get("model_type")
    model_type = node.get("value") if isinstance(node, dict) else None
    if model_type not in FAMILIES:
        raise DescriptorError(f"model_type: {model_type!r} is no family ({', '.join(sorted(FAMILIES))})")
    return FAMILIES[model_type]


def render(data: dict) -> str:
    """The header text of one descriptor."""
    decls = constants(data)
    lines = [
        f"// Generated by executor/tools/gen_model_header.py from executor/models/{data['model_id']}.json",
        "// (tasks/selftune-spec.md D1, ST1a). Do not edit: the build regenerates it from the descriptor.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "// kFamily for the preprocessor (CERE_FAMILY_* of include/cere_executor.h).",
        f"#define CERE_GEN_FAMILY {family(data)}",
        "",
        "namespace cere::gen {",
        "",
    ]
    lines += [decls[name][1] for name in sorted(decls)]
    lines += ["", "}  // namespace cere::gen", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        if src.stem != data.get("model_id"):
            raise DescriptorError(f"{src.name}: file name is not <model_id>.json ({data.get('model_id')!r})")
        text = render(data)
    except (OSError, ValueError) as e:
        print(f"gen_model_header: {e}", file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(text, encoding="ascii")
    os.replace(tmp, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
