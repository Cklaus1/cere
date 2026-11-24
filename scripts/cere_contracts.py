"""Strict wire/fixture contracts. CPU tooling, not a production runtime."""
from __future__ import annotations
import ast
import hashlib
import ipaddress
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any
from jsonschema import Draft202012Validator, validators

# Wire uints must have the same JSON representation as Rust u64.
StrictValidator = validators.extend(Draft202012Validator, type_checker=
    Draft202012Validator.TYPE_CHECKER.redefine("integer", lambda _checker, value: type(value) is int))

ROOT = Path(__file__).resolve().parents[1]
MAX_U64 = (1 << 64) - 1

class ContractError(ValueError):
    """Input violates the documented contract."""

class ArtifactError(ContractError):
    """Artifact/header/token-map rejection carrying one class of the closed list ERROR_CLASSES (M2.1)."""
    def __init__(self, cls: str, message: str):
        if cls not in ALL_ERROR_CLASSES:
            raise ValueError(f"unknown artifact error class: {cls}")
        super().__init__(f"{cls}: {message}")
        self.cls = cls

# Closed list shared with cerec through tests/fixtures/artifact_invalid.json (M2.1 Q19).
ERROR_CLASSES = {
    "header": ("header_bounds", "header_utf8", "header_json", "duplicate_key", "metadata", "tensor_fields", "dtype",
               "shape", "too_many_tensors", "byte_size", "offsets"),
    "artifact": ("manifest_size", "manifest_digest", "manifest_noncanonical", "manifest_schema", "manifest_rules",
                 "unsafe_path", "too_many_files", "unlisted_file", "symlink", "special_file", "file_size",
                 "header_hash", "family_unmatched", "family_ambiguous", "family_dtype", "family_shape",
                 "expert_metadata", "group_nvfp4", "group_ple_table", "group_ple_module", "family_totals",
                 "file_hash", "stamp"),
    "token_map": ("zip_structure", "zip_unsupported", "zip_entry", "pickle_grammar", "token_range", "token_order",
                  "token_empty"),
    "inventory": ("source_tree",),
}
ALL_ERROR_CLASSES = frozenset(c for v in ERROR_CLASSES.values() for c in v)

# Q15 bounds; tests override them per case.
DEFAULT_LIMITS = {"max_header_bytes": 16 * 1024 * 1024, "max_tensors_per_file": 1_000_000,
                  "max_total_tensors": 4_000_000, "max_files": 16_384, "max_manifest_bytes": 4 * 1024 * 1024,
                  "max_token_map_pkl_bytes": 16 * 1024 * 1024}

# M2.8 D5 (crates/cere-core/src/capabilities.rs JSON_SCHEMA_DIALECT / JSON_SCHEMA_KEYWORD_CANDIDATES): the one dialect
# a json_schema profile advertises and the keywords it may list (enforcement candidates + annotations; never oneOf).
JSON_SCHEMA_DIALECT = "json-schema-2020-12/xgrammar-0.2.1"
JSON_SCHEMA_DIALECT_LLAMACPP = "json-schema-2020-12/llamacpp-162e2bb2e"  # AmdAmd PAR-3, amd-amd design note
JSON_SCHEMA_KEYWORDS_LLAMACPP = ("type", "properties", "required", "additionalProperties", "items", "enum", "const",
                                 "anyOf", "allOf", "$ref", "$defs", "minItems", "maxItems", "minLength", "maxLength",
                                 "title", "description", "default")
JSON_SCHEMA_KEYWORDS = ("type", "properties", "required", "additionalProperties", "items", "prefixItems", "enum",
                        "const", "anyOf", "allOf", "$ref", "$defs", "minItems", "maxItems", "minLength", "maxLength",
                        "pattern", "format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "title",
                        "description", "default")
GRAMMAR_OUTPUTS = ("json_schema", "regex", "ebnf", "tool_calls_constrained")

# m23 design note PI-01/PI-02: the lever registry, single source of truth crates/cere-runtime/src/levers.rs; a Rust
# test pins this fixture to it, so a profile's `[deployments.native.levers]` is checked here against the same table.
LEVER_REGISTRY = "tests/fixtures/lever_registry.json"
# The speculation levers: a non-default implementation needs a speculating deployment (LeverSet::from_table); they
# leave a request that does not speculate unchanged. grammar-spec design note GS-04: `grammar_drafts` "on" lets a
# constrained request speculate, so it needs the drafts too.
SPEC_LEVERS = ("drafter", "draft_tokens", "draft_policy", "grammar_drafts")

def lever_registry() -> list[dict]:
    """The registry's levers in registry order: name, slot, layer, class, default, impls (id, numerics, contract, arch,
    domain, workspace, requires, params, proven_in, ...)."""
    return load_json(ROOT / LEVER_REGISTRY)["levers"]

def numerics_classes() -> list[dict]:
    """SelfTune E4: the registered numerics classes (name, reference profile, changing selections)."""
    return load_json(ROOT / LEVER_REGISTRY)["classes"]

def device_arch(device_uuid: str) -> str:
    """The arch a `device_uuid` names (Rust: levers.rs `Target::arch_of`): `hip:<ordinal>` is the AMD HIP port's
    gfx1151, `vk:<ordinal>` the Vulkan executor's vulkan (one arch per API), a `GPU-` UUID the CUDA library's
    sm_120a."""
    if device_uuid.startswith("hip:"):
        return "gfx1151"
    return "vulkan" if device_uuid.startswith("vk:") else "sm_120a"

def check_levers(p: str, native: dict, max_rows: int | None = None) -> None:
    """`native.levers` names only registered levers and implementation ids (fail-closed), and every selected
    implementation (defaults filled) gets what it `requires` (the speculation levers: speculative_draft_tokens = 4).
    SelfTune D3 (Rust: levers.rs `LeverSet::check_target`): each selected implementation is proven on the arch of
    every device the deployment names and, with `max_rows` (the profile's decode_batch_max), covers its decode rows and
    `native.max_context_tokens`. SdarOptIn (Rust: `Family::admits`): a lever scoped to another family than
    `native.family` (`sdar_gemm` on qwen4_exp) must be its default, which that library never receives, so no other
    rule applies to it. S10: the shared impls' domain is the Qwen model entry's; an inkling2_loop library runs none of
    those kernels (its own forward bounds rows and context), so the domain does not bind it."""
    registry = {lever["name"]: lever for lever in lever_registry()}
    levers = native.get("levers", {})
    for name, impl in levers.items():
        lever = registry.get(name)
        if lever is None:
            raise ContractError(f"{p}.native.levers.{name}: unknown lever (registered: {', '.join(registry)})")
        ids = [i["id"] for i in lever["impls"]]
        if impl not in ids:
            raise ContractError(f"{p}.native.levers.{name}: unknown implementation {impl!r} "
                                f"(registered: {', '.join(ids)})")
    uuids = [native["device_uuid"]] if "device_uuid" in native else \
        [r["device_uuid"] for r in native.get("replicas", [])]
    arches = sorted({device_arch(u) for u in uuids})
    context = native.get("max_context_tokens", 0)
    family = native.get("family", "qwen4_exp")
    for name, lever in registry.items():
        impl = levers.get(name, lever["default"])
        if lever.get("family", family) != family:
            if impl != lever["default"]:
                raise ContractError(f"{p}.native.levers.{name}: {impl!r} is not built for family {family}")
            continue
        if family == "inkling2_loop":
            # Inkling S13 (Rust: `LeverSet::check_target`): an inkling2_loop library runs its own kernels on every arch
            # it is built for and ignores the shared slots (each at its default), so the Qwen impls' arch, domain and
            # requirements do not apply to it.
            continue
        spec = next(i for i in lever["impls"] if i["id"] == impl)
        for arch in arches:
            if arch not in spec["arch"]:
                raise ContractError(f"{p}.native.levers.{name}: {impl!r} is not proven on {arch}")
        domain = spec["domain"]
        unbound = "family" not in lever and family == "inkling2_loop"  # a Qwen kernel that library does not run
        if max_rows is not None and not unbound and (max_rows > domain["max_rows"] or context > domain["max_context"]):
            raise ContractError(f"{p}.native.levers.{name}: {impl!r} runs at most {domain['max_rows']} rows and "
                                f"{domain['max_context']} context tokens, not {max_rows} and {context}")
        for need in spec["requires"]:
            want = need["speculative_draft_tokens"]
            if native.get("speculative_draft_tokens", 0) != want:
                raise ContractError(f"{p}.native.levers.{name} {impl!r} needs speculative_draft_tokens = {want}")

def check_ab(p: str, native: dict) -> None:
    """tasks/ab-split-spec.md AB-01/AB-02 (Rust: config.rs `validate_ab`, ab.rs `check_arm_lever`): arm ids are
    unique, the weights sum above 0, and every arm lever is a registered runtime-layer lever (an executor-layer one
    is process-global) at a registered bit_identical implementation (a changing one would change live outputs),
    runnable by the deployment's drafts (the speculation levers need speculative_draft_tokens = 4)."""
    ab = native.get("ab")
    if ab is None:
        return
    registry = {lever["name"]: lever for lever in lever_registry()}
    seen = set()
    for i, arm in enumerate(ab["arms"]):
        at = f"{p}.native.ab.arms[{i}]"
        if arm["id"] in seen:
            raise ContractError(f"{at}.id {arm['id']!r} is a duplicate arm id")
        seen.add(arm["id"])
        for name, impl in arm.get("levers", {}).items():
            lever = registry.get(name)
            if lever is None:
                raise ContractError(f"{at}.levers.{name}: unknown lever")
            if lever["layer"] != "runtime":
                raise ContractError(f"{at}.levers.{name}: an executor-layer lever is process-global; an arm takes "
                                    "runtime-layer levers only")
            numerics = {x["id"]: x["numerics"] for x in lever["impls"]}
            if impl not in numerics:
                raise ContractError(f"{at}.levers.{name}: unknown implementation {impl!r} "
                                    f"(registered: {', '.join(numerics)})")
            if numerics[impl] != "bit_identical":
                raise ContractError(f"{at}.levers.{name}: {impl!r} is a changing implementation; an arm takes "
                                    "bit-identical implementations only")
            if name in SPEC_LEVERS and impl != lever["default"] and native.get("speculative_draft_tokens", 0) != 4:
                raise ContractError(f"{at}.levers.{name} {impl!r} needs speculative_draft_tokens = 4")
    if sum(arm["weight"] for arm in ab["arms"]) == 0:
        raise ContractError(f"{p}.native.ab.arms weights must sum above 0")

def check_loop_guard(p: str, native: dict) -> None:
    """tasks/loop-guard-spec.md LG-03 (Rust: config.rs `validate_loop_guard`; the schema holds the per-key bounds):
    at least one detector on and, with LG-D1 on (min_repeats > 0), min_period_bytes <= max_period_bytes and
    min_repeats x max_period_bytes <= 65536. Absent keys take the Rust defaults."""
    guard = native.get("loop_guard")
    if guard is None:
        return
    at = f"{p}.native.loop_guard"
    min_repeats = guard.get("min_repeats", 4)
    if min_repeats == 0 and guard.get("tool_repeat_run", 10) == 0:
        raise ContractError(f"{at} turns both detectors off (min_repeats = 0, tool_repeat_run = 0); "
                            "remove the table instead")
    if min_repeats > 0:
        max_period = guard.get("max_period_bytes", 1024)
        if guard.get("min_period_bytes", 16) > max_period:
            raise ContractError(f"{at}.min_period_bytes must be at most max_period_bytes")
        if min_repeats * max_period > 65536:
            raise ContractError(f"{at}: min_repeats x max_period_bytes must be at most 65536")

def _test_defined(path: Path, cls: str | None, test: str) -> bool:
    """`test` is a test function of `path` (a method of class `cls` when given): Python by its syntax tree, C/C++
    by a definition `<type> test(` at the start of a line."""
    text = path.read_text()
    if path.suffix != ".py":
        return cls is None and re.search(rf"^[A-Za-z_][\w:<>*& ]*\b{re.escape(test)}\s*\(", text, re.M) is not None
    tree = ast.parse(text)
    scopes = [tree] if cls is None else [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    return any(isinstance(n, ast.FunctionDef) and n.name == test for s in scopes for n in s.body)

def lever_contract_errors(levers: list[dict], root: Path = ROOT) -> list[str]:
    """tasks/m23-spec.md PI-01: every registered implementation names its test contract, `<repo path>::<test>` or
    `<repo path>::<Class>::<test>`, and that test exists (an implementation registered without its contract fails
    CI). The bit_identical class's contract is its ktest equal to the default; a changing implementation is opt-in
    (never the default) and must be listed for the batched D9 (`d9_batch`) unless its lever is restart-class
    (another library). A shape-dependent "auto" implementation (`picks`) may pick only registered bit_identical
    implementations, which are bit-identical to each other, so every row's output stays independent of the batch."""
    errors = []
    for lever in levers:
        numerics = {impl["id"]: impl["numerics"] for impl in lever["impls"]}
        if numerics.get(lever["default"]) != "bit_identical":
            errors.append(f"{lever['name']}: the default {lever['default']!r} must be a registered bit_identical "
                          "implementation")
        for impl in lever["impls"]:
            where = f"{lever['name']}/{impl['id']}"
            parts = impl.get("contract", "").split("::")
            path = root / parts[0] if parts[0] else None
            if len(parts) not in (2, 3) or path is None or not path.is_file():
                errors.append(f"{where}: contract {impl.get('contract')!r} names no test file")
            elif not _test_defined(path, parts[1] if len(parts) == 3 else None, parts[-1]):
                errors.append(f"{where}: contract {impl['contract']!r} names no test")
            if impl["numerics"] == "changing" and not impl["d9_batch"] and lever["class"] != "restart":
                errors.append(f"{where}: a changing implementation must be listed for the batched D9 (d9_batch)")
            for pick in impl.get("picks", []):
                if pick == impl["id"] or numerics.get(pick) != "bit_identical" or impl["numerics"] != "bit_identical":
                    errors.append(f"{where}: an auto implementation may pick only registered bit_identical "
                                  f"implementations, not {pick!r} (row invariance)")
    return errors

def limits_with(overrides: dict | None) -> dict:
    out = dict(DEFAULT_LIMITS)
    for k, v in (overrides or {}).items():
        if k not in out or type(v) is not int or v < 0:
            raise ValueError(f"invalid limit {k}")
        out[k] = v
    return out

def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ArtifactError("duplicate_key", f"duplicate JSON key: {key}")
        out[key] = value
    return out

def json_int(text: str) -> int | float:
    """`parse_int` hook: `-0` is the float -0.0, as serde_json reads it (cerec), so it is no integer here either."""
    return -0.0 if text == "-0" else int(text)

def parse_json(text: str | bytes) -> Any:
    def invalid(value):
        raise ContractError(f"non-finite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=invalid, parse_int=json_int)

def load_json(path: Path) -> Any:
    return parse_json(path.read_bytes())

def canonical_digest(value: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, ensure_ascii=False).encode()).hexdigest()

def safe_relative(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ContractError("invalid artifact path")
    raw = path.split("/")
    if path.startswith("/") or any(x in ("", ".", "..") for x in raw) or ":" in path:
        raise ContractError(f"unsafe artifact path: {path!r}")
    return PurePosixPath(path)

def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("non-finite number")
    if isinstance(value, dict):
        for x in value.values(): _finite(x)
    elif isinstance(value, list):
        for x in value: _finite(x)

def _schema_errors(name: str, data: Any, root: Path) -> None:
    validator = StrictValidator(load_json(root / "schemas" / f"{name}.schema.json"))
    errors = sorted(validator.iter_errors(data), key=lambda e: str(list(e.absolute_path)))
    if errors:
        e = errors[0]
        raise ContractError(f"{name} {list(e.absolute_path)}: {e.message}")

# selftune design note D1/E1 (SelfTune ST0): one native model descriptor per model under executor/models, named
# `<model_id>.json`. ST1a: the build generates `cere_model_constants.h` (namespace cere::gen) from BUILD_DESCRIPTOR with
# GEN_MODEL_HEADER; executor/CMakeLists.txt and the Makefile host build name both literally.
MODEL_DESCRIPTOR_DIR = "executor/models"
BUILD_DESCRIPTOR = "executor/models/qwen38-flash-next.json"
GEN_MODEL_HEADER = "executor/tools/gen_model_header.py"
_EXECUTOR_LITERAL = re.compile(r"^\s*(?:static\s+)?(?:inline\s+)?constexpr\s+[\w:\s]+?\b(k\w+)\s*=\s*([^;]+);")
_INT_LITERAL = re.compile(r"[+-]?\d+[uUlL]*")
_FLOAT_LITERAL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fF]?")
# An executor constant that reads the generated header: `gen::kName`, optionally under one arithmetic static_cast.
_GENERATED_REF = re.compile(r"(?:static_cast<(?:int|int64_t|uint32_t|uint64_t|float|double)>\()?gen::(k\w+)\)?")

def gen_model_header(root: Path = ROOT):
    """The generator module (executor/tools/gen_model_header.py, stdlib only), loaded from `root`."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_model_header", root / GEN_MODEL_HEADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def generated_constants(root: Path = ROOT) -> dict:
    """Constant name -> descriptor value of the header the build generates from BUILD_DESCRIPTOR (facts and plain
    fields; the per-layer arrays and kHas<Section> flags are left out)."""
    data = load_json(root / BUILD_DESCRIPTOR)
    out = {}
    for name, (field, _decl) in gen_model_header(root).constants(data).items():
        node = data
        for key in field.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and set(node) == {"value", "sources"}:
            node = node["value"]
        if node is not None and not isinstance(node, dict):
            out[name] = node
    return out

_GEN_USE = re.compile(r"\bgen::(k\w+)")
_HEADER_NAME = "cere_model_constants.h"

def model_header_check(root: Path = ROOT) -> int:
    """ST1a (D1): the generated constants header cannot drift from the descriptors. Every descriptor renders, the same
    bytes twice, ASCII with no repo path in it; no copy of the header is committed under executor/ (the build
    regenerates it); executor/CMakeLists.txt's add_custom_command names the generator literally and the descriptor as
    models/${CERE_MODEL}.json (ST4 S3a: a library is built for one model), both as dependencies, with CERE_MODEL
    defaulting to BUILD_DESCRIPTOR's id; the Makefile host rule names the generator and BUILD_DESCRIPTOR literally;
    and every `gen::kName` under executor/cuda/src is a constant the header of BUILD_DESCRIPTOR defines. Returns the
    number of `gen::` references checked."""
    gen = gen_model_header(root)
    for p in sorted((root / MODEL_DESCRIPTOR_DIR).glob("*.json")):
        data = load_json(p)
        try:
            text = gen.render(data)
        except ValueError as e:
            raise ContractError(f"{GEN_MODEL_HEADER}: {p.name}: {e}") from None
        if text != gen.render(load_json(p)) or not text.isascii() or str(root) in text:
            raise ContractError(f"{GEN_MODEL_HEADER}: the header of {p.name} is not byte-deterministic and path-free")
    committed = sorted(str(p.relative_to(root)) for p in (root / "executor").rglob(_HEADER_NAME))
    if committed:
        raise ContractError(f"{_HEADER_NAME} is generated by the build, never committed: {committed}")
    cmake = (root / "executor/CMakeLists.txt").read_text()
    command = re.search(r"add_custom_command\(OUTPUT [^)]*" + re.escape(_HEADER_NAME) + r"[^)]*\)", cmake)
    rel = lambda p: "${CMAKE_CURRENT_SOURCE_DIR}/" + p.removeprefix("executor/")
    descriptor = "${CMAKE_CURRENT_SOURCE_DIR}/models/${CERE_MODEL}.json"
    default = re.search(r"^set\(CERE_MODEL ([a-z0-9.-]+) CACHE STRING ", cmake, re.M)
    if (command is None or command.group(0).count(rel(GEN_MODEL_HEADER)) != 2
            or command.group(0).count(descriptor) != 2 or default is None
            or f"{MODEL_DESCRIPTOR_DIR}/{default.group(1)}.json" != BUILD_DESCRIPTOR):
        raise ContractError(f"executor/CMakeLists.txt: no add_custom_command for {_HEADER_NAME} that runs and depends on "
                            f"{GEN_MODEL_HEADER} and executor/models/${{CERE_MODEL}}.json, with CERE_MODEL defaulting "
                            f"to {BUILD_DESCRIPTOR}")
    makefile = (root / "Makefile").read_text()
    if not re.search(r"^\$\(HOST_MODEL_HEADER\): " + re.escape(f"{GEN_MODEL_HEADER} {BUILD_DESCRIPTOR}") + r"\n\t.*"
                     + re.escape(f"python3 {GEN_MODEL_HEADER} {BUILD_DESCRIPTOR} $@"), makefile, re.M):
        raise ContractError(f"Makefile: the $(HOST_MODEL_HEADER) rule must run and depend on {GEN_MODEL_HEADER} "
                            f"{BUILD_DESCRIPTOR}")
    # ST4 S3a: code under `#if CERE_GEN_FAMILY == CERE_FAMILY_<F>` (or its `#else`) is compiled only into libraries of
    # those families, so its references are checked against every descriptor of them; the rest against BUILD_DESCRIPTOR.
    # ST4 S3c: a file in a family's own subdirectory (`kernels/sdar/`, `model/sdar/`, ...; executor/CMakeLists.txt
    # compiles them only into that family's library) is checked against that family's descriptors as a whole.
    families: dict[str, list[tuple[str, set]]] = {}
    for p in sorted((root / MODEL_DESCRIPTOR_DIR).glob("*.json")):
        data = load_json(p)
        families.setdefault(_descriptor_family(data), []).append((f"{MODEL_DESCRIPTOR_DIR}/{p.name}",
                                                                  set(gen.constants(data))))
    build = (BUILD_DESCRIPTOR, set(gen.constants(load_json(root / BUILD_DESCRIPTOR))))
    count = 0
    for p in sorted((root / "executor/cuda/src").rglob("*")):
        if p.suffix not in (".h", ".cuh", ".cu", ".cpp") or not p.is_file():
            continue
        dirs = p.relative_to(root / "executor/cuda/src").parts[:-1]
        own = {f for f in families if FAMILY_DIRS.get(f, f.lower()) in dirs}
        for line, allowed in _family_lines(p.read_text(encoding="latin-1"), set(families)):
            allowed = own or allowed
            for name in _GEN_USE.findall(line):
                count += 1
                checks = [build] if allowed is None else [d for f in sorted(allowed) for d in families.get(f, [])]
                for descriptor, defined in checks:
                    if name not in defined:
                        raise ContractError(f"{p.relative_to(root)}: gen::{name} is not a constant of {descriptor}'s "
                                            "header")
    return count

# The CERE_FAMILY_<F> suffix of a model_type when it is not the model_type upper-cased, and the family's own source
# subdirectory (`kernels/<dir>/`, ...) when it is not the suffix lower-cased (Inkling 2 Mini, inkling design note S0).
FAMILY_MACROS = {"inkling2_loop": "INKLING2"}
FAMILY_DIRS = {"INKLING2": "inkling"}

def _descriptor_family(data: dict) -> str:
    """The CERE_FAMILY_<F> suffix of a descriptor's `model_type` (`qwen4_exp` -> QWEN4_EXP, `inkling2_loop` ->
    INKLING2)."""
    node = data.get("model_type")
    model_type = str((node.get("value") if isinstance(node, dict) else None) or "")
    return FAMILY_MACROS.get(model_type, model_type.upper())

_DIRECTIVE = re.compile(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)")
_FAMILY_GUARD = re.compile(r"\s*CERE_GEN_FAMILY\s*==\s*CERE_FAMILY_(\w+)\s*(?://.*)?")

def _family_lines(text: str, families: set):
    """Every non-directive line of a C/C++ source as (line, the families whose libraries compile it), the second
    None outside any `#if CERE_GEN_FAMILY == CERE_FAMILY_<F>` branch (an `#else` admits the other families)."""
    frames: list[set | None] = []  # one per open #if: the families it admits, None for any other condition
    for line in text.splitlines():
        directive = _DIRECTIVE.match(line)
        if directive is None:
            admitted = [f for f in frames if f is not None]
            yield line, set.intersection(*admitted) if admitted else None
            continue
        kind, condition = directive.groups()
        guard = _FAMILY_GUARD.fullmatch(condition)
        if kind in ("if", "ifdef", "ifndef"):
            frames.append({guard.group(1)} if guard and kind == "if" else None)
        elif kind == "elif" and frames:
            frames[-1] = {guard.group(1)} if guard else None
        elif kind == "else" and frames and frames[-1] is not None:
            frames[-1] = families - frames[-1]
        elif kind == "endif" and frames:
            frames.pop()

def _descriptor_facts(data: dict):
    """Every `{value, sources}` fact of a model descriptor as (dotted field, fact), in document order."""
    def walk(node, path):
        if isinstance(node, dict):
            if set(node) == {"value", "sources"}:
                yield path, node
                return
            for key, child in node.items():
                yield from walk(child, f"{path}.{key}" if path else key)
    yield from walk({k: v for k, v in data.items() if k not in ("layers", "checkpoint")}, "")

def executor_constant(root: Path, source: str) -> int | float:
    """The value of `executor:<repo path>#<constant>`: the one `constexpr ... <constant> = <x>;` line of that file
    that BUILD_DESCRIPTOR's library compiles (a branch of another family, ST4 S3a, is skipped), where x is a decimal
    literal or `gen::kName` (the generated header's constant, the BUILD_DESCRIPTOR value it is generated from); an
    other expression or a second definition is refused."""
    path, name = source.removeprefix("executor:").split("#")
    rel = safe_relative(path)
    file = root / rel
    if not os.path.isfile(file):
        raise ContractError(f"{source}: no such executor file")
    family = _descriptor_family(load_json(root / BUILD_DESCRIPTOR))
    families = {_descriptor_family(load_json(p)) for p in (root / MODEL_DESCRIPTOR_DIR).glob("*.json")}
    hits = [m.group(2).strip() for m in (_EXECUTOR_LITERAL.match(line) for line, allowed
                                         in _family_lines(file.read_text(), families)
                                         if allowed is None or family in allowed)
            if m and m.group(1) == name]
    if len(hits) != 1:
        raise ContractError(f"{source}: {len(hits)} constexpr definitions of {name} (need exactly one)")
    literal = hits[0]
    if _INT_LITERAL.fullmatch(literal):
        return int(literal.rstrip("uUlL"))
    if _FLOAT_LITERAL.fullmatch(literal):
        return float(literal.rstrip("fF"))
    ref = _GENERATED_REF.fullmatch(literal)
    if ref:
        generated = generated_constants(root)
        if ref.group(1) not in generated:
            raise ContractError(f"{source}: {name} = {literal}: the generated header has no {ref.group(1)}")
        return generated[ref.group(1)]
    raise ContractError(f"{source}: {name} = {literal} is not a decimal literal or a generated constant")

def _same(a: Any, b: Any) -> bool:
    """Equality that keeps booleans apart from 0/1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b

# A `config.json!<key>` source (the checkpoint's config.json has no such key) is allowed only on these fields, whose
# value must then be the field's absent value. tie_word_embeddings: a config with no tie key (Inkling 2 Mini) stores
# `embed` and `unembed` as separate tensors (transformers `_tied_weights_keys = {}`).
_ABSENT_VALUES = {"weights.quant_algo": "none", "weights.tie_word_embeddings": False}
# The layer-kind sources: config.json lists of the kinds, `use_sliding_window` false with every layer full attention
# (a config with no layer_types, as SDAR's), `local_layer_ids` (the sliding_attention layers, the rest full attention),
# or hf_quant_config.json's parameter groups (each moe layer's expert format, layers.kinds[].experts).
_QCFG_GROUPS = "hf_quant_config.json:model.parameter_groups"
_LAYER_KIND_SOURCES = ("config.json:text_config.layer_types", "config.json:text_config.ple_layer_ids",
                       "config.json:use_sliding_window", "config.json:text_config.local_layer_ids", _QCFG_GROUPS)
# hf_quant_config.json group compute_dtype -> layers.kinds[].experts (inkling design note D-I6).
_QCFG_EXPERT_FORMATS = {"nvfp4": "nvfp4", "mxfp8": "mxfp8"}

def model_descriptor_rules(data: dict, root: Path = ROOT) -> None:
    """What the schema cannot say of a model descriptor: each executor source equals its value, an absent-key source
    holds the field's absent value, and the layer list, rotary width, head groupings, graph segmentation and decode
    policy are consistent. config.json sources are checked against the checkpoint by `descriptor_config_errors`."""
    for field, fact in _descriptor_facts(data):
        for src in fact["sources"]:
            if src.startswith("config.json!"):
                if field not in _ABSENT_VALUES:
                    raise ContractError(f"{field}: absent-key source {src} on a field with no absent value")
                if not _same(fact["value"], _ABSENT_VALUES[field]):
                    raise ContractError(f"{field}: value {fact['value']!r} with absent-key source {src} "
                                        f"(absent value {_ABSENT_VALUES[field]!r})")
            if not src.startswith("executor:"):
                continue
            if not isinstance(fact["value"], (int, float)) or isinstance(fact["value"], bool):
                raise ContractError(f"{field}: executor source {src} for a non-numeric value")
            got = executor_constant(root, src)
            if not _same(got, fact["value"]):
                raise ContractError(f"{field}: value {fact['value']} differs from {src} = {got}")
    for src in data["layers"]["sources"]:
        if src not in _LAYER_KIND_SOURCES:
            raise ContractError(f"layers.sources: {src} is not a layer-kind source")
    d = {k: v["value"] for k, v in data["dimensions"].items()}
    kinds = data["layers"]["kinds"]
    if len(kinds) != d["num_hidden_layers"]:
        raise ContractError(f"layers.kinds has {len(kinds)} entries for num_hidden_layers {d['num_hidden_layers']}")
    interval = d.get("full_attention_interval")
    for i, kind in enumerate(kinds):
        full = kind["mixer"] == "full_attention"
        if kind["mixer"] == "sliding_attention":
            if interval is not None:
                raise ContractError(f"layers.kinds[{i}].mixer sliding_attention with full_attention_interval")
            continue
        if interval is None and not full:
            raise ContractError(f"layers.kinds[{i}].mixer {kind['mixer']} without full_attention_interval")
        if interval is not None and full != (interval > 0 and (i + 1) % interval == 0):
            raise ContractError(f"layers.kinds[{i}].mixer {kind['mixer']} contradicts full_attention_interval {interval}")
    if [i + 1 for i, kind in enumerate(kinds) if kind["ple"]] != d.get("ple_layer_ids", []):
        raise ContractError("layers.kinds[].ple contradicts ple_layer_ids (1-based)")
    with_experts = [i for i, kind in enumerate(kinds) if "experts" in kind]
    if with_experts and len(with_experts) != len(kinds):
        raise ContractError("layers.kinds[].experts on some layers only (every entry or none)")
    for i in with_experts:
        if kinds[i]["ffn"] != "moe":
            raise ContractError(f"layers.kinds[{i}].experts on a {kinds[i]['ffn']} layer")
    if with_experts and _QCFG_GROUPS not in data["layers"]["sources"]:
        raise ContractError(f"layers.kinds[].experts without the layer-kind source {_QCFG_GROUPS}")
    if "rotary_dim" in d and d["rotary_dim"] != d["head_dim"] * d.get("partial_rotary_factor", 1):
        raise ContractError("rotary_dim is not head_dim x partial_rotary_factor")
    if "mrope_section" in d and 2 * sum(d["mrope_section"]) != d.get("rotary_dim", 0):
        raise ContractError("mrope_section does not cover rotary_dim / 2")
    if "relative_position" in data and ("rotary_dim" in d or "rope_theta" in d):
        raise ContractError("relative_position with rotary_dim / rope_theta (a model has one position scheme)")
    loop = data.get("loop")
    if loop is not None:
        start, end, passes = (loop[k]["value"] for k in ("core_start", "core_end", "num_passes"))
        if not 0 < start < end <= d["num_hidden_layers"] or passes < 1:
            raise ContractError(f"loop core_start {start} / core_end {end} / num_passes {passes} outside "
                                f"0 < core_start < core_end <= num_hidden_layers, num_passes >= 1")
    if any(src.startswith("hf_quant_config.json:") for _, fact in _descriptor_facts(data) for src in fact["sources"]) \
            or _QCFG_GROUPS in data["layers"]["sources"]:
        if "hf_quant_config_sha256" not in data["checkpoint"]:
            raise ContractError("an hf_quant_config.json source without checkpoint.hf_quant_config_sha256")
    if d["num_key_value_heads"] == 0 or d["num_attention_heads"] % d["num_key_value_heads"]:
        raise ContractError("num_attention_heads is not a multiple of num_key_value_heads")
    if "linear_num_key_heads" in d and (d["linear_num_key_heads"] == 0
                                        or d["linear_num_value_heads"] % d["linear_num_key_heads"]):
        raise ContractError("linear_num_value_heads is not a multiple of linear_num_key_heads")
    if "num_experts" in d and not 0 < d["num_experts_per_tok"] <= d["num_experts"]:
        raise ContractError("num_experts_per_tok outside 1..num_experts")
    if data["graph"] is not None:
        seg = data["graph"]["verify_segment_layers"]["value"]
        if seg == 0 or d["num_hidden_layers"] % seg:
            raise ContractError("num_hidden_layers is not a multiple of graph.verify_segment_layers")
    head = data["draft_head"]
    if head["kind"] == "none":
        extra = sorted(set(head) - {"kind"})
        if extra:
            raise ContractError(f"draft_head.kind none with draft-head fields {extra}")
    else:
        if head["num_hidden_layers"]["value"] == 0:
            raise ContractError("draft_head.kind contradicts draft_head.num_hidden_layers")
        if len(head["layer_types"]["value"]) != head["num_hidden_layers"]["value"]:
            raise ContractError("draft_head.layer_types length differs from draft_head.num_hidden_layers")
    model_slots = data["slots"]["model"]
    decode = data.get("decode")
    if (decode is not None) != ("decode.block_diffusion" in model_slots):
        raise ContractError("slots.model decode.block_diffusion contradicts decode")
    if decode is not None:
        if decode["block_size"]["value"] < 2:
            raise ContractError("decode.block_size below 2 is autoregressive decode, not block diffusion")
        if decode["mask_token_id"]["value"] >= d["vocab_size"]:
            raise ContractError("decode.mask_token_id outside vocab_size")
        # Named refusal (SelfTune ST0b): speculative drafting over a block-diffusion decode is not defined.
        if head["kind"] != "none":
            raise ContractError("decode block_diffusion with a draft head is not expressible (draft_head.kind must be none)")

def _config_get(config: dict, dotted: str) -> tuple[bool, Any]:
    """The value at a dotted key; a decimal segment indexes a list (hf_quant_config.json's parameter groups)."""
    node: Any = config
    for key in dotted.split("."):
        if isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return False, None
    return True, node

def qcfg_expert_formats(qcfg: dict, num_layers: int) -> list[str | None]:
    """Per layer, the expert format hf_quant_config.json's parameter groups give
    `model.llm.layers.<i>.mlp.experts.w13_weight` and `.w2_weight` (None: no such tensor, or the two disagree, or a
    group with another compute_dtype)."""
    found: dict[str, str] = {}
    for group in qcfg.get("model", {}).get("parameter_groups", []):
        for name in group.get("tensors", []):
            found[name] = group.get("compute_dtype")
    out = []
    for i in range(num_layers):
        pair = {found.get(f"model.llm.layers.{i}.mlp.experts.{w}") for w in ("w13_weight", "w2_weight")}
        out.append(_QCFG_EXPERT_FORMATS.get(pair.pop()) if len(pair) == 1 else None)
    return out

def _unquantized_errors(patterns: list[str], qcfg: dict) -> list[str]:
    """weights.unquantized_modules sourced from the parameter groups: every tensor of a quantized group (compute_dtype
    nvfp4 / mxfp8) matches none of the globs, every other tensor matches one."""
    import fnmatch
    errors = []
    for group in qcfg.get("model", {}).get("parameter_groups", []):
        quantized = group.get("compute_dtype") in _QCFG_EXPERT_FORMATS
        for name in group.get("tensors", []):
            if any(fnmatch.fnmatchcase(name, p) for p in patterns) == quantized:
                errors.append(f"weights.unquantized_modules: {name} ({group.get('compute_dtype')}) is "
                              f"{'matched' if quantized else 'not matched'}")
    return errors

def descriptor_config_errors(data: dict, config: dict, qcfg: dict | None = None) -> list[str]:
    """Every `config.json:<key>` source of the descriptor against the checkpoint's parsed config.json: the key exists
    and equals the value; every `config.json!<key>` key is absent; the layer kinds equal `layer_types` and
    `ple_layer_ids` or `local_layer_ids`, or are all full attention under `use_sliding_window` false. Every
    `hf_quant_config.json:<key>` source against the parsed hf_quant_config.json `qcfg` (None: each such source is an
    error): the key equals the value, except `model.parameter_groups` itself, which sources the layers' expert
    formats and weights.unquantized_modules (the globs split the groups' tensors into quantized and not)."""
    errors = []
    for field, fact in _descriptor_facts(data):
        for src in fact["sources"]:
            if src.startswith("config.json!"):
                if _config_get(config, src.removeprefix("config.json!"))[0]:
                    errors.append(f"{field}: {src} names a key config.json has")
                continue
            if src.startswith("hf_quant_config.json:"):
                if qcfg is None:
                    errors.append(f"{field}: {src} without the checkpoint's hf_quant_config.json")
                elif src == _QCFG_GROUPS and field == "weights.unquantized_modules":
                    errors += _unquantized_errors(fact["value"], qcfg)
                else:
                    found, got = _config_get(qcfg, src.removeprefix("hf_quant_config.json:"))
                    if not found:
                        errors.append(f"{field}: {src} is absent from hf_quant_config.json")
                    elif not _same(got, fact["value"]):
                        errors.append(f"{field}: value {fact['value']!r} differs from {src} = {got!r}")
                continue
            if not src.startswith("config.json:"):
                continue
            found, got = _config_get(config, src.removeprefix("config.json:"))
            if not found:
                errors.append(f"{field}: {src} is absent from config.json")
            elif not _same(got, fact["value"]):
                errors.append(f"{field}: value {fact['value']!r} differs from {src} = {got!r}")
    kinds = data["layers"]["kinds"]
    for src in data["layers"]["sources"]:
        if src == _QCFG_GROUPS:
            if qcfg is None:
                errors.append(f"layers: {src} without the checkpoint's hf_quant_config.json")
            elif qcfg_expert_formats(qcfg, len(kinds)) != [k.get("experts") for k in kinds]:
                errors.append(f"layers.kinds[].experts differs from {src}")
            continue
        found, got = _config_get(config, src.removeprefix("config.json:"))
        if not found:
            errors.append(f"layers: {src} is absent from config.json")
        elif src.endswith(".layer_types") and got != [k["mixer"] for k in kinds]:
            errors.append(f"layers.kinds[].mixer differs from {src}")
        elif src.endswith(".ple_layer_ids") and got != [i + 1 for i, k in enumerate(kinds) if k["ple"]]:
            errors.append(f"layers.kinds[].ple differs from {src}")
        elif src.endswith(".local_layer_ids") and (
                got != [i for i, k in enumerate(kinds) if k["mixer"] == "sliding_attention"]
                or any(k["mixer"] not in ("sliding_attention", "full_attention") for k in kinds)):
            errors.append(f"layers.kinds[].mixer differs from {src} (sliding_attention there, full_attention elsewhere)")
        elif src.endswith(":use_sliding_window") and (got is not False
                                                      or any(k["mixer"] != "full_attention" for k in kinds)):
            errors.append(f"layers.kinds[].mixer is not full attention everywhere under {src} = {got!r}")
    return errors

def model_descriptor_checkpoint(data: dict, config_path: str | None = None, root: Path = ROOT) -> bool:
    """Check a descriptor against its checkpoint's config.json when the file is readable here; False when it is not
    (CI runs non-root without /var/lib/cere/models: lessons 2026-10-06 M2.1 PR CI). The file is `config_path`, else the
    committed `checkpoint.config_json_fixture` (always checked: missing refuses, and its source.json must name the
    descriptor's hub repo, revision and sha256), else `checkpoint.reference_path`. A file whose sha256 differs from
    `checkpoint.config_json_sha256`, or any contradicting source, refuses."""
    ckpt = data["checkpoint"]
    if config_path:
        path = config_path
    elif "config_json_fixture" in ckpt:
        fixture = root / safe_relative(ckpt["config_json_fixture"])
        if not fixture.is_file():
            raise ContractError(f"model descriptor {data['model_id']}: {ckpt['config_json_fixture']} is missing")
        source = load_json(fixture.parent / "source.json")
        for key, want in (("repo", ckpt["hub"]["repo"]), ("revision", ckpt["hub"]["revision"]),
                          ("sha256", ckpt["config_json_sha256"])):
            if source.get(key) != want:
                raise ContractError(f"model descriptor {data['model_id']}: {fixture.parent.name}/source.json {key} "
                                    f"{source.get(key)!r} differs from the descriptor's {want!r}")
        pinned = source.get("files", {}).get("config.json", {}).get("sha256", ckpt["config_json_sha256"])
        if pinned != ckpt["config_json_sha256"]:
            raise ContractError(f"model descriptor {data['model_id']}: {fixture.parent.name}/source.json files "
                                f"config.json sha256 {pinned!r} differs from the descriptor's")
        path = str(fixture)
    else:
        path = ckpt["reference_path"]
    if not (os.path.isfile(path) and os.access(path, os.R_OK)):
        return False
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ckpt["config_json_sha256"]:
        raise ContractError(f"model descriptor {data['model_id']}: {path} sha256 differs from checkpoint.config_json_sha256")
    qcfg = None
    if "hf_quant_config_sha256" in ckpt:
        # D-I6 (inkling design note): the hf_quant_config.json beside that config.json, pinned by sha256.
        qpath = Path(path).parent / "hf_quant_config.json"
        if not qpath.is_file():
            raise ContractError(f"model descriptor {data['model_id']}: {qpath} is missing")
        qraw = qpath.read_bytes()
        if hashlib.sha256(qraw).hexdigest() != ckpt["hf_quant_config_sha256"]:
            raise ContractError(f"model descriptor {data['model_id']}: {qpath} sha256 differs from "
                                "checkpoint.hf_quant_config_sha256")
        qcfg = parse_json(qraw)
    errors = descriptor_config_errors(data, parse_json(raw), qcfg)
    if errors:
        raise ContractError(f"model descriptor {data['model_id']}: {errors[0]}")
    return True

def validate(name: str, data: dict, root: Path = ROOT, *, limits: dict | None = None) -> None:
    _finite(data)
    _schema_errors(name, data, root)
    if name == "execution_artifact":
        # The embedded profile and family rules are part of the manifest schema (manifest_schema class).
        _schema_errors("artifact_profile", data["profile"], root)
        _schema_errors("artifact_families", data["family_rules"], root)
    if name == "model_descriptor":
        model_descriptor_rules(data, root)
    elif name == "state_object":
        if not data["immutable"] and data["sharing_count"] > 1:
            raise ContractError("mutable state cannot have multiple sharers")
    elif name == "resource_graph":
        ids = [x["id"] for x in data["nodes"]]
        if len(ids) != len(set(ids)): raise ContractError("duplicate resource id")
        for link in data["links"]:
            if link["from"] not in ids or link["to"] not in ids:
                raise ContractError("unknown link endpoint")
            if link["from"] == link["to"]: raise ContractError("self link")
    elif name == "worker_node":
        if data["lifecycle_state"] == "ready" and (not data["reconciled"] or data["desired_state"] != "ready"):
            raise ContractError("ready worker is unreconciled or operator-draining")
        if data["lifecycle_state"] == "lost" and data["model_instance_ids"]:
            raise ContractError("lost worker cannot advertise authoritative resident models")
        if data["transport"] == "local" and data["labels"].get("network") == "tailscale":
            raise ContractError("local worker cannot claim remote network transport")
    elif name == "cluster_config":
        cp = data["control_plane"]
        if data["scheduling"]["allow_remote_workers"] and cp["transport"] == "local":
            raise ContractError("remote workers require remote transport")
        if cp["admission_lease_ms"] >= cp["lost_after_ms"]:
            raise ContractError("admission lease must expire before lost threshold")
        if not data["security"]["coordinator_principals"] or not data["security"]["worker_bindings"]:
            raise ContractError("explicit coordinator principals and worker bindings required")
        if cp["suspect_after_ms"] <= cp["heartbeat_interval_ms"]:
            raise ContractError("suspect threshold must exceed heartbeat interval")
        if cp["lost_after_ms"] <= cp["suspect_after_ms"]:
            raise ContractError("lost threshold must exceed suspect threshold")
        if cp["transport"] == "tailscale_grpc" and not data["security"]["require_tailscale_policy"]:
            raise ContractError("tailscale control plane requires deployment policy")
        if data["data_plane"]["mode"] == "future_accelerated" and not data["scheduling"]["allow_cross_node_cooperative_inference"]:
            raise ContractError("accelerated cross-node data plane cannot be enabled in v1 contract")
    elif name == "execution_capabilities":
        if data["decode_batch_min"] > data["decode_batch_max"]:
            raise ContractError("batch minimum exceeds maximum")
        mode = data["execution_mode"]
        native_flags = [v for k,v in data.items() if k.startswith("native_")]
        if mode in ("serial", "unsupported") and any(native_flags):
            raise ContractError("serial/unsupported profile cannot claim native execution")
        if mode == "serial" and data["decode_batch_max"] != 1:
            raise ContractError("serial native row capacity must be one")
        if mode == "unsupported" and data["operations"]:
            raise ContractError("unsupported profile advertises operations")
        if data["native_decode_batch"] and "decode" not in data["operations"]:
            raise ContractError("native decode without decode operation")
        if data["native_speculative_batch"] and "verify" not in data["operations"]:
            raise ContractError("speculation without verify capability")
        if data["checkpoint_mode"] == "exact" and not data["native_state_snapshot"]:
            raise ContractError("exact checkpoint without snapshot support")
        has_schema = "json_schema" in data["structured_outputs"]
        dialect_keywords = {JSON_SCHEMA_DIALECT: JSON_SCHEMA_KEYWORDS,
                            JSON_SCHEMA_DIALECT_LLAMACPP: JSON_SCHEMA_KEYWORDS_LLAMACPP}.get(data["json_schema_dialect"])
        if has_schema and (dialect_keywords is None or not data["json_schema_keywords"]):
            raise ContractError("json_schema capability requires the D5 dialect and keywords")
        if has_schema and not set(data["json_schema_keywords"]) <= set(dialect_keywords):
            raise ContractError("json_schema keyword outside the D5 candidates")
        if not has_schema and (data["json_schema_dialect"] is not None or data["json_schema_keywords"]):
            raise ContractError("schema dialect/keywords advertised without json_schema output")
        # M2.7 D11: a native sampled row needs cered's sampler, which a native profile declares with sampling_params.
        if mode == "native" and not data.get("sampling_params") and any(
                c["sampler_mode"] == "sampled" for c in data["validated_combinations"]):
            raise ContractError("native sampled combination lacks sampling_params")
        combinations = data["validated_combinations"]
        seen = set()
        for c in combinations:
            key = canonical_digest(c)
            if key in seen: raise ContractError("duplicate capability combination")
            seen.add(key)
            for field, pool in (("operation", "operations"), ("modality", "modalities"), ("output", "structured_outputs"), ("numerical_mode", "numerical_modes")):
                if c[field] not in data[pool]: raise ContractError("combination exceeds profile declaration")
            if c["max_context_tokens"] > data["max_context_tokens"]:
                raise ContractError("combination context exceeds profile")
            if any(not data["decode_batch_min"] <= n <= data["decode_batch_max"] for n in c["row_counts"]):
                raise ContractError("combination row count exceeds profile")
            # M2.4 D13: tool calls are text decoding plus a parser, so each one needs its text twin. M2.13 D5: an
            # image decode combination needs the same text twin.
            twin = any(
                t["operation"] == "decode" and t["modality"] == "text" and t["output"] == "text"
                and t["sampler_mode"] == c["sampler_mode"] and t["row_counts"] == c["row_counts"]
                and t["max_context_tokens"] == c["max_context_tokens"] for t in combinations)
            if c["operation"] == "decode" and c["output"] == "tool_calls" and not twin:
                raise ContractError("tool_calls decode combination lacks the matching text decode combination")
            # M2.8 D10/I3: constrained tickets never speculate, so no grammar output pairs with verify.
            if c["operation"] == "verify" and c["output"] in GRAMMAR_OUTPUTS:
                raise ContractError("grammar output combined with verify")
            if c["operation"] == "decode" and c["modality"] == "image" and c["output"] == "text" and not twin:
                raise ContractError("image decode combination lacks the matching text decode combination")
        # M2.13 D5: `modalities` lists `image` iff an image decode combination exists (the converse is above).
        if "image" in data["modalities"] and not any(
                c["operation"] == "decode" and c["modality"] == "image" for c in combinations):
            raise ContractError("modalities lists image without an image decode combination")
    elif name == "execution_artifact":
        execution_artifact_rules(data, limits_with(limits))
    elif name == "artifact_families":
        families_rules(data)
    elif name == "artifact_inventory":
        inventory_rules(data)
    elif name == "artifact_pin":
        if not data["artifact_id"].endswith("-" + data["manifest_sha256"][:16]):
            raise ContractError("pin artifact_id must end with the manifest sha256 prefix")
    elif name == "work_unit":
        if (data["deadline_mono_ns"] is None) != (data["deadline_clock_domain"] is None):
            raise ContractError("local deadline and clock domain must appear together")
        if data["work_type"] in ("prefill","decode","verify","draft","infill","state_restore") and data["session_id"] is None:
            raise ContractError("stateful inference work requires session identity")
        if data["work_type"] in ("decode","verify","draft","infill") and data["max_new_tokens"] == 0:
            raise ContractError("generation work needs an output budget")
    elif name == "control_command":
        if (data["model_generation"] is None) != (data["method"] in CONTROL_METHODS_WITHOUT_MODEL):
            raise ContractError("model_generation required exactly for model-targeting methods")
    elif name == "runtime_config":
        _runtime_config_rules(data, root)
    elif name == "worker_agent_config":
        _worker_agent_config_rules(data, root)
    elif name == "worker_inventory":
        worker_inventory_rules(data)
    elif name == "model_instance":
        reserved = data["reserved_bytes"]
        if data["cleanup_pending"] and (data["lifecycle_state"] != "failed" or reserved == 0):
            raise ContractError("cleanup must retain failed-load reservation")
        if reserved and (not data["resource_ids"] or data["generation"] == 0 or data["worker_epoch"] == 0):
            raise ContractError("reservation requires placement and a live generation")
        if reserved < data["resident_bytes"]:
            raise ContractError("reservation below resident bytes")
        if data["lifecycle_state"] in ("loading", "ready", "draining", "unloading") and reserved == 0:
            raise ContractError("live allocation must retain reservation")
        if data["lifecycle_state"] == "unloaded" and (reserved or data["cleanup_pending"]):
            raise ContractError("unloaded instance still owns reservation")
        if data["lifecycle_state"] == "failed" and not data["cleanup_pending"] and reserved:
            raise ContractError("acknowledged cleanup must release the reservation")
        state = data["lifecycle_state"]
        if state == "unloaded" and (data["resource_ids"] or data["resident_bytes"]):
            raise ContractError("unloaded model cannot retain resources")
        if state in ("ready","draining","unloading") and (not data["resource_ids"] or data["resident_bytes"] == 0):
            raise ContractError("resident lifecycle state requires resources and bytes")
        if state not in ("ready", "draining") and (data["lease_count"] or data["active_request_count"]):
            raise ContractError("only ready or draining models may hold leases or requests")
        if state == "failed" and not data["failure_reason"]:
            raise ContractError("failed model requires reason")
        if state != "failed" and data["failure_reason"] is not None:
            raise ContractError("failure reason on nonfailed model")
    elif name == "execution_plan":
        if data["status"] == "planned" and data["completion_ticket"] is not None:
            raise ContractError("planned work cannot already carry completion ticket")
        if data["status"] in ("submitted","completed","cancel_pending") and data["completion_ticket"] is None:
            raise ContractError("submitted work requires a completion ticket")
    elif name == "trace_event":
        if (data["worker_id"] is None) != (data["worker_epoch"] is None):
            raise ContractError("worker id and epoch must be present together")
        if data["event"].startswith("request_") or data["event"] in ("first_token", "last_token"):
            if not data["request_id"] or not data["scope_id"] or data["attempt"] is None:
                raise ContractError("request event requires scoped identity and attempt")
            if data["event"] in ("first_token", "last_token") and data["worker_id"] is None:
                raise ContractError("token execution event requires worker identity")

# Authority-carrying RPCs that address the worker, not a model instance (Authority.model_generation unset).
CONTROL_METHODS_WITHOUT_MODEL = frozenset(("describe_resources", "query_operation", "stream_telemetry"))

DEPLOYMENT_PARTS = (("recipe", "model_recipe"), ("lock", "deployment_lock"),
                    ("capabilities", "execution_capabilities"))

def parse_loopback_base_url(url: str) -> tuple[str, int]:
    """`http://<127.0.0.0/8 literal or [::1]>:<port>`; no userinfo, path, query, fragment or trailing slash."""
    if not isinstance(url, str) or not url.startswith("http://"):
        raise ContractError("backend base_url must use http://")
    host, sep, port = url[len("http://"):].rpartition(":")
    if not sep or not re.fullmatch(r"[1-9][0-9]{0,4}", port, re.ASCII) or not 1 <= int(port) <= 65535:
        raise ContractError("backend base_url needs an explicit port 1..65535 and nothing after it")
    if host == "[::1]":
        return "::1", int(port)
    try:
        ip = ipaddress.IPv4Address(host)
    except ValueError:
        raise ContractError("backend base_url host must be a literal 127.0.0.0/8 address or [::1]") from None
    if not ip.is_loopback or str(ip) != host:
        raise ContractError("backend base_url host must be a literal 127.0.0.0/8 address or [::1]")
    return host, int(port)

TAILNET = ipaddress.IPv4Network("100.64.0.0/10")
RFC1918 = tuple(ipaddress.IPv4Network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
# `[federation]` timings when absent: configs/cluster/dev-tailnet.yaml control_plane (M3.5 Q7).
FEDERATION_DEFAULTS = {"heartbeat_interval_ms": 1000, "suspect_after_ms": 5000, "lost_after_ms": 15000,
                       "admission_lease_ms": 4000, "rpc_timeout_ms": 5000, "rpc_quota_per_principal": 64}
# Agent `[limits]` when absent (M3.5 Q14); output_stall_ms defaults to rpc_timeout_ms.
AGENT_LIMIT_DEFAULTS = {"heartbeat_interval_ms": 1000, "lost_after_ms": 15000, "rpc_timeout_ms": 5000}

def parse_listen(field: str, text: str, *, rfc1918: bool):
    """A canonical `<IPv4>:<port>` or `[<IPv6>]:<port>` literal, port 1..65535, never a wildcard; the host is a
    loopback or 100.64.0.0/10 literal, plus RFC1918 for the agent (a bridge container's own address). Mirrors
    `parse_listen` in crates/cere-runtime/src/config.rs."""
    bad = ContractError(f"{field} must be a canonical <ip>:<port> literal with a port 1..65535")
    host, sep, port = text.rpartition(":")
    if not sep or not re.fullmatch(r"[1-9][0-9]{0,4}", port, re.ASCII) or int(port) > 65535:
        raise bad
    try:
        if host.startswith("[") and host.endswith("]"):
            ip = ipaddress.IPv6Address(host[1:-1])
            canonical = f"[{ip}]" == host
        else:
            ip = ipaddress.IPv4Address(host)
            canonical = str(ip) == host
    except ValueError:
        raise bad from None
    if not canonical:
        raise bad
    if ip.is_unspecified:
        raise ContractError(f"{field} must not be a wildcard address")
    if ip.version == 6:
        allowed = ip == ipaddress.IPv6Address("::1")
    else:
        allowed = ip.is_loopback or ip in TAILNET or (rfc1918 and any(ip in net for net in RFC1918))
    if not allowed:
        kinds = "a loopback, 100.64.0.0/10 or RFC1918" if rfc1918 else "a 100.64.0.0/10 or loopback"
        raise ContractError(f"{field} host must be {kinds} literal")
    return ip, int(port)

def parse_tailnet_endpoint(field: str, url: str):
    """`https://<literal 100.64.0.0/10, 127.0.0.0/8 or [::1]>:<port 1..65535>`, nothing after the port."""
    if not url.startswith("https://"):
        raise ContractError(f"{field} must use https://")
    host, sep, port = url[len("https://"):].rpartition(":")
    if not sep or not re.fullmatch(r"[1-9][0-9]{0,4}", port, re.ASCII) or int(port) > 65535:
        raise ContractError(f"{field} needs an explicit port 1..65535 and nothing after it")
    if host == "[::1]":
        return "::1", int(port)
    try:
        ip = ipaddress.IPv4Address(host)
    except ValueError:
        ip = None
    if ip is None or str(ip) != host or not (ip.is_loopback or ip in TAILNET):
        raise ContractError(f"{field} host must be a literal 100.64.0.0/10, 127.0.0.0/8 or [::1] address")
    return host, int(port)

def _under(path: str, base) -> bool:
    p, b = PurePosixPath(path), PurePosixPath(base)
    return p == b or b in p.parents

def _outside_checkout(field: str, path: str, checkout: Path) -> None:
    if any(_under(path, repo) for repo in (checkout, checkout.resolve())):
        raise ContractError(f"{field} must be outside any repository checkout")

def _pem_paths(prefix: str, data: dict, checkout: Path, state_dir: str) -> None:
    """PEM paths by shape only (the schema makes them absolute; nothing is opened): outside the checkout, not under
    the state dir (M3.5 Q16)."""
    for key in ("ca_file", "cert_file", "key_file"):
        _outside_checkout(prefix + key, data[key], checkout)
        if _under(data[key], state_dir):
            raise ContractError(f"{prefix}{key} must not be under the state_dir")

# config.rs REMOTE_BODY_MAX: with [federation] a request body must fit one SubmitWork under wire::INFERENCE_MESSAGE_MAX
# (8 MiB) with 64 KiB kept for the envelope (M3.5 Q14).
REMOTE_BODY_MAX = (8 << 20) - 65536

def _federation_rules(data: dict, root: Path) -> None:
    """`[federation]` (M3.5 Q16, Q7, Q14); the global cap with remote workers is checked with the sglang_http cap."""
    fed = data["federation"]
    if any(e["kind"] != "sglang_http" for e in data["deployments"]):
        raise ContractError("[federation] requires backend kind sglang_http")
    if data["api"]["max_request_body_bytes"] > REMOTE_BODY_MAX:
        raise ContractError(f"api.max_request_body_bytes must be <= {REMOTE_BODY_MAX} with [federation] (one SubmitWork "
                            "under the remote message cap)")
    parse_listen("federation.listen", fed["listen"], rfc1918=False)
    _pem_paths("federation.", fed, root, data["ledger"]["state_dir"])
    t = {**FEDERATION_DEFAULTS, **fed}
    heartbeat, suspect, lost, lease = (t["heartbeat_interval_ms"], t["suspect_after_ms"], t["lost_after_ms"],
                                       t["admission_lease_ms"])
    if not heartbeat < lease < lost:
        raise ContractError("federation timings must satisfy heartbeat_interval_ms < admission_lease_ms < lost_after_ms")
    if suspect <= heartbeat:
        raise ContractError("federation suspect_after_ms must exceed heartbeat_interval_ms")
    if lost <= suspect:
        raise ContractError("federation lost_after_ms must exceed suspect_after_ms")
    if lease >= suspect:
        raise ContractError("federation admission_lease_ms must be below suspect_after_ms")
    ids, principals = set(), set()
    for w in fed["workers"]:
        parse_tailnet_endpoint("federation.workers.endpoint", w["endpoint"])
        if w["worker_id"] in ids:
            raise ContractError(f"duplicate federation worker_id {w['worker_id']}")
        ids.add(w["worker_id"])
        if w["worker_id"] in data["scheduler"]["devices"]:
            raise ContractError(f"federation worker_id equals a scheduler device: {w['worker_id']}")
        if w["principal"] == fed["principal"]:
            raise ContractError("federation worker principal equals the coordinator principal")
        if w["principal"] in principals:
            raise ContractError(f"duplicate federation worker principal {w['principal']}")
        principals.add(w["principal"])
    _team_pool_rules(data, fed)

# net-teleport design note 4.4 (S6): `owner_only` windows, mirroring `crate::pool::Window::parse`.
_WINDOW_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_ZONE_PART = re.compile(r"[A-Za-z0-9_+-]+")
REMOTE_SMOKE_SCOPE = "cere-smoke"

def parse_owner_only_window(text: str) -> None:
    """`"<days> <HH:MM>-<HH:MM> <zone>"`: days Mon..Sun, ranges or comma lists; start before end (24:00 ends at
    midnight); zone `UTC` or an IANA name whose file under `$TZDIR` (else /usr/share/zoneinfo) is TZif."""
    def bad(why: str) -> ContractError:
        return ContractError(f"owner_only window {text!r}: {why}")
    parts = text.split(" ")
    if len(parts) != 3:
        raise bad('expected "<days> <HH:MM>-<HH:MM> <zone>"')
    days, times, zone = parts
    for item in days.split(","):
        a, sep, b = item.partition("-")
        if a not in _WINDOW_DAYS or (sep and b not in _WINDOW_DAYS):
            raise bad("days are Mon..Sun, ranges a-b or comma lists")
    start, sep, end = times.partition("-")
    if not sep:
        raise bad("times are HH:MM-HH:MM")
    def minute(t: str):
        h, sep, m = t.partition(":")
        if not sep or not re.fullmatch(r"[0-9]{2}", h) or not re.fullmatch(r"[0-9]{2}", m):
            return None
        h, m = int(h), int(m)
        return h * 60 + m if m < 60 and (h < 24 or (h == 24 and m == 0)) else None
    s, e = minute(start), minute(end)
    if s is None or e is None:
        raise bad("times are HH:MM (00:00..24:00)")
    if s >= e:
        raise bad("the start must be before the end (an overnight window is two entries)")
    if zone == "UTC":
        return
    if not (0 < len(zone) <= 64 and all(p and not p.startswith(".") and _ZONE_PART.fullmatch(p)
                                         for p in zone.split("/"))):
        raise bad(f"zone {zone!r} is not an IANA zone name")
    path = Path(os.environ.get("TZDIR") or "/usr/share/zoneinfo") / zone
    try:
        head = path.read_bytes()[:4]
    except OSError as exc:
        raise bad(f"zone {zone!r}: cannot read {path}: {exc}") from None
    if head != b"TZif":
        raise bad(f"zone {zone!r}: {path} is not TZif")

def _team_pool_rules(data: dict, fed: dict) -> None:
    scopes = {s["scope_id"] for s in data["auth"]["scopes"]}
    for w in fed["workers"]:
        owner = w.get("owner")
        if owner is not None and owner not in scopes:
            raise ContractError(f"federation worker {w['worker_id']} owner must name a configured auth scope")
        if "owner_only" in w:
            if owner is None:
                raise ContractError(f"federation worker {w['worker_id']} owner_only needs an owner")
            for text in w["owner_only"]:
                parse_owner_only_window(text)
    if any("owner" in w for w in fed["workers"]) and REMOTE_SMOKE_SCOPE in scopes:
        raise ContractError(f"auth scope {REMOTE_SMOKE_SCOPE} is reserved for the remote readiness smoke when a "
                            "federation worker has an owner")

def _worker_agent_config_rules(data: dict, root: Path) -> None:
    """`cere-worker` config (M3.5 Q16, Q12, Q14; harness H1). `root` is the checkout bounding state_dir and the PEMs;
    the Rust loader applies that bound only when an ancestor of the config holds PACKAGE_VERSION."""
    parse_listen("listen", data["listen"], rfc1918=True)
    parse_tailnet_endpoint("coordinator.endpoint", data["coordinator"]["endpoint"])
    if data["principal"] == data["coordinator"]["principal"]:
        raise ContractError("principal equals coordinator.principal")
    parse_loopback_base_url(data["backend"]["base_url"])
    limits = {**AGENT_LIMIT_DEFAULTS, **data.get("limits", {})}
    launch = data["backend"].get("launch")
    if launch is not None:
        # m36 design note Q3: the schema bounds characters, the Rust loader bounds UTF-8 bytes (execve strings) in
        # every argv entry, argv[0] included.
        if any(not a or "\x00" in a or len(a.encode()) > 4096 for a in launch["argv"]):
            raise ContractError("backend.launch.argv entries must be 1..4096 bytes without NUL")
        if any(len(v.encode()) > 4096 for v in launch.get("env", {}).values()):
            raise ContractError("backend.launch.env values must be at most 4096 bytes without NUL")
    if limits["heartbeat_interval_ms"] >= limits["lost_after_ms"]:
        raise ContractError("limits.heartbeat_interval_ms must be below limits.lost_after_ms")
    if limits.get("output_stall_ms", limits["rpc_timeout_ms"]) > limits["lost_after_ms"]:
        raise ContractError("limits.output_stall_ms must not exceed limits.lost_after_ms")
    _outside_checkout("state_dir", data["state_dir"], root)
    owner = data.get("owner")
    if owner == REMOTE_SMOKE_SCOPE:
        raise ContractError("owner must not be the remote readiness smoke scope")
    if "owner_only" in data:
        if owner is None:
            raise ContractError("owner_only needs an owner")
        for text in data["owner_only"]:
            parse_owner_only_window(text)
    _pem_paths("", data, root, data["state_dir"])

def validate_deployment(root: Path, prefix: str) -> dict:
    """Load and cross-check `<prefix>.{recipe,lock,capabilities}.json` (repo-relative prefix)."""
    docs, raw = {}, {}
    for part, schema in DEPLOYMENT_PARTS:
        path = root / f"{prefix}.{part}.json"
        if not path.is_file():
            raise ContractError(f"deployment file missing: {prefix}.{part}.json")
        raw[part] = path.read_bytes()
        docs[part] = parse_json(raw[part])
        validate(schema, docs[part], root)
    recipe, lock, profile = docs["recipe"], docs["lock"], docs["capabilities"]
    if lock["recipe_sha256"] != hashlib.sha256(raw["recipe"]).hexdigest():
        raise ContractError("deployment lock does not pin the recipe bytes")
    if not recipe["backend_profile_id"] == lock["backend_profile_id"] == profile["profile_id"]:
        raise ContractError("deployment recipe, lock and capability profile ids differ")
    if profile["artifact_sha256"] != lock["artifact_sha256"]:
        raise ContractError("capability profile artifact differs from the lock")
    if not recipe["context_tokens"] == lock["effective_context_tokens"] == profile["max_context_tokens"]:
        raise ContractError("deployment context lengths differ")
    if not set(recipe["required_operations"]) <= set(lock["effective_operations"]):
        raise ContractError("lock does not provide every operation the recipe requires")
    return {"recipe": recipe, "lock": lock, "profile": profile,
            "lock_sha256": hashlib.sha256(raw["lock"]).hexdigest()}

def _runtime_config_rules(data: dict, root: Path) -> None:
    port = int(data["server"]["bind"].split(":")[-1])
    if not 1 <= port <= 65535: raise ContractError("invalid server port")
    mm = data["model_management"]
    if mm["unload_idle"] and mm["idle_unload_seconds"] < mm["min_resident_seconds"]:
        raise ContractError("idle unload must not undercut minimum residency")
    if len(mm["preload_recipe_ids"]) > mm["max_loaded_models_per_device"] * len(data["scheduler"]["devices"]):
        raise ContractError("preload set exceeds configured model slots")
    if data["state"]["allow_cross_device_resume"]:
        raise ContractError("cross-device resume not implemented in this starter")
    # NetTeleport S3 (net-teleport design note 4.2 L1): parking runs on the single native form only.
    if data["state"].get("park_on_stop") and not (
            data["deployments"][0].get("kind") == "native_cuda"
            and "replicas" not in (data["deployments"][0].get("native") or {})):
        raise ContractError("state.park_on_stop needs deployments[0] native_cuda in the single form (no replicas)")
    api, admission = data["api"], data["admission"]
    if api["default_max_output_tokens"] > api["max_output_tokens"]:
        raise ContractError("api default_max_output_tokens exceeds max_output_tokens")
    if api["default_deadline_ms"] > api["max_deadline_ms"]:
        raise ContractError("api default_deadline_ms exceeds max_deadline_ms")
    if (api.get("resume_buffer_total_bytes", 67108864)
            < api.get("resume_buffer_bytes", 4194304) * admission["max_backend_inflight"]):
        raise ContractError("api resume_buffer_total_bytes is below resume_buffer_bytes x admission.max_backend_inflight")
    if admission["per_scope_inflight"] > admission["max_backend_inflight"]:
        raise ContractError("per-scope in-flight cap exceeds the global backend cap")
    scopes = data["auth"]["scopes"]
    if len({s["scope_id"] for s in scopes}) != len(scopes):
        raise ContractError("duplicate auth scope_id")
    envs = [e for s in scopes for e in [s["token_env"], *s.get("token_envs", [])]]
    if len(set(envs)) != len(envs):
        raise ContractError("one token_env cannot authenticate two scopes")
    for s in scopes:
        if s.get("per_scope_inflight", 0) > admission["max_backend_inflight"]:
            raise ContractError("per-scope in-flight cap exceeds the global backend cap")
    operator_env = data["auth"].get("operator_token_env")
    if operator_env is not None and operator_env in envs:
        raise ContractError("auth.operator_token_env must differ from every scope token_env")
    keyless = data["auth"].get("keyless_scope")
    if keyless is not None and keyless not in {s["scope_id"] for s in scopes}:
        raise ContractError("auth.keyless_scope must name a configured auth scope")
    state_dir = PurePosixPath(data["ledger"]["state_dir"])
    for repo in {PurePosixPath(root), PurePosixPath(root.resolve())}:
        if state_dir == repo or repo in state_dir.parents:
            raise ContractError("ledger state_dir must be outside the repository")
    if "federation" in data:
        _federation_rules(data, root)
    entries = data["deployments"]
    if len(entries) > 1 and any(e["kind"] != "sglang_http" for e in entries):
        raise ContractError("more than one deployment requires every entry to be sglang_http (native_cuda, "
                            "unconfigured and cpu_reference allow exactly one)")
    kind = entries[0]["kind"]
    if kind not in ("sglang_http", "native_cuda"):
        return
    devices = data["scheduler"]["devices"]
    if kind == "sglang_http":
        _sglang_http_rules(data, entries, devices, admission, scopes)
    else:
        native = entries[0].get("native", {})
        # M2.11 D8 (F6): preemption parks snapshots in the host-RAM arena, which is host spill.
        if native.get("preempt", False):
            if native.get("snapshot_host_bytes", 0) == 0:
                raise ContractError("deployments[0].native.preempt requires snapshot_host_bytes > 0")
            if not data["state"]["host_spill_enabled"]:
                raise ContractError("deployments[0].native.preempt requires [state] host_spill_enabled")
        if native.get("prefix_state_slots", 0) > 0:
            _cross_scope_rule(entries[0], scopes)
        _native_workers(entries[0], devices)
    for i, entry in enumerate(entries):
        _check_deployment(data, i, entry, validate_deployment(root, entry["deployment"]))

def _sglang_http_rules(data: dict, entries: list, devices: list, admission: dict, scopes: list) -> None:
    """Every entry is `sglang_http` (M5 D2): one entry keeps the M3 rules (exactly one of `base_url` and `replicas`);
    several entries refuse `base_url` and make `replicas` optional. Then the cross-entry rules: unique names and
    locks (c), disjoint replica devices covering `scheduler.devices` (d), each remote worker bound to the entry
    serving its `expect.served_model_name` (e), a worker per entry (f), equal cache keys (g), the summed global cap
    (j) and the cross-scope rule per entry (i). Mirrors `validate_sglang_http` in the Rust loader."""
    multi = len(entries) > 1
    for i, e in enumerate(entries):
        p = f"deployments[{i}]"
        if multi:
            if "base_url" in e:
                raise ContractError(f"{p}.base_url is refused with more than one deployment; use {p}.replicas")
        elif ("base_url" in e) == ("replicas" in e):
            raise ContractError(f"sglang_http requires exactly one of {p}.base_url and {p}.replicas")
        if multi and "load_endpoint" in e:
            raise ContractError(f"{p}.load_endpoint is refused with more than one deployment; set it per replica")
    names, locks = set(), set()
    for e in entries:
        if e["served_model_name"] in names:
            raise ContractError(f"duplicate deployments served_model_name {e['served_model_name']}")
        names.add(e["served_model_name"])
        if e["deployment"] in locks:
            raise ContractError(f"duplicate deployments deployment {e['deployment']}")
        locks.add(e["deployment"])
    if "base_url" in entries[0]:
        parse_loopback_base_url(entries[0]["base_url"])
        local = [_single_worker(devices)]
    else:
        local = _replica_workers(entries, devices)
    remote = [[] for _ in entries]
    for w in data.get("federation", {}).get("workers", []):
        i = next((i for i, e in enumerate(entries) if e["served_model_name"] == w["expect"]["served_model_name"]), None)
        if i is None:
            raise ContractError(f"federation worker {w['worker_id']} expect.served_model_name "
                                f"{w['expect']['served_model_name']} matches no deployments served_model_name")
        remote[i].append(w)
    for i, (n, r) in enumerate(zip(local, remote)):
        if n == 0 and not r:
            raise ContractError(f"deployments[{i}] has no worker (no local replica and no remote worker)")
    keys = [(e["shared_backend_tenancy"], e["allow_cross_scope_prefix_cache"], e.get("prefix_cache_isolation", "none"))
            for e in entries]
    for i, k in enumerate(keys[1:], 1):
        if k != keys[0]:
            raise ContractError(f"deployments[{i}] shared_backend_tenancy, allow_cross_scope_prefix_cache and "
                                "prefix_cache_isolation must equal deployments[0]'s")
    # Global cap <= sum over entries of N_e x min(per-worker cap of e); every replica's cap is its entry's
    # expect.max_running_requests (M3 Q6). With [federation], N_e also counts e's remote workers and each of their
    # max_inflight bounds e's per-worker cap (M3.5 Q16, M5 D2(j)).
    cap = sum((n + len(r)) * min([e["expect"]["max_running_requests"]] + [w["max_inflight"] for w in r])
              for e, n, r in zip(entries, local, remote))
    if admission["max_backend_inflight"] > cap:
        formula = ("local + remote workers x min(expect.max_running_requests, every max_inflight)"
                   if "federation" in data else "replicas x expect.max_running_requests")
        summed = ", summed over deployments" if multi else ""
        raise ContractError(f"admission max_backend_inflight exceeds {formula}{summed}")
    for e in entries:
        _cross_scope_rule(e, scopes)

def _cross_scope_rule(entry: dict, scopes: list) -> None:
    """The cross-scope rule of a backend radix cache (Q2; tasks/m210-spec.md D2 for the native cache): with more than
    one auth scope, scopes share prefixes only by allow_cross_scope_prefix_cache, else every scope needs its own
    namespace (prefix_cache_isolation = cache_salt). Every sglang_http entry; native_cuda with prefix_state_slots > 0."""
    if (len(scopes) > 1 and not entry["allow_cross_scope_prefix_cache"]
            and entry.get("prefix_cache_isolation", "none") != "cache_salt"):
        raise ContractError("more than one auth scope on a shared radix cache requires "
                            "allow_cross_scope_prefix_cache or prefix_cache_isolation = cache_salt")

def max_snapshot_bytes(max_context_tokens: int, draft: bool = False) -> int:
    """Bytes of the largest CERESNP1 snapshot (tasks/m211-spec.md D2: kinds 0-8, 128 B header, 24 B per table entry,
    256-B aligned sections; kind 8 the sampler's out_counts, ledger CR13; with `draft` kind 10, the MTP layer's K/V
    rows, QSA keys and ring and the 16-B pending proposals, M2.9) of a Qwen3.8-Flash-Next session at `position` =
    `token_count` = `max_context_tokens`. Mirrors `native::snapshot::max_snapshot_bytes` (Rust) and the executor's
    plan (snapshot_format.cpp)."""
    p = max_context_tokens
    sections = (4 * p, 12 * p * 512, 12 * p * 512, 12 * (p // 4) * 256, 12 * (p % 4) * 256,
                36 * 10_240 * 3 * 2, 36 * 48 * 128 * 128 * 2, 2 * 8 + 10_240 * 9 * 2, 248_320 * 4)
    if draft:
        sections += (2 * p * 512 + (p // 4) * 256 + (p % 4) * 256 + 16,)
    end = 128 + 24 * len(sections)
    for size in sections:
        end = -(-end // 256) * 256 + size
    return end

def arena_slot_bytes(max_context_tokens: int, draft: bool = False) -> int:
    """Arena bytes one parked snapshot at `max_context_tokens` may take: `max_snapshot_bytes` rounded up to the
    arena's 256-B sub-allocation alignment (D2). Mirrors `native::snapshot::arena_slot_bytes` (Rust)."""
    return -(-max_snapshot_bytes(max_context_tokens, draft) // 256) * 256

def _check_deployment(data: dict, i: int, entry: dict, d: dict) -> None:
    """Cross-checks between entry `i` and its deployment files (M5 D2(i))."""
    recipe, lock, profile = d["recipe"], d["lock"], d["profile"]
    p = f"deployments[{i}]"
    if profile["profile_id"] != entry["profile_id"]:
        raise ContractError("backend profile_id differs from the deployment profile")
    if profile["execution_mode"] != entry["mode"]:
        raise ContractError("deployment execution mode differs from backend mode")
    if entry["kind"] == "sglang_http":
        expect = entry["expect"]
        if recipe["model_ref"] != "local://" + expect["model_path"].lstrip("/"):
            raise ContractError("deployment recipe does not pin expect.model_path")
        if "sha256:" + lock["backend_build_sha256"] != expect["image_digest"]:
            raise ContractError("deployment lock does not pin expect.image_digest")
        if profile["max_context_tokens"] != expect["context_length"]:
            raise ContractError("deployment context differs from expect.context_length")
        if expect["max_running_requests"] > profile["decode_batch_max"]:
            raise ContractError("expect.max_running_requests exceeds the profile's validated rows")
    else:
        native = entry["native"]
        # m25 D4 (ledger D 72): absent or 0 = the derived pool; the executor bounds a non-zero pool above.
        pool = native.get("kv_pool_tokens", 0)
        if pool and pool % 64:
            raise ContractError(f"{p}.native.kv_pool_tokens must be 0 or a multiple of 64")
        if pool and pool < native["max_context_tokens"]:
            raise ContractError(f"{p}.native.kv_pool_tokens must be 0 or at least {p}.native.max_context_tokens")
        if lock["backend_build_sha256"] != native["library_sha256"]:
            raise ContractError(f"deployment lock backend_build_sha256 differs from {p}.native.library_sha256")
        if lock["artifact_sha256"] != native["expected_manifest_sha256"]:
            raise ContractError(f"deployment lock artifact_sha256 differs from {p}.native.expected_manifest_sha256")
        if profile["max_context_tokens"] != native["max_context_tokens"]:
            raise ContractError(f"deployment context differs from {p}.native.max_context_tokens")
        arena = native.get("snapshot_host_bytes", 0)
        preempt = native.get("preempt", False)
        # M2.11: the executor reports native_state_snapshot exactly when it has an arena (ledger C 56). The arena is
        # the one cered passes (M2.10 D9): 0 with the prefix cache on and [state] host_spill_enabled = false.
        desc_arena = 0 if native.get("prefix_state_slots", 0) > 0 and not data["state"]["host_spill_enabled"] else arena
        if (desc_arena > 0) != profile["native_state_snapshot"]:
            raise ContractError(f"{p}.native.snapshot_host_bytes > 0 must equal the profile's native_state_snapshot")
        # inkling design note D-I-S9B (Rust: config.rs INKLING_DRAFT_WINDOWS): the verify window is 0 or 2..8 on
        # family inkling2_loop (1..7 drafts of the MTP drafter's block), 0 or 4 on every other family.
        drafts = native.get("speculative_draft_tokens", 0)
        family = native.get("family", "qwen4_exp")
        if family == "inkling2_loop":
            if drafts != 0 and not 2 <= drafts <= 8:
                raise ContractError(f"{p}.native.speculative_draft_tokens must be 0 or 2..8 with family {family}")
        elif drafts not in (0, 4):
            raise ContractError(f"{p}.native.speculative_draft_tokens must be 0 or 4 with family {family}")
        # m29 design note D11/D12 (I3): the executor reports native_speculative_batch exactly when it has a draft.
        if (drafts > 0) != profile["native_speculative_batch"]:
            raise ContractError(f"{p}.native.speculative_draft_tokens > 0 must equal the profile's "
                                "native_speculative_batch")
        # m23 design note PI-02 (registry levers); m29l design note D6: suffix-lookup drafts (levers.drafter, the
        # former spec_drafts) ride the MTP verify chain.
        check_levers(p, native, profile["decode_batch_max"])
        # The hot swap (cere_model_set_selection) does not flush the prefix cache: a NO_SPECULATION request after a swap
        # of a numerics-changing lever could hit states computed under the previous levers.
        if native.get("lever_hot_swap") and native.get("prefix_state_slots", 0) > 0:
            raise ContractError(f"{p}.native.lever_hot_swap with prefix_state_slots > 0: the hot swap leaves prefix "
                                "states computed under the previous levers; serve the hot swap cache-off")
        # ab-split design note AB-02: the A/B arms' levers, runtime-layer and bit_identical only.
        check_ab(p, native)
        check_loop_guard(p, native)
        # D8 composition (I2) was refused with decode_batch_max > 1 until rows could leave a batch for a park
        # (sessions design note SE-09).
        draft = native.get("speculative_draft_tokens", 0) > 0
        if native.get("max_parked_sessions", 0) * arena_slot_bytes(native["max_context_tokens"], draft) > arena:
            raise ContractError(f"{p}.native.max_parked_sessions x the snapshot bytes at max_context_tokens exceed "
                                "snapshot_host_bytes")
        # D13 (ledger P1 suffix rule, LR5): the dev knob runs only on the dev/live snapshot profile.
        if native.get("preempt_every", 0) > 0 and not (
                profile["native_state_snapshot"] and profile["profile_id"].endswith("-snapshot")):
            raise ContractError(f"{p}.native.preempt_every requires the dev/live <profile id>-snapshot profile")
        # D13: the knob forces U1 preemptions, so it needs U1.
        if native.get("preempt_every", 0) > 0 and not preempt:
            raise ContractError(f"{p}.native.preempt_every requires preempt = true")
        # M211R-04: a forced preemption parks the victim, so it needs room for one parked session.
        if native.get("preempt_every", 0) > 0 and native.get("max_parked_sessions", 0) == 0:
            raise ContractError(f"{p}.native.preempt_every requires max_parked_sessions >= 1")
        # SE-09: the reserve is whole 64-token pages (M2.5 D4) and the grow is a U1 park.
        reserve = native.get("reserve_output_tokens", 0)
        if reserve % 64:
            raise ContractError(f"{p}.native.reserve_output_tokens must be a multiple of 64")
        if reserve and not preempt:
            raise ContractError(f"{p}.native.reserve_output_tokens requires preempt = true")
        # One native worker per replica (single form: one) whose cap is the profile's validated rows (M3 Q6, Q13),
        # plus the sessions U1 parks in the snapshot arena (M2.11 D8); m32n design note D16: replicas x that.
        parked = native.get("max_parked_sessions", 0) if preempt else 0
        workers = len(native.get("replicas", [])) or 1
        if data["admission"]["max_backend_inflight"] > workers * (profile["decode_batch_max"] + parked):
            raise ContractError("admission max_backend_inflight exceeds the profile's validated rows "
                                "(+ max_parked_sessions with preempt)" if workers == 1 else
                                "admission max_backend_inflight exceeds native replicas x the profile's validated "
                                "rows (+ max_parked_sessions with preempt)")
        # SE-09: every generation in flight must be able to park at once to grow.
        if reserve and native.get("max_parked_sessions", 0) < data["admission"]["max_backend_inflight"]:
            raise ContractError(f"{p}.native.reserve_output_tokens requires max_parked_sessions >= "
                                "admission.max_backend_inflight")
    if "chat" not in lock["effective_operations"]:
        raise ContractError("deployment lock does not provide chat")

def _single_worker(devices: list) -> int:
    """The single backend form is one worker on scheduler.devices[0] (M3 Q3)."""
    if len(devices) != 1:
        raise ContractError("single backend form needs exactly one scheduler device")
    return 1

NATIVE_REPLICAS = 2
GPU_UUID = re.compile(r"GPU-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}")

def _native_workers(entry: dict, devices: list) -> int:
    """`native_cuda` workers (tasks/m32n-spec.md D2): exactly one of `native.device_uuid` (the single form, one worker
    on scheduler.devices[0]) and `[[deployments.native.replicas]]` (exactly two, unique devices and GPU uuids, together
    exactly `scheduler.devices`). Mirrors `validate_native` and `native_workers` in the Rust loader."""
    native, p = entry.get("native", {}), "deployments[0]"
    replicas = native.get("replicas")
    if replicas is not None and "device_uuid" in native:
        raise ContractError(f"{p}.native.device_uuid is refused with {p}.native.replicas; set it per replica")
    if replicas is None:
        if "device_uuid" not in native:
            raise ContractError(f"{p}.native requires exactly one of device_uuid and replicas")
        return _single_worker(devices)
    if len(replicas) != NATIVE_REPLICAS:
        raise ContractError(f"{p}.native.replicas must list exactly {NATIVE_REPLICAS} replicas")
    if native.get("lever_hot_swap", False):
        raise ContractError(f"{p}.native.lever_hot_swap is refused with {p}.native.replicas")
    seen, uuids = set(), set()
    for r in replicas:
        if not GPU_UUID.fullmatch(r["device_uuid"]):
            raise ContractError(f"{p}.native.replicas.device_uuid must be GPU-<8-4-4-4-12 lowercase hex>")
        if r["device"] in seen:
            raise ContractError(f"duplicate native replica device {r['device']}")
        if r["device_uuid"] in uuids:
            raise ContractError(f"duplicate native replica device_uuid {r['device_uuid']}")
        seen.add(r["device"])
        uuids.add(r["device_uuid"])
    for r in replicas:
        if r["device"] not in devices:
            raise ContractError(f"native replica device is not in scheduler.devices: {r['device']}")
    if len(devices) != len(replicas):
        raise ContractError("scheduler.devices must list exactly the native replica devices")
    return len(replicas)

def _replica_workers(entries: list, devices: list) -> list:
    """`[[deployments.replicas]]` (M3 Q2, M5 D2(b)(d)): devices are unique within and across entries, at least one
    entry has a replica, and together they are exactly `scheduler.devices`; base URLs are unique.
    parse_loopback_base_url admits only a literal host and an explicit port without a trailing slash, so its
    (host, port) is the normalised URL. Returns each entry's local worker count."""
    owners, seen_urls, counts = {}, set(), []
    for i, e in enumerate(entries):
        replicas = e.get("replicas", [])
        if replicas and "load_endpoint" in e:
            raise ContractError(f"deployments[{i}].load_endpoint is refused with deployments[{i}].replicas; "
                                "set it per replica")
        for r in replicas:
            if r["device"] in owners:
                j = owners[r["device"]]
                raise ContractError(f"duplicate replica device {r['device']}" if j == i else
                                    f"replica device {r['device']} belongs to deployments[{j}] and deployments[{i}]")
            owners[r["device"]] = i
            url = parse_loopback_base_url(r["base_url"])
            if url in seen_urls:
                raise ContractError(f"duplicate replica base_url {r['base_url']}")
            seen_urls.add(url)
            if r["device"] not in devices:
                raise ContractError(f"replica device is not in scheduler.devices: {r['device']}")
        counts.append(len(replicas))
    if not owners:
        raise ContractError("at least one deployment needs a local replica")
    if len(devices) != len(owners):
        raise ContractError("scheduler.devices must list exactly the replica devices")
    return counts

def require_physical_capacity(graph: dict, device_id: str) -> int:
    validate("resource_graph", graph)
    if graph["evidence"] != "physical_probe":
        raise ContractError("physical admission requires a real probe, not a template/simulation")
    node = next((n for n in graph["nodes"] if n["id"] == device_id), None)
    if not node or node["kind"] != "gpu" or not node["capacity_bytes"]:
        raise ContractError("unknown device capacity")
    return node["capacity_bytes"]

def require_capability(profile: dict, *, rows: int, context: int, operation: str,
                       modality: str = "text", output: str = "text",
                       numerical_mode: str = "reference", sampler_mode: str = "greedy",
                       allow_synthetic: bool = False) -> str:
    validate("execution_capabilities", profile)
    allowed = ("measured", "synthetic") if allow_synthetic else ("measured",)
    if profile["evidence"] not in allowed:
        raise ContractError("unverified capability profile")
    if type(rows) is not int or not profile["decode_batch_min"] <= rows <= profile["decode_batch_max"]:
        raise ContractError("unsupported row count")
    if type(context) is not int or not 0 < context <= profile["max_context_tokens"]:
        raise ContractError("unsupported context")
    if (operation not in profile["operations"] or modality not in profile["modalities"] or
        output not in profile["structured_outputs"] or numerical_mode not in profile["numerical_modes"]):
        raise ContractError("unsupported feature combination")
    if rows > 1 and operation in ("decode", "verify") and not profile["native_decode_batch"]:
        raise ContractError("requested native rows would fall back to serial/unknown execution")
    if operation == "verify" and rows > 1 and not profile["native_speculative_batch"]:
        raise ContractError("batched verification unsupported")
    if not any(c["operation"] == operation and c["modality"] == modality and
               c["output"] == output and c["numerical_mode"] == numerical_mode and
               c["sampler_mode"] == sampler_mode and rows in c["row_counts"] and
               context <= c["max_context_tokens"] for c in profile["validated_combinations"]):
        raise ContractError("requested feature combination is not validated")
    return profile["execution_mode"]

def validate_assignment(plan: dict, work_units: list[dict], workers: list[dict], graph: dict,
                        profile: dict) -> None:
    """Cross-document coherence only; not authorization, freshness or memory admission."""
    validate('execution_plan', plan); validate('resource_graph', graph)
    validate('execution_capabilities', profile)
    for worker in workers: validate('worker_node', worker)
    for work in work_units: validate('work_unit', work)
    worker_map = {w['worker_id']: w for w in workers}
    if len(worker_map) != len(workers): raise ContractError('duplicate worker identity')
    nodes = {n['id']: n for n in graph['nodes']}; exclusive = {}
    for worker in workers:
        if worker['resource_graph_name'] != graph['name']: raise ContractError('worker refers to different graph')
        for rid in worker['resource_ids']:
            if rid not in nodes: raise ContractError('worker advertises unknown resource')
            if nodes[rid]['kind'] in ('gpu', 'device_memory'):
                if rid in exclusive: raise ContractError('device has competing lifecycle owners')
                exclusive[rid] = worker['worker_id']
    owner = worker_map.get(plan['worker_id'])
    if owner is None or owner['lifecycle_state'] != 'ready': raise ContractError('plan worker is not ready')
    for k in ('worker_epoch', 'coordinator_term'):
        if owner[k] != plan[k]: raise ContractError('plan authority differs from worker inventory')
    if profile['profile_id'] != plan['capability_profile_id']: raise ContractError('wrong capability profile')
    if profile['profile_id'] not in owner['backend_profile_ids']: raise ContractError('worker lacks profile')
    if plan['mode'] != profile['execution_mode']: raise ContractError('execution mode mismatch')
    if any(exclusive.get(rid) != owner['worker_id'] or nodes[rid]['kind'] != 'gpu' for rid in plan['device_ids']):
        raise ContractError('plan uses unowned/non-GPU device')
    ids=[w['id'] for w in work_units]
    if len(set(ids)) != len(ids) or set(ids) != set(plan['work_unit_ids']):
        raise ContractError('plan/work inventory mismatch')
    for work in work_units:
        for k in ('worker_id','worker_epoch','coordinator_term','model_generation'):
            if work[k] != plan[k]: raise ContractError('work fence differs from plan')
        if work['artifact_sha256'] != profile['artifact_sha256']: raise ContractError('work/profile artifact mismatch')
        if work['estimated_peak_bytes'] > plan['reserved_bytes']: raise ContractError('work peak exceeds plan reservation')
        if work['input_tokens'] + work['max_new_tokens'] > profile['max_context_tokens']:
            raise ContractError('requested context exceeds profile; no truncation')
    contexts: dict[str, list[int]] = {}
    for work in work_units:
        contexts.setdefault(work['work_type'], []).append(work['input_tokens'] + work['max_new_tokens'])
    for work_type, ctx in contexts.items():
        # Rows of one operation in this plan must form a validated combination of the profile.
        require_capability(profile, rows=len(ctx), context=max(ctx), operation=work_type,
                           allow_synthetic=True)

# ---- M2.1 prepared artifact contract (schema v2, Q4/Q5/Q12/Q17) ----

DTYPE_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "U16": 2,
                  "U32": 4, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}
_ROLE_BY_NAME = {"model.safetensors.index.json": "weight_index", "config.json": "config",
                 "hf_quant_config.json": "quant_config", "generation_config.json": "generation_config",
                 "tokenizer.json": "tokenizer", "tokenizer_config.json": "tokenizer", "vocab.json": "tokenizer",
                 "merges.txt": "tokenizer", "chat_template.jinja": "template",
                 "preprocessor_config.json": "preprocessor", "video_preprocessor_config.json": "preprocessor",
                 "processor_config.json": "preprocessor",
                 "draft/token_map.u32le": "draft_token_map"}
# role -> (min, max) occurrences in a manifest
ROLE_CARDINALITY = {"weights": (1, None), "weight_index": (1, 1), "config": (1, 1), "quant_config": (0, 1),
                    "generation_config": (0, 1), "tokenizer": (1, None), "template": (1, 1),
                    "preprocessor": (0, None), "draft_token_map": (0, 1)}
MODEL_HASH_ROLES = ("weights", "weight_index", "config", "quant_config")
TOKEN_MAP_TRANSFORM = "pickle_int_list_to_u32le:v1"

def canonical_json(value: Any) -> bytes:
    """Q17: UTF-8, ASCII-only, sorted keys, 2-space indent, `,`/`: ` separators, integers only, trailing newline."""
    def no_floats(v):
        if isinstance(v, float):
            raise ContractError("canonical JSON forbids floats")
        if isinstance(v, dict):
            for x in v.values(): no_floats(x)
        elif isinstance(v, list):
            for x in v: no_floats(x)
    no_floats(value)
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, separators=(",", ": ")) + "\n").encode()

def payload_role(path: str) -> str | None:
    """The manifest role is a function of the path (Q4)."""
    if path.endswith(".safetensors") and "/" not in path:
        return "weights"
    return _ROLE_BY_NAME.get(path)

def lock_line_digest(entries) -> str:
    """CONFIGURATION.md lock digest: sha256 over `<sha256>  <path>\\n` lines in bytewise path order."""
    lines = "".join(f"{sha}  {path}\n" for sha, path in sorted(entries, key=lambda e: e[1].encode()))
    return hashlib.sha256(lines.encode()).hexdigest()

def manifest_identity(files: list[dict]) -> dict:
    def digest(roles):
        return lock_line_digest([(f["sha256"], f["path"]) for f in files if f["role"] in roles])
    template = [f["sha256"] for f in files if f["role"] == "template"]
    return {"model_hash": digest(MODEL_HASH_ROLES), "tokenizer_hash": digest(("tokenizer",)),
            "template_hash": template[0] if len(template) == 1 else None}

def family_totals(families: list[dict]) -> dict:
    by_dtype: dict = {}
    by_placement: dict = {}
    for f in families:
        if f["count"] == 0:
            continue
        d = by_dtype.setdefault(f["dtype"], {"count": 0, "bytes": 0, "misaligned": 0})
        d["count"] += f["count"]; d["bytes"] += f["bytes"]; d["misaligned"] += f["misaligned"]
        p = by_placement.setdefault(f["placement"], {"count": 0, "bytes": 0})
        p["count"] += f["count"]; p["bytes"] += f["bytes"]
    return {"by_dtype": by_dtype, "by_placement": by_placement}

def _family_entries_consistent(families: list[dict]) -> None:
    names = [f["family"] for f in families]
    if len(set(names)) != len(names):
        raise ContractError("duplicate family")
    for f in families:
        if f["misaligned"] > f["count"] or (f["count"] == 0) != (f["bytes"] == 0):
            raise ContractError(f"inconsistent family counters: {f['family']}")

def families_rules(data: dict) -> None:
    names = [r["family"] for r in data["rules"]]
    if len(set(names)) != len(names):
        raise ContractError("duplicate family rule")
    for r in data["rules"]:
        try:
            re.compile(r["pattern"], re.ASCII)
        except re.error as e:
            raise ContractError(f"family {r['family']}: invalid pattern: {e}") from None

def worker_inventory_rules(inv: dict) -> None:
    """What the schema cannot say of a Q7 inventory, as the Rust `Inventory::validate` (fed/wire.rs) refuses it:
    identifiers bounded in UTF-8 bytes, and models[0] repeating the top-level identity."""
    for key in ("worker_id", "served_model_name", "kv_cache_dtype", "instance_id"):
        if inv[key] is not None and not 1 <= len(inv[key].encode()) <= 256:
            raise ContractError(f"inventory: {key} must be 1..=256 bytes")
    model = inv["models"][0]
    if any(model[k] != inv[k] for k in ("served_model_name", "instance_id", "model_generation")):
        raise ContractError("inventory: models[0] must repeat served_model_name, instance_id and model_generation")

def inventory_rules(inv: dict) -> None:
    files, fams, t = inv["files"], inv["families"], inv["totals"]
    paths = [f["path"].encode() for f in files]
    if any(a >= b for a, b in zip(paths, paths[1:])):
        raise ContractError("inventory files must be unique and bytewise sorted")
    _family_entries_consistent(fams)
    expect = {"files": len(files), "tensors": sum(f["tensors"] for f in files),
              "file_bytes": sum(f["bytes"] for f in files), "header_bytes": sum(f["header_bytes"] for f in files),
              "data_bytes": sum(f["bytes"] - f["header_bytes"] for f in files),
              "misaligned": sum(f["misaligned"] for f in fams)}
    if t != expect:
        raise ContractError(f"inventory totals differ from its files: {t} != {expect}")
    if (sum(f["count"] for f in fams), sum(f["bytes"] for f in fams)) != (t["tensors"], t["data_bytes"]):
        raise ContractError("inventory totals differ from its per-family table")
    sums = family_totals(fams)
    if inv["by_dtype"] != sums["by_dtype"] or inv["by_placement"] != sums["by_placement"]:
        raise ContractError("inventory dtype/placement tables differ from its per-family table")
    up = [f["upstream"] for f in files]
    lfs = [(u, f) for u, f in zip(up, files) if u is not None and "lfs_sha256" in u]
    expect_up = {"lfs_files": len(lfs), "lfs_size_match": sum(u["size"] == f["bytes"] for u, f in lfs),
                 "non_lfs_files": sum(u is not None and "blob_id" in u for u in up),
                 "not_in_tree": sum(u is None for u in up)}
    if inv["upstream"] != expect_up:
        raise ContractError("inventory upstream counters differ from its files")

def execution_artifact_rules(m: dict, limits: dict) -> None:
    files = m["files"]
    for path in [f["path"] for f in files] + [t["output"] for t in m["transforms"]]:
        try:
            safe_relative(path)
        except ContractError as e:
            raise ArtifactError("unsafe_path", str(e)) from None
        if path in ("manifest.json", "manifest.sha256"):
            raise ArtifactError("unsafe_path", f"reserved manifest path listed as payload: {path}")
    if len(files) > limits["max_files"]:
        raise ArtifactError("too_many_files", f"{len(files)} files > {limits['max_files']}")
    def rule(msg):
        return ArtifactError("manifest_rules", msg)
    paths = [f["path"].encode() for f in files]
    if any(a >= b for a, b in zip(paths, paths[1:])):
        raise rule("files must be unique and bytewise sorted by path")
    counts = dict.fromkeys(ROLE_CARDINALITY, 0)
    for f in files:
        if payload_role(f["path"]) != f["role"]:
            raise rule(f"role {f['role']} does not match path {f['path']}")
        if ("safetensors" in f) != (f["role"] == "weights"):
            raise rule(f"safetensors facts must be present exactly for weights: {f['path']}")
        counts[f["role"]] += 1
    for role, (lo, hi) in ROLE_CARDINALITY.items():
        if counts[role] < lo or (hi is not None and counts[role] > hi):
            raise rule(f"role {role} occurs {counts[role]} times")
    profile, family_rules = m["profile"], m["family_rules"]
    if m["profile_id"] != profile["profile_id"]:
        raise rule("profile_id differs from the embedded profile")
    try:
        families_rules(family_rules)
        _family_entries_consistent(m["families"])
    except ContractError as e:
        raise rule(str(e)) from None
    rules = family_rules["rules"]
    if len(m["families"]) != len(rules):
        raise rule("families must list every family rule in order")
    for r, f in zip(rules, m["families"]):
        if r["role"] not in profile["placement"]:
            raise rule(f"profile has no placement for role {r['role']}")
        want = {k: r[k] for k in ("family", "component", "role", "dtype")}
        want["placement"] = profile["placement"][r["role"]]
        if {k: f[k] for k in want} != want:
            raise rule(f"family {r['family']} differs from its rule or the profile placement")
    sums = family_totals(m["families"])
    want_totals = {"by_dtype": {k: {"count": v["count"], "bytes": v["bytes"]} for k, v in sums["by_dtype"].items()},
                   "by_placement": sums["by_placement"]}
    if m["totals"] != want_totals:
        raise rule("totals differ from the sum of the families")
    ident = manifest_identity(files)
    for k, v in ident.items():
        if m[k] != v:
            raise rule(f"{k} differs from the listed files")
    maps = [f for f in files if f["role"] == "draft_token_map"]
    outputs = [t["output"] for t in m["transforms"]]
    if len(set(outputs)) != len(outputs) or outputs != [f["path"] for f in maps]:
        raise rule("transforms must produce exactly the draft token map")
    if any(f["bytes"] % 4 for f in maps):
        raise rule("draft token map is not a u32 array")
