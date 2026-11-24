"""Deterministic generator of the shared invalid-case table `tests/fixtures/artifact_invalid.json` (M2.1 B2).

Both verifiers iterate the same table: Python (`tests/test_artifacts.py`) and cerec (`crates/cere-compiler/tests`).
Case kinds (every case names exactly one error class of `error_classes`):
* header:    file = u64le(n) || header bytes (`header` text or `header_hex`) || zeros(data_len); n defaults to the
             header length. Parsed with `limits` (defaults `default_limits`).
* token_map: the token-map file bytes `file_hex`, converted with `vocab_size`.
`valid_token_maps` lists positive maps (`file_hex`, `vocab_size`) and the `u32le_hex` both converters must produce.
* artifact:  `ops` applied to a temp copy of `examples/artifact`, then a full verify against the sha256 of the final
             manifest.json (or `expected_manifest_sha256` when given). Ops are documented in tests/test_artifacts.py.
* inventory: a copy of `tests/fixtures/model_src` whose HF tree file is replaced by `tree_name` holding `tree_text`,
             then an inventory with `tests/fixtures/model_src.families.json` (`cerec inventory`, model_inventory.py).
`python3 scripts/fixtures/make_artifact_invalid.py [--check]`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from cere_contracts import DEFAULT_LIMITS, ERROR_CLASSES, canonical_json  # noqa: E402

OUT = HERE.parents[1] / "tests/fixtures/artifact_invalid.json"
F32_2 = '{"dtype":"F32","shape":[2],"data_offsets":[%d,%d]}'


def header(id_, error, text=None, *, hex_=None, n=None, data_len=0, limits=None):
    case = {"id": id_, "tag": "header", "error": error}
    if hex_ is not None:
        case["header_hex"] = hex_
    else:
        case["header"] = text
    if n is not None:
        case["n"] = n
    if data_len:
        case["data_len"] = data_len
    if limits:
        case["limits"] = limits
    return case


def header_cases():
    t = '"t":' + F32_2 % (0, 8)
    return [
        header("header_n_below_2", "header_bounds", hex_="7b"),
        header("header_n_over_limit", "header_bounds", "{" + t + "}", data_len=8, limits={"max_header_bytes": 16}),
        header("header_n_past_eof", "header_bounds", "{}", n=100),
        header("header_n_huge", "header_bounds", "{}", n=1 << 63),
        header("header_not_utf8", "header_utf8", hex_=b'{"t\xff":{}}'.hex()),
        header("header_json_array", "header_json", "[]"),
        header("header_json_truncated", "header_json", '{"t":'),
        header("header_json_trailing", "header_json", "{} x"),
        header("header_duplicate_tensor", "duplicate_key", "{" + t + "," + t + "}", data_len=8),
        header("header_duplicate_field", "duplicate_key",
               '{"t":{"dtype":"F32","dtype":"F32","shape":[2],"data_offsets":[0,8]}}', data_len=8),
        header("header_duplicate_metadata_key", "duplicate_key", '{"__metadata__":{"a":"1","a":"2"}}'),
        header("header_metadata_list", "metadata", '{"__metadata__":["x"]}'),
        header("header_metadata_non_string", "metadata", '{"__metadata__":{"layer":0},' + t + "}", data_len=8),
        header("header_metadata_null", "metadata", '{"__metadata__":null}'),
        header("header_unknown_field", "tensor_fields",
               '{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"align":64}}', data_len=8),
        header("header_missing_offsets", "tensor_fields", '{"t":{"dtype":"F32","shape":[2]}}', data_len=8),
        header("header_offsets_three", "tensor_fields", '{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8,8]}}',
               data_len=8),
        header("header_offsets_negative", "tensor_fields",
               '{"t":{"dtype":"F32","shape":[2],"data_offsets":[-8,0]}}', data_len=8),
        header("header_entry_not_object", "tensor_fields", '{"t":[0,8]}', data_len=8),
        header("header_dtype_unknown", "dtype", '{"t":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}}', data_len=1),
        header("header_dtype_v1_name", "dtype", '{"t":{"dtype":"float32","shape":[2],"data_offsets":[0,8]}}',
               data_len=8),
        header("header_shape_17_dims", "shape",
               '{"t":{"dtype":"U8","shape":[' + ",".join(["1"] * 17) + '],"data_offsets":[0,1]}}', data_len=1),
        header("header_shape_zero_dim", "shape", '{"t":{"dtype":"F32","shape":[2,0],"data_offsets":[0,0]}}'),
        header("header_shape_not_list", "shape", '{"t":{"dtype":"F32","shape":2,"data_offsets":[0,8]}}', data_len=8),
        header("header_shape_bool", "shape", '{"t":{"dtype":"U8","shape":[true],"data_offsets":[0,1]}}', data_len=1),
        header("header_too_many_tensors", "too_many_tensors",
               '{"a":' + F32_2 % (0, 8) + ',"b":' + F32_2 % (8, 16) + "}", data_len=16,
               limits={"max_tensors_per_file": 1}),
        header("header_scalar_shape_size", "byte_size", '{"t":{"dtype":"F32","shape":[],"data_offsets":[0,8]}}',
               data_len=8),
        header("header_size_overflow", "byte_size",
               '{"t":{"dtype":"U64","shape":[4294967296,4294967296],"data_offsets":[0,8]}}', data_len=8),
        header("header_end_before_begin", "offsets", '{"t":' + F32_2 % (8, 0) + "}", data_len=8),
        header("header_offsets_gap", "offsets", '{"a":' + F32_2 % (0, 8) + ',"b":' + F32_2 % (12, 20) + "}",
               data_len=20),
        header("header_offsets_overlap", "offsets", '{"a":' + F32_2 % (0, 8) + ',"b":' + F32_2 % (4, 12) + "}",
               data_len=12),
        header("header_offsets_start_nonzero", "offsets", '{"t":' + F32_2 % (4, 12) + "}", data_len=12),
        header("header_data_short_of_eof", "offsets", '{"t":' + F32_2 % (0, 8) + "}", data_len=12),
        header("header_end_past_data", "offsets", '{"t":' + F32_2 % (0, 8) + "}", data_len=4),
        # serde_json reads `-0` as the float -0.0: not an unsigned integer in either parser.
        header("header_data_offsets_negative_zero", "tensor_fields",
               '{"t":{"dtype":"U8","shape":[1],"data_offsets":[-0,1]}}', data_len=1),
        header("header_shape_negative_zero", "shape", '{"t":{"dtype":"U8","shape":[-0],"data_offsets":[0,0]}}'),
    ]


def zip_bytes(entries, *, method=0, flags=0x808, local_names=None, eocd=None, pad=False, zip64=None, disk_start=0,
              cd_trailing=b""):
    """Stored zip in the torch layout (sizes in the data descriptor and the central directory).

    pad=True adds PyTorch's 'FB' local extra field aligning each entry's data to 64 bytes. zip64 (a dict, possibly
    empty) appends PyTorch's zip64 end record + locator before the classic EOCD; its keys override the written
    values: `size` (record size field), record fields `disk`/`cd_disk`/`n_disk`/`n_total`/`cd_size`/`cd_off`,
    locator fields `loc_disk`/`loc_total`/`loc_off_delta` (added to the record offset), and `gap` (hex bytes
    written between the locator and the EOCD). disk_start is every central entry's disk-number field; cd_trailing
    is written after the central entries and counted in the central directory size."""
    out, central = bytearray(), bytearray()
    for k, (name, data) in enumerate(entries):
        nb = name.encode()
        lnb = (local_names or {}).get(k, name).encode()
        payload = data
        if method == 8:
            c = zlib.compressobj(9, zlib.DEFLATED, -15)
            payload = c.compress(data) + c.flush()
        crc = zlib.crc32(data) & 0xFFFFFFFF
        offset = len(out)
        extra = b""
        if pad:
            n = -(offset + 30 + len(lnb) + 4) % 64
            extra = struct.pack("<HH", 0x4246, n) + b"Z" * n
        out += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, flags, method, 0, 0x21, 0, 0, 0, len(lnb), len(extra))
        out += lnb + extra + payload + struct.pack("<IIII", 0x08074B50, crc, len(payload), len(data))
        central += struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 0x031E, 20, flags, method, 0, 0x21, crc,
                               len(payload), len(data), len(nb), 0, 0, disk_start, 0, 0, offset) + nb
    central += cd_trailing
    cd_off = len(out)
    out += central
    fields = {"disk": 0, "cd_disk": 0, "n_disk": len(entries), "n_total": len(entries), "cd_size": len(central),
              "cd_off": cd_off}
    fields.update(eocd or {})
    if zip64 is not None:
        z = {**fields, **zip64}
        rec = len(out)
        out += struct.pack("<IQHHIIQQQQ", 0x06064B50, z.get("size", 44), 798, 45, z["disk"], z["cd_disk"],
                           z["n_disk"], z["n_total"], z["cd_size"], z["cd_off"])
        out += struct.pack("<IIQI", 0x07064B50, z.get("loc_disk", 0), rec + z.get("loc_off_delta", 0),
                           z.get("loc_total", 1))
        out += bytes.fromhex(z.get("gap", ""))
    out += struct.pack("<IHHHHIIH", 0x06054B50, fields["disk"], fields["cd_disk"], fields["n_disk"],
                       fields["n_total"], fields["cd_size"], fields["cd_off"], 0)
    return bytes(out)


def torch_zip(pkl: bytes, **kw) -> bytes:
    return zip_bytes([("tm/data.pkl", pkl), ("tm/version", b"3\n"), ("tm/byteorder", b"little")], **kw)


def pkl(ids) -> bytes:
    return pickle.dumps(list(ids), protocol=2)


def token_cases():
    ok = pkl([1, 5, 300, 70000])
    vocab = 100000

    def case(id_, error, data, vocab_size=vocab, limits=None):
        c = {"id": id_, "tag": "token_map", "error": error, "file_hex": data.hex(), "vocab_size": vocab_size}
        if limits:
            c["limits"] = limits
        return c
    return [
        case("token_map_global_opcode", "pickle_grammar", torch_zip(b"\x80\x02c__builtin__\nlist\n)\x85R.")),
        case("token_map_reduce_after_list", "pickle_grammar", torch_zip(b"\x80\x02]K\x01aR.")),
        case("token_map_float_item", "pickle_grammar", torch_zip(b"\x80\x02]G?\xf0\x00\x00\x00\x00\x00\x00a.")),
        case("token_map_proto_1", "pickle_grammar", torch_zip(b"\x80\x01]K\x01a.")),
        case("token_map_frame_protocol_4", "pickle_grammar", torch_zip(pickle.dumps([1, 2], protocol=4))),
        case("token_map_trailing_after_stop", "pickle_grammar", torch_zip(ok + b"\x00")),
        case("token_map_truncated_pickle", "pickle_grammar", torch_zip(ok[:-1])),
        case("token_map_id_at_vocab", "token_range", torch_zip(ok), vocab_size=70000),
        case("token_map_negative_binint", "token_range", torch_zip(b"\x80\x02]q\x00(K\x01J\xff\xff\xff\xffe.")),
        case("token_map_duplicate_id", "token_order", torch_zip(pkl([1, 3, 3]))),
        case("token_map_decreasing", "token_order", torch_zip(pkl([5, 3]))),
        case("token_map_empty_list", "token_empty", torch_zip(b"\x80\x02].")),
        case("token_map_deflated_entry", "zip_unsupported", torch_zip(ok, method=8)),
        case("token_map_encrypted_entry", "zip_unsupported", torch_zip(ok, flags=0x809)),
        case("token_map_zip64", "zip_unsupported",
             torch_zip(ok, eocd={"n_disk": 0xFFFF, "n_total": 0xFFFF, "cd_size": 0xFFFFFFFF, "cd_off": 0xFFFFFFFF})),
        case("token_map_zip64_record_cd_offset", "zip_structure", torch_zip(ok, pad=True, zip64={"cd_off": 0})),
        case("token_map_zip64_record_entries", "zip_structure", torch_zip(ok, pad=True, zip64={"n_total": 4})),
        case("token_map_zip64_locator_offset", "zip_structure",
             torch_zip(ok, pad=True, zip64={"loc_off_delta": -8})),
        case("token_map_zip64_record_size", "zip_structure", torch_zip(ok, pad=True, zip64={"size": 56})),
        case("token_map_zip64_locator_disks", "zip_structure", torch_zip(ok, pad=True, zip64={"loc_total": 2})),
        case("token_map_entry_disk_start", "zip_unsupported", torch_zip(ok, disk_start=1)),
        case("token_map_entry_disk_start_zip64", "zip_unsupported", torch_zip(ok, disk_start=0xFFFF)),
        case("token_map_cd_trailing_bytes", "zip_structure", torch_zip(ok, cd_trailing=bytes(8))),
        # Entry bounds are checked before the entry's flags and method: a compressed entry overrunning the central
        # directory is `zip_structure`.
        case("token_map_entry_overruns_cd_before_method", "zip_structure",
             zip_bytes([("tm/data.pkl", ok)], method=8, eocd={"cd_size": 46 + len("tm/data.pkl") - 1})),
        # Ids are checked after the whole grammar (incl. STOP last): empty, then range, then order.
        case("token_map_range_before_order", "token_range", torch_zip(pkl([5, 3, 10**6])), vocab_size=1000),
        case("token_map_grammar_before_range", "pickle_grammar", torch_zip(pkl([1, 10**6]) + b"\x00"),
             vocab_size=1000),
        case("token_map_grammar_before_order", "pickle_grammar", torch_zip(pkl([5, 3])[:-1])),
        case("token_map_zip64_locator_not_before_eocd", "zip_structure",
             torch_zip(ok, pad=True, zip64={"gap": "00000000"})),
        case("token_map_multi_disk", "zip_unsupported", torch_zip(ok, eocd={"disk": 1})),
        case("token_map_two_data_pkl", "zip_entry",
             zip_bytes([("a/data.pkl", ok), ("b/data.pkl", ok)])),
        case("token_map_no_data_pkl", "zip_entry", zip_bytes([("tm/other.pkl", ok)])),
        case("token_map_nested_data_pkl", "zip_entry", zip_bytes([("tm/sub/data.pkl", ok)])),
        case("token_map_local_name_mismatch", "zip_entry", torch_zip(ok, local_names={0: "tm/datx.pkl"})),
        case("token_map_pkl_over_limit", "zip_entry", torch_zip(ok), limits={"max_token_map_pkl_bytes": 8}),
        case("token_map_no_eocd", "zip_structure", torch_zip(ok)[:-22]),
        case("token_map_cd_out_of_range", "zip_structure", torch_zip(ok, eocd={"cd_off": 1 << 20})),
    ]


def inventory_cases():
    """HF tree files (`tree_name`, `tree_text`) replacing the one of a copy of tests/fixtures/model_src."""
    trees = HERE.parents[1] / "tests/fixtures/model_src/.cache/huggingface/trees"
    (path,) = trees.iterdir()
    good = json.loads(path.read_text())
    name = path.name

    def case(id_, tree, tree_name=name):
        text = tree if isinstance(tree, str) else json.dumps(tree, indent=2, sort_keys=True) + "\n"
        return {"id": id_, "tag": "inventory", "error": "source_tree", "tree_name": tree_name, "tree_text": text}

    def entry(file, **kw):
        t = json.loads(json.dumps(good))
        t["files"][file].update(kw)
        for k, v in kw.items():
            if v is None:
                del t["files"][file][k]
        return t
    lfs = next(f for f, e in good["files"].items() if "lfs_sha256" in e)
    text = json.dumps(good, indent=2, sort_keys=True) + "\n"
    return [
        case("inventory_tree_revision_branch_name", good, "main.json"),
        case("inventory_tree_revision_upper_hex", good, name.upper().replace(".JSON", ".json")),
        case("inventory_tree_revision_empty", good, ".json"),
        case("inventory_tree_blob_id_short", entry("config.json", blob_id="5690296e")),
        case("inventory_tree_blob_id_missing", entry(lfs, blob_id=None)),
        case("inventory_tree_lfs_sha256_upper", entry(lfs, lfs_sha256=good["files"][lfs]["lfs_sha256"].upper())),
        case("inventory_tree_lfs_sha256_not_string", entry(lfs, lfs_sha256=1)),
        case("inventory_tree_size_negative", entry("config.json", size=-1)),
        case("inventory_tree_size_negative_zero", text.replace('"size": 806', '"size": -0', 1)),
        case("inventory_tree_size_float", text.replace('"size": 806', '"size": 806.0', 1)),
        case("inventory_tree_files_not_object", {"files": []}),
    ]


SH_EXPERT = "layer-00000-experts-0000-0001.safetensors"
SH_BF16 = "model-bf16-00001.safetensors"
SH_PLE = "model-plefp8-00000.safetensors"
EXP = "model.language_model.layers.0.mlp.experts.0.gate_proj."
PLE = "model.language_model.layers.1.ple."


def artifact_cases():
    def case(id_, error, *ops, limits=None, expected=None):
        c = {"id": id_, "tag": "artifact", "error": error, "ops": list(ops)}
        if limits:
            c["limits"] = limits
        if expected:
            c["expected_manifest_sha256"] = expected
        return c

    def shard(path, edit, reseal=True):
        return {"op": "shard", "path": path, "edit": edit, "reseal": reseal}

    def mset(pointer, value):
        return {"op": "manifest_set", "pointer": pointer, "value": value}

    def madd(pointer, delta=1):
        return {"op": "manifest_add", "pointer": pointer, "delta": delta}

    def fam(name):
        return {"family": name}

    def replace(old, new):
        return {"op": "manifest_raw", "transform": "replace", "old": old, "new": new}
    return [
        case("artifact_manifest_oversize", "manifest_size", limits={"max_manifest_bytes": 1024}),
        case("artifact_manifest_digest", "manifest_digest", expected="0" * 64),
        case("artifact_manifest_symlink", "symlink", {"op": "symlink", "path": "manifest.json"}),
        case("artifact_manifest_duplicate_key", "duplicate_key", {"op": "manifest_raw", "transform": "duplicate_key"}),
        case("artifact_manifest_reindented", "manifest_noncanonical", {"op": "manifest_raw", "transform": "reindent4"}),
        case("artifact_manifest_reordered_keys", "manifest_noncanonical",
             {"op": "manifest_raw", "transform": "unsort_keys"}),
        # A number outside the canonical value set has no canonical encoding (Q17): manifest_noncanonical.
        case("artifact_manifest_float", "manifest_noncanonical",
             replace('"nvfp4_group_size": 16,', '"nvfp4_group_size": 16.0,')),
        case("artifact_manifest_exponent", "manifest_noncanonical",
             replace('"nvfp4_group_size": 16,', '"nvfp4_group_size": 16e0,')),
        case("artifact_manifest_negative_zero", "manifest_noncanonical", replace('"misaligned": 0,', '"misaligned": -0,')),
        # The artifact_families schema patterns and bounds (Q5).
        case("artifact_families_rules_id_pattern", "manifest_schema", mset(["family_rules", "rules_id"], "has space")),
        case("artifact_families_rules_id_too_long", "manifest_schema", mset(["family_rules", "rules_id"], "r" * 129)),
        case("artifact_families_family_pattern", "manifest_schema",
             mset(["family_rules", "rules", 0, "family"], "-leading-dash")),
        case("artifact_families_nvfp4_group_size_max", "manifest_schema",
             mset(["family_rules", "groups", "nvfp4_group_size"], 4097)),
        case("artifact_families_ple_table_shards_max", "manifest_schema",
             mset(["family_rules", "groups", "ple_table_shards"], 65537)),
        case("artifact_families_member_pattern", "manifest_schema",
             mset(["family_rules", "groups", "ple_module_members", 0], "norm key")),
        case("artifact_families_members_max_items", "manifest_schema",
             mset(["family_rules", "groups", "ple_index_members"], [f"m{i}" for i in range(65)])),
        case("artifact_families_pattern_too_long", "manifest_schema",
             mset(["family_rules", "rules", 0, "pattern"], "a" * 4097)),
        case("artifact_schema_v1", "manifest_schema", mset(["schema_version"], "1")),
        case("artifact_schema_unknown_role", "manifest_schema", mset(["files", {"path": "vocab.json"}, "role"], "vocab")),
        case("artifact_schema_v1_dtype", "manifest_schema", mset(["families", 0, "dtype"], "uint8")),
        case("artifact_schema_profile_field", "manifest_schema", mset(["profile", "placement", "nvfp4_weight"], "gpu")),
        case("artifact_rules_duplicate_path", "manifest_rules",
             mset(["files", {"path": "vocab.json"}, "path"], "video_preprocessor_config.json")),
        case("artifact_rules_role_mismatch", "manifest_rules", mset(["files", {"path": "merges.txt"}, "role"], "config")),
        case("artifact_rules_totals", "manifest_rules", mset(["totals", "by_placement", "device", "bytes"], 1)),
        case("artifact_rules_profile_id", "manifest_rules", mset(["profile_id"], "other-profile")),
        case("artifact_rules_model_hash", "manifest_rules", mset(["model_hash"], "0" * 64)),
        case("artifact_rules_placement", "manifest_rules", mset(["families", 0, "placement"], "host_pinned")),
        case("artifact_path_traversal", "unsafe_path", mset(["files", {"path": "vocab.json"}, "path"], "../vocab.json")),
        case("artifact_path_absolute", "unsafe_path", mset(["files", {"path": "vocab.json"}, "path"], "/etc/passwd")),
        case("artifact_too_many_files", "too_many_files", limits={"max_files": 3}),
        case("artifact_unlisted_file", "unlisted_file", {"op": "add_file", "path": "stray.json", "hex": "7b7d0a"}),
        case("artifact_unlisted_dir", "unlisted_file", {"op": "add_file", "path": "extra/a.bin", "hex": "00"}),
        case("artifact_symlinked_payload", "symlink", {"op": "symlink", "path": "config.json"}),
        case("artifact_symlinked_shard", "symlink", {"op": "symlink", "path": SH_PLE}),
        case("artifact_fifo_unlisted", "special_file", {"op": "fifo", "path": "pipe"}),
        case("artifact_fifo_payload", "special_file", {"op": "fifo", "path": "vocab.json"}),
        case("artifact_file_size", "file_size", {"op": "add_file", "path": "merges.txt", "hex": "00"}),
        case("artifact_too_many_total_tensors", "too_many_tensors", limits={"max_total_tensors": 10}),
        case("artifact_header_hash", "header_hash", {"op": "flip_byte", "path": SH_PLE, "offset": 20}),
        case("artifact_family_unmatched", "family_unmatched",
             shard(SH_BF16, {"rename": ["model.visual.blocks.0.attn.qkv.bias", "model.audio.blocks.0.bias"]})),
        case("artifact_family_ambiguous", "family_ambiguous",
             {"op": "manifest_append", "pointer": ["family_rules", "rules"],
              "value": {"family": "dup_lm_head", "pattern": "lm_head\\.weight", "component": "target",
                        "role": "dense", "dtype": "BF16", "shape_rule": "any", "placement": "device"}},
             {"op": "manifest_append", "pointer": ["families"],
              "value": {"family": "dup_lm_head", "component": "target", "role": "dense", "dtype": "BF16",
                        "placement": "device", "count": 0, "bytes": 0, "misaligned": 0}}),
        case("artifact_family_dtype", "family_dtype",
             shard(SH_BF16, {"set": {"tensor": "model.visual.blocks.0.attn.qkv.bias", "dtype": "F16"}})),
        case("artifact_scalar_shape_misuse", "family_shape",
             shard(SH_EXPERT, {"set": {"tensor": EXP + "weight_scale_2", "shape": [1]}})),
        case("artifact_family_rank", "family_shape",
             shard(SH_EXPERT, {"set": {"tensor": EXP + "weight", "shape": [256]}})),
        case("artifact_expert_metadata_range", "expert_metadata",
             shard(SH_EXPERT, {"metadata": {"expert_start": "0", "expert_end": "3", "format": "pt", "layer": "0"}})),
        case("artifact_expert_metadata_layer", "expert_metadata",
             shard(SH_EXPERT, {"metadata": {"expert_start": "0", "expert_end": "2", "format": "pt", "layer": "1"}})),
        case("artifact_quad_missing_member", "group_nvfp4", shard(SH_EXPERT, {"drop": EXP + "input_scale"})),
        case("artifact_quad_scale_shape", "group_nvfp4",
             shard(SH_EXPERT, {"set": {"tensor": EXP + "weight_scale", "shape": [8, 4]}})),
        case("artifact_ple_shard_missing", "group_ple_table",
             shard(SH_PLE, {"drop": PLE + "ple_embedding.ngram_embedding.shard_3.weight"})),
        case("artifact_ple_shard_shape", "group_ple_table",
             shard(SH_PLE, {"set": {"tensor": PLE + "ple_embedding.ngram_embedding.shard_1.weight", "shape": [8, 5]}})),
        case("artifact_ple_index_missing", "group_ple_table",
             shard(SH_BF16, {"drop": PLE + "ple_embedding.layer_multipliers"})),
        case("artifact_ple_module_missing", "group_ple_module", shard(SH_BF16, {"drop": PLE + "norm_key.weight"})),
        case("artifact_family_count", "family_totals",
             madd(["families", fam("vision"), "count"]), madd(["totals", "by_dtype", "BF16", "count"]),
             madd(["totals", "by_placement", "unused", "count"])),
        case("artifact_file_tensor_count", "family_totals",
             mset(["files", {"path": SH_PLE}, "safetensors", "tensors"], 5)),
        case("artifact_tensor_byte_flip", "file_hash", {"op": "flip_byte", "path": SH_BF16, "offset": -1}),
        case("artifact_config_byte_flip", "file_hash", {"op": "flip_byte", "path": "config.json", "offset": 0}),
        case("artifact_token_map_byte_flip", "file_hash", {"op": "flip_byte", "path": "draft/token_map.u32le",
                                                           "offset": 0}),
    ]


def valid_token_maps():
    """Positive token maps: the fixture map rewritten in PyTorch's real layout (FB-padded local headers, zip64 end
    record + locator before the classic EOCD), with the u32le bytes it must convert to."""
    import zipfile
    import token_map
    src = HERE.parents[1] / "tests/fixtures/model_src"
    raw = (src / "token_map.pt").read_bytes()
    vocab = token_map.vocab_size_of((src / "config.json").read_bytes())
    with zipfile.ZipFile(src / "token_map.pt") as z:
        entries = [(i.filename, z.read(i)) for i in z.infolist()]
    data = zip_bytes(entries, pad=True, zip64={})
    return [{"id": "token_map_torch_layout_fixture", "file_hex": data.hex(), "vocab_size": vocab,
             "u32le_hex": token_map.convert(raw, vocab).hex()}]


def build() -> bytes:
    cases = header_cases() + token_cases() + artifact_cases() + inventory_cases()
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids)
    classes = [c for tag in ("header", "artifact", "token_map", "inventory") for c in ERROR_CLASSES[tag]]
    assert all(c["error"] in classes for c in cases)
    return canonical_json({"schema_version": "1", "error_classes": classes, "default_limits": DEFAULT_LIMITS,
                           "cases": cases, "valid_token_maps": valid_token_maps()})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    data = build()
    if args.check:
        if not OUT.is_file() or OUT.read_bytes() != data:
            sys.stderr.write(f"{OUT} is stale; rerun {Path(__file__).name}\n")
            return 2
        return 0
    OUT.write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
