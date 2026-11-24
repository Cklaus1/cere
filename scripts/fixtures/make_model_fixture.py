"""Deterministic, stdlib-only generator of the tiny source model fixture (M2.1 Q14).

Writes `tests/fixtures/model_src/` (3 safetensors shards in the real model's naming, the HF payload stubs, stray
non-payload files, the HF tree metadata and a torch-style stored-zip token map) plus
`tests/fixtures/model_src.{lock,families,profile}.json`. `--check` verifies the committed bytes instead of writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
VOCAB = 70000
HIDDEN = 32
MOE = 16
EXPERTS = 2
PLE_SHARDS = 4
TOKEN_MAP_COUNT = 2001  # three MARK..APPENDS batches (1000, 1000, 1) over BININT1, BININT2 and BININT ids


def canonical(value) -> bytes:
    """Q17 canonical JSON (sorted keys, 2-space indent, ASCII, trailing newline, integers only)."""
    def no_floats(v):
        if isinstance(v, float):
            raise ValueError("canonical JSON forbids floats")
        if isinstance(v, dict):
            for x in v.values():
                no_floats(x)
        elif isinstance(v, list):
            for x in v:
                no_floats(x)
    no_floats(value)
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, separators=(",", ": ")) + "\n").encode()


def det_bytes(seed: str, n: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}:{i}".encode()).digest()
        i += 1
    return bytes(out[:n])


ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "U8": 1, "U16": 2,
            "U32": 4, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}


def write_safetensors(tensors, metadata=None, header_mod8: int | None = 0) -> bytes:
    """`tensors`: list of (name, dtype, shape, data) in data order. Header is compact JSON padded with spaces so
    that (8+n) % 8 == header_mod8 (None: no padding)."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    off = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
    hb = json.dumps(header, separators=(",", ":")).encode()
    if header_mod8 is not None:
        while (8 + len(hb)) % 8 != header_mod8:
            hb += b" "
    return struct.pack("<Q", len(hb)) + hb + b"".join(t[3] for t in tensors)


def tensor(name, dtype, shape):
    n = ITEMSIZE[dtype]
    for d in shape:
        n *= d
    return (name, dtype, list(shape), det_bytes(name, n))


def data_order(ts):
    """safetensors writer order: larger element alignment first, then name."""
    return sorted(ts, key=lambda t: (-ITEMSIZE[t[1]], t[0]))


def expert_shard():
    ts = []
    for e in range(EXPERTS):
        for proj, (o, i) in (("gate_proj", (MOE, HIDDEN)), ("up_proj", (MOE, HIDDEN)), ("down_proj", (HIDDEN, MOE))):
            p = f"model.language_model.layers.0.mlp.experts.{e}.{proj}"
            ts += [tensor(p + ".weight", "U8", [o, i // 2]), tensor(p + ".weight_scale", "F8_E4M3", [o, i // 16]),
                   tensor(p + ".weight_scale_2", "F32", []), tensor(p + ".input_scale", "F32", [])]
    md = {"expert_start": "0", "expert_end": str(EXPERTS), "format": "pt", "layer": "0"}
    return write_safetensors(data_order(ts), md, header_mod8=0)


def bf16_shard():
    L = "model.language_model."
    ple = L + "layers.1.ple."
    ts = [
        tensor("lm_head.weight", "BF16", [8, HIDDEN]),
        tensor(L + "embed_tokens.weight", "BF16", [8, HIDDEN]),
        tensor(L + "hyper_connection_mixer.hc_norm.weight", "BF16", [HIDDEN]),
        tensor(L + "layers.0.attn_hyper_connection.hc_norm.weight", "BF16", [HIDDEN]),
        tensor(L + "layers.0.mlp_hyper_connection.block_inject_weight.weight", "BF16", [4, HIDDEN]),
        tensor(L + "layers.0.linear_attn.A_log", "BF16", [4]),
        tensor(L + "layers.0.linear_attn.conv1d.weight", "BF16", [8, 1, 4]),
        tensor(L + "layers.0.linear_attn.in_proj_qkv.weight", "BF16", [8, HIDDEN]),
        tensor(L + "layers.0.self_attn.q_proj.weight", "BF16", [8, HIDDEN]),
        tensor(L + "layers.0.self_attn.indexer.k_layernorm.weight", "BF16", [8]),
        tensor(L + "layers.0.mlp.gate.weight", "BF16", [EXPERTS, HIDDEN]),
        tensor(L + "layers.0.mlp.shared_expert.down_proj.weight", "BF16", [HIDDEN, MOE]),
        tensor(L + "layers.0.mlp.shared_expert_gate.weight", "BF16", [1, HIDDEN]),
        tensor(ple + "key_proj.weight", "BF16", [16, HIDDEN]),
        tensor(ple + "value_proj.weight", "BF16", [HIDDEN, HIDDEN]),
        tensor(ple + "conv1d.weight", "BF16", [16, 1, 4]),
        tensor(ple + "norm_conv.weight", "BF16", [16]),
        tensor(ple + "norm_key.weight", "BF16", [16]),
        tensor(ple + "norm_query.weight", "BF16", [16]),
        tensor(ple + "ple_embedding.ngram_embedding.weight_scale", "BF16", [1]),
        tensor(ple + "ple_embedding.layer_multipliers", "I64", [3]),
        tensor(ple + "ple_embedding.ngram_heads_offsets", "I64", [4]),
        tensor(ple + "ple_embedding.ngram_heads_vocab_sizes", "I64", [4]),
        tensor("mtp.fc_hidden.weight", "BF16", [HIDDEN, HIDDEN]),
        tensor("mtp.pre_fc_norm_hidden.weight", "BF16", [HIDDEN]),
        # Larger than three 64 KiB staging buffers (R5 sink tiling test).
        tensor("mtp.layers.0.mlp.experts.gate_up_proj", "BF16", [EXPERTS, 320, 128]),
        tensor("model.visual.patch_embed.proj.weight", "BF16", [4, 3, 2, 2, 2]),
        tensor("model.visual.blocks.0.attn.qkv.bias", "BF16", [12]),
    ]
    # (8+n) odd: every BF16/I64 tensor in this shard starts at an element-misaligned absolute offset, as in the
    # real model-bf16-* shards (Q16).
    return write_safetensors(data_order(ts), None, header_mod8=1)


def ple_shard():
    ts = [tensor(f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{k}.weight", "F8_E4M3", [5, 8])
          for k in range(PLE_SHARDS)]
    return write_safetensors(data_order(ts), None, header_mod8=3)


def token_map_ids() -> list[int]:
    ids = list(range(0, 256, 2)) + list(range(256, 65536, 50)) + list(range(65536, VOCAB, 3))
    return ids[:TOKEN_MAP_COUNT]


def stored_zip(entries, prefix_extra_align: int = 64) -> bytes:
    """torch-style stored zip: flag bits 0x808 (data descriptor + UTF-8), local 'FB' padding extra field aligning
    each entry's data to 64 bytes, sizes/CRC in the data descriptor and central directory only."""
    out = bytearray()
    central = bytearray()
    for name, data in entries:
        nb = name.encode()
        offset = len(out)
        base = offset + 30 + len(nb) + 4
        pad = (-base) % prefix_extra_align
        extra = struct.pack("<HH", 0x4246, pad) + b"Z" * pad
        crc = zlib.crc32(data) & 0xFFFFFFFF
        out += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0x808, 0, 0, 0x21, 0, 0, 0, len(nb), len(extra))
        out += nb + extra + data
        out += struct.pack("<IIII", 0x08074B50, crc, len(data), len(data))
        central += struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 0x031E, 20, 0x808, 0, 0, 0x21, crc, len(data),
                               len(data), len(nb), 0, 0, 0, 0, 0, offset) + nb
    cd_off = len(out)
    out += central
    out += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries), len(central), cd_off, 0)
    return bytes(out)


def torch_list_zip(pkl: bytes, archive: str = "fixture-map", extra_entries=()) -> bytes:
    return stored_zip([(f"{archive}/data.pkl", pkl), *extra_entries,
                       (f"{archive}/.format_version", b"1"), (f"{archive}/.storage_alignment", b"64"),
                       (f"{archive}/byteorder", b"little"), (f"{archive}/version", b"3\n"),
                       (f"{archive}/.data/serialization_id", b"0280012345678901234567890123456789012345")])


def pretty(value) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def lock_digest(files: dict, names) -> str:
    lines = "".join(f"{hashlib.sha256(files[n]).hexdigest()}  {n}\n" for n in sorted(names, key=str.encode))
    return hashlib.sha256(lines.encode()).hexdigest()


def families(rules_id: str, ple_shards: int) -> dict:
    L = r"model\.language_model\."
    LN = L + r"layers\.[0-9]+\."
    EX = LN + r"mlp\.experts\.[0-9]+\.(gate|up|down)_proj\."
    PE = LN + r"ple\.ple_embedding\."

    def rule(family, pattern, component, role, dtype, shape_rule, placement):
        return {"family": family, "pattern": pattern, "component": component, "role": role, "dtype": dtype,
                "shape_rule": shape_rule, "placement": placement}
    return {
        "schema_version": "1",
        "rules_id": rules_id,
        "groups": {
            "nvfp4_group_size": 16,
            "ple_table_shards": ple_shards,
            "ple_module_members": ["conv1d.weight", "key_proj.weight", "norm_conv.weight", "norm_key.weight",
                                   "norm_query.weight", "value_proj.weight"],
            "ple_index_members": ["layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"],
        },
        "rules": [
            rule("target_nvfp4_weight", EX + r"weight", "target", "nvfp4_weight", "U8", "rank:2", "device"),
            rule("target_nvfp4_block_scale", EX + r"weight_scale", "target", "nvfp4_block_scale", "F8_E4M3",
                 "rank:2", "device"),
            rule("target_nvfp4_global_scale", EX + r"weight_scale_2", "target", "nvfp4_global_scale", "F32",
                 "rank:0", "device"),
            rule("target_nvfp4_input_scale", EX + r"input_scale", "target", "nvfp4_input_scale", "F32", "rank:0",
                 "device"),
            rule("target_dense",
                 r"lm_head\.weight|" + L + r"(embed_tokens\.weight|hyper_connection_mixer\.[a-z_]+\.weight|"
                 r"layers\.[0-9]+\.((attn|mlp)_hyper_connection\.[a-z_]+\.weight|linear_attn\.[A-Za-z0-9_.]+|"
                 r"self_attn\.[a-z_.]+\.weight|mlp\.(gate|shared_expert_gate|shared_expert\.(gate|up|down)_proj)"
                 r"\.weight))",
                 "target", "dense", "BF16", "any", "device"),
            rule("target_ple_module",
                 LN + r"ple\.(key_proj|value_proj|conv1d|norm_conv|norm_key|norm_query)\.weight",
                 "target", "ple_module", "BF16", "any", "device"),
            rule("target_ple_table", PE + r"ngram_embedding\.shard_[0-9]+\.weight", "target", "ple_table",
                 "F8_E4M3", "rank:2", "host_pinned"),
            rule("target_ple_scale", PE + r"ngram_embedding\.weight_scale", "target", "ple_scale", "BF16",
                 "rank:1", "host_pinned"),
            rule("target_ple_index", PE + r"(layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes)",
                 "target", "ple_index", "I64", "rank:1", "host_pinned"),
            rule("draft_mtp", r"mtp\..+", "draft", "draft", "BF16", "any", "device"),
            rule("vision", r"model\.visual\..+", "vision", "vision", "BF16", "any", "unused"),
        ],
    }


def profile(profile_id, device_model, memory_mib, cc) -> dict:
    return {
        "schema_version": "1",
        "profile_id": profile_id,
        "device_model": device_model,
        "device_memory_mib": memory_mib,
        "compute_capability": cc,
        "compute_capability_source": "inferred",
        "modalities": ["text"],
        "speculative": "nextn",
        "placement": {
            "nvfp4_weight": "device", "nvfp4_block_scale": "device", "nvfp4_global_scale": "device",
            "nvfp4_input_scale": "device", "dense": "device", "ple_module": "device", "draft": "device",
            "ple_table": "host_pinned", "ple_scale": "host_pinned", "ple_index": "host_pinned", "vision": "unused",
        },
    }


def build() -> dict:
    """Return {path relative to tests/fixtures: bytes}."""
    src = {}
    shards = {
        "layer-00000-experts-0000-0001.safetensors": expert_shard(),
        "model-bf16-00001.safetensors": bf16_shard(),
        "model-plefp8-00000.safetensors": ple_shard(),
    }
    src.update(shards)
    weight_map = {}
    for shard, raw in shards.items():
        n = struct.unpack("<Q", raw[:8])[0]
        for name in json.loads(raw[8:8 + n]):
            if name != "__metadata__":
                weight_map[name] = shard
    src["model.safetensors.index.json"] = pretty(
        {"metadata": {"total_size": sum(len(b) for b in shards.values())}, "weight_map": dict(sorted(weight_map.items()))})
    text_config = {
        "model_type": "qwen4_exp_text", "vocab_size": VOCAB, "hidden_size": HIDDEN, "moe_intermediate_size": MOE,
        "num_experts": EXPERTS, "num_experts_per_tok": 1, "num_hidden_layers": 1, "full_attention_interval": 4,
        "ple_layer_ids": [2], "ngram_size": 3, "split_ngram_parts": PLE_SHARDS, "ple_embedding_dtype": "float8_e4m3fn",
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False, "mamba_ssm_dtype": "float32",
    }
    quant = {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16, "ignore": ["lm_head"]}
    upstream_config = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
                       "language_model_only": False, "tie_word_embeddings": False,
                       "quantization_config": quant, "text_config": text_config}
    local_quant = dict(quant, ignore=["lm_head", "model.visual*", "mtp*"],
                       exclude_modules=["lm_head", "model.visual*", "mtp*", "*.ple.*"])
    src["config.json.orig"] = pretty(upstream_config)
    src["config.json"] = pretty(dict(upstream_config, quantization_config=local_quant))
    src["hf_quant_config.json"] = pretty({"producer": {"name": "modelopt", "version": "0.46.0"},
                                          "quantization": {"exclude_modules": ["lm_head", "mtp.*", "*.ple.*"],
                                                           "group_size": 16, "quant_algo": "NVFP4"}})
    src["generation_config.json"] = pretty({"do_sample": True, "eos_token_id": [7, 6], "temperature": 1.0,
                                            "top_k": 20, "top_p": 0.95})
    src["preprocessor_config.json"] = pretty({"image_processor_type": "FixtureImageProcessor", "patch_size": 2})
    src["video_preprocessor_config.json"] = pretty({"video_processor_type": "FixtureVideoProcessor", "patch_size": 2})
    src["tokenizer.json"] = pretty({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}}})
    src["tokenizer_config.json"] = pretty({"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 262144})
    src["vocab.json"] = pretty({"a": 0, "b": 1, "ab": 2})
    src["merges.txt"] = b"#version: 0.2\na b\n"
    src["chat_template.jinja"] = (b"{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
                                  b"{%- endfor %}\n")
    src["layer-00000-experts-0000-0001.complete.json"] = pretty({"layer": 0, "experts": [0, EXPERTS - 1]})
    src["qualification-notes.txt"] = b"Fixture stray file: present in the source directory, never in the payload.\n"
    src["token_map.pt"] = torch_list_zip(pickle.dumps(token_map_ids(), protocol=2))

    tree = {}
    for name in sorted(src):
        data = src["config.json.orig"] if name == "config.json" else src[name]
        if name in ("config.json.orig", "token_map.pt", "layer-00000-experts-0000-0001.complete.json"):
            continue
        entry = {"size": len(data), "blob_id": git_blob_sha1(data)}
        if name.endswith(".safetensors") or name in ("model.safetensors.index.json", "tokenizer.json"):
            sha = hashlib.sha256(data).hexdigest()
            entry.update(lfs_sha256=sha, lfs_size=len(data), xet_hash=hashlib.sha256(b"xet" + data).hexdigest())
        tree[name] = entry
    tree_bytes = json.dumps({"format_version": 1, "files": tree}, indent=1).encode()

    out = {f"model_src/{k}": v for k, v in src.items()}
    out[f"model_src/.cache/huggingface/trees/{REVISION}.json"] = tree_bytes
    top = list(src)
    out["model_src.lock.json"] = pretty({
        "schema_version": "1",
        "lock_id": "fixture-model-lock-v1",
        "recipe_sha256": "0" * 64,
        "artifact_sha256": lock_digest(src, top),
        "model_content_sha256": lock_digest(src, [*shards, "model.safetensors.index.json", "config.json",
                                                  "hf_quant_config.json"]),
        "tokenizer_sha256": lock_digest(src, ["merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]),
        "template_sha256": hashlib.sha256(src["chat_template.jinja"]).hexdigest(),
        "backend_build_sha256": "0" * 64,
        "backend_profile_id": "fixture-backend-v1",
        "effective_context_tokens": 4096,
        "effective_operations": ["chat"],
        "adapter_sha256s": [],
    })
    out["model_src.families.json"] = canonical(families("fixture-qwen4_exp-nvfp4-v1", PLE_SHARDS))
    out["model_src.profile.json"] = canonical(profile("fixture-plehost-text-v1", "fixture-cpu", 1024, "0.0"))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "tests/fixtures", help="fixtures directory to write")
    p.add_argument("--check", action="store_true", help="verify the files under --out instead of writing")
    args = p.parse_args()
    files = build()
    if args.check:
        bad = [k for k, v in files.items() if not (args.out / k).is_file() or (args.out / k).read_bytes() != v]
        extra = sorted(p.relative_to(args.out).as_posix() for p in (args.out / "model_src").rglob("*")
                       if p.is_file() and p.relative_to(args.out).as_posix() not in files)
        if bad or extra:
            print(json.dumps({"differs": bad, "unexpected": extra}), file=sys.stderr)
            return 2
        return 0
    for k, v in sorted(files.items()):
        path = args.out / k
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
