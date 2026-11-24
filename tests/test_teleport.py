"""NetTeleport S4 (net-teleport design note 4.2 L2, 4.1, 4.3, D-T1, D-T5, D-T7): scripts/live/teleport.py (the client
tool: key custody, tier policy before any byte moves, the decision rule, export/import/move over the loopback API,
direct and through a vault) and scripts/live/teleport_vault.py (the vault and its at-rest scan), on the CPU against
fake loopback installs. Nothing starts a container, touches a GPU or leaves the host."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import io
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/live"))
import teleport  # noqa: E402
import teleport_vault as tv  # noqa: E402

FRAME = 4 << 20


def sealed_blob(image_bytes: int = 300_000, tokens: int = 1000, frame: int = FRAME) -> bytes:
    """A blob shaped like CERESTB1 (header, then image + one tag per frame of random bytes)."""
    header = {"format": 1, "library_sha256": "a" * 64, "profile_id": "p", "layout_id": "b" * 64,
              "state_compat_id": "c" * 64, "image_sha256": "d" * 64, "image_bytes": image_bytes, "tokens": tokens,
              "source_tier": "own", "allowed_tiers": ["own"], "created_unix_ms": 1, "frame_bytes": frame,
              "salt": "e" * 64, "key_check": "f" * 32}
    raw = json.dumps(header, separators=(",", ":")).encode()
    frames = -(-image_bytes // frame)
    return tv.MAGIC + struct.pack("<I", len(raw)) + raw + os.urandom(image_bytes + frames * tv.TAG_BYTES)


TABLE = {
    "schema": teleport.TABLE_SCHEMA,
    "boards": {"rtx-pro-6000": {"rows": [
        {"tokens": 18_000, "blob_bytes": 300_000_000, "export_ms": 200.0, "import_ms": 150.0, "first_turn_ms": 100.0,
         "reprefill_ms": 1_900.0},
        {"tokens": 72_000, "blob_bytes": 1_000_000_000, "export_ms": 600.0, "import_ms": 450.0,
         "first_turn_ms": 150.0, "reprefill_ms": 8_200.0},
        {"tokens": 128_000, "blob_bytes": 1_730_000_000, "export_ms": 1_000.0, "import_ms": 800.0,
         "first_turn_ms": 200.0, "reprefill_ms": 20_000.0}]}},
    "links": {"loopback": {"transfer_mb_per_s": None}, "tailnet-hop": {"transfer_mb_per_s": 100.0},
              "tailnet-relay": {"transfer_mb_per_s": 50.0}},
}


class FakeInstall:
    """A loopback `cered` stand-in serving the two state routes: export answers `blob` (or `export_refusal`),
    import reads the whole body and records it (or answers `import_refusal`)."""

    def __init__(self, blob: bytes = b"", export_refusal: tuple[int, str] | None = None,
                 import_refusal: tuple[int, str] | None = None) -> None:
        self.blob, self.export_refusal, self.import_refusal = blob, export_refusal, import_refusal
        self.requests: list[dict] = []
        self.received = b""
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def refuse(self, status: int, code: str) -> None:
                body = json.dumps({"error": {"code": code}}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                fake.requests.append({"path": self.path, "key": self.headers.get(teleport.KEY_HEADER),
                                      "auth": self.headers.get("Authorization"), "length": n})
                if self.path == "/v1/cere/state/export":
                    request = json.loads(self.rfile.read(n))
                    fake.requests[-1]["body"] = request
                    if fake.export_refusal:
                        return self.refuse(*fake.export_refusal)
                    self.send_response(200)
                    self.send_header("Content-Type", teleport.BLOB_TYPE)
                    self.send_header("Content-Length", str(len(fake.blob)))
                    self.send_header("X-Cere-State-Tokens", "1000")
                    self.send_header("X-Cere-State-Image-Bytes", "300000")
                    self.send_header("X-Cere-State-Compat-Id", "c" * 64)
                    self.end_headers()
                    for i in range(0, len(fake.blob), 65536):
                        self.wfile.write(fake.blob[i:i + 65536])
                    return None
                if self.path.startswith("/v1/cere/state/import?model="):
                    if fake.import_refusal:
                        self.rfile.read(n)
                        return self.refuse(*fake.import_refusal)
                    fake.received = self.rfile.read(n)
                    body = json.dumps({"imported_tokens": 1000, "image_bytes": 300000, "tier": "own"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return None
                return self.refuse(404, "not_found")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeInstall":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="test-teleport-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.spool = self.tmp / "spool"
        self.spool.mkdir(mode=0o700)
        self.key_file = teleport.keygen(self.tmp / "cfg" / "teleport.key")
        self.messages = self.tmp / "chat.json"
        self.messages.write_text(json.dumps([{"role": "user", "content": "x"}]))
        self.table = self.tmp / "table.json"
        self.table.write_text(json.dumps(TABLE))
        self.env = {"CERE_TELEPORT_SPOOL": str(self.spool), "TOK_A": "token-a", "TOK_B": "token-b"}
        patcher = unittest.mock.patch.dict(os.environ, {"CERE_TELEPORT_SPOOL": str(self.spool)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_tool(self, *argv: str) -> tuple[int, dict]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = teleport.main(["--key-file", str(self.key_file), *argv], environ=self.env)
        return code, json.loads(out.getvalue())


class KeyCustody(TempCase):
    def test_keygen_writes_32_random_bytes_0600_in_a_0700_dir_and_never_overwrites(self):
        self.assertEqual(stat.S_IMODE(self.key_file.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.key_file.parent.stat().st_mode), 0o700)
        key = teleport.load_key(self.key_file)
        self.assertEqual(len(bytes.fromhex(key)), 32)
        with self.assertRaises(teleport.TeleportError) as e:
            teleport.keygen(self.key_file)
        self.assertEqual(e.exception.code, "key_file")
        self.assertEqual(teleport.load_key(self.key_file), key)
        self.assertNotEqual(teleport.load_key(teleport.keygen(self.tmp / "other.key")), key)

    def test_load_key_refuses_a_shared_file_bad_content_and_a_missing_file(self):
        os.chmod(self.key_file, 0o644)
        with self.assertRaisesRegex(teleport.TeleportError, "chmod 600"):
            teleport.load_key(self.key_file)
        bad = self.tmp / "bad.key"
        bad.write_text("zz" * 32)
        os.chmod(bad, 0o600)
        with self.assertRaisesRegex(teleport.TeleportError, "64 hex"):
            teleport.load_key(bad)
        with self.assertRaisesRegex(teleport.TeleportError, "keygen"):
            teleport.load_key(self.tmp / "absent.key")

    def test_default_key_file_honours_the_environment(self):
        self.assertEqual(teleport.default_key_file({"CERE_TELEPORT_KEY_FILE": "/x/k"}), Path("/x/k"))
        self.assertEqual(teleport.default_key_file({}), Path.home() / ".config/cere/teleport.key")


class TierPolicy(TempCase):
    def test_allowed_tiers_default_own_env_and_validation(self):
        self.assertEqual(teleport.allowed_tiers(None, {}), ("own",))
        self.assertEqual(teleport.allowed_tiers(None, {"CERE_TELEPORT_ALLOWED_TIERS": "own,team"}), ("own", "team"))
        self.assertEqual(teleport.allowed_tiers("provider", {"CERE_TELEPORT_ALLOWED_TIERS": "own"}), ("provider",))
        for bad in ("own,own", "lan", ","):
            with self.assertRaises(teleport.TeleportError):
                teleport.parse_tiers(bad)
        teleport.check_tier("destination", "own", ("own",))
        with self.assertRaises(teleport.TeleportError) as e:
            teleport.check_tier("destination", "team", ("own",))
        self.assertEqual(e.exception.code, "tier_not_allowed")

    def test_move_to_a_disallowed_tier_refuses_before_any_byte_moves(self):
        """D-T7: the client refuses first; neither install sees a request, and no replay is suggested."""
        with FakeInstall(sealed_blob()) as src, FakeInstall() as dst:
            code, out = self.run_tool("move", "--source", src.url, "--dest", dst.url, "--source-token-env", "TOK_A",
                                      "--dest-token-env", "TOK_B", "--model", "m", "--messages", str(self.messages),
                                      "--dest-tier", "team", "--force-state")
        self.assertEqual(code, 2)
        self.assertEqual((out["mode"], out["code"]), ("refused", "tier_not_allowed"))
        self.assertEqual(src.requests + dst.requests, [])

    def test_export_to_a_disallowed_vault_tier_refuses_before_the_export(self):
        with FakeInstall(sealed_blob()) as src:
            code, out = self.run_tool("export", "--source", src.url, "--token-env", "TOK_A", "--model", "m",
                                      "--messages", str(self.messages), "--dest-tier", "own",
                                      "--to-vault", str(self.tmp / "vault"), "--vault-tier", "provider")
        self.assertEqual((code, out["code"]), (2, "tier_not_allowed"))
        self.assertEqual(src.requests, [])


class DecisionRule(unittest.TestCase):
    def test_interpolation_is_piecewise_linear_and_extends_the_outer_segments(self):
        rows = TABLE["boards"]["rtx-pro-6000"]["rows"]
        self.assertEqual(teleport.interpolate(rows, "reprefill_ms", 72_000), 8_200.0)
        self.assertAlmostEqual(teleport.interpolate(rows, "reprefill_ms", 45_000), 5_050.0)
        self.assertAlmostEqual(teleport.interpolate(rows, "reprefill_ms", 200_000),
                               20_000 + (20_000 - 8_200) / 56_000 * 72_000)
        self.assertAlmostEqual(teleport.interpolate(rows, "export_ms", 0), 200 - 400 / 3)  # extended below
        falling = [{"tokens": 10, "x": 5.0}, {"tokens": 20, "x": 15.0}]
        self.assertEqual(teleport.interpolate(falling, "x", 0), 0.0)  # extended below 0: clamped

    def test_ship_iff_export_transfer_import_first_turn_beat_the_reprefill(self):
        # Loopback: 200 + 0 + 150 + 100 = 450 ms < 1,900 ms.
        d = teleport.decide(TABLE, 18_000, link="loopback")
        self.assertEqual((d["mode"], d["reason"], d["transfer_ms"], d["teleport_ms"]),
                         ("state", "state_faster", 0.0, 450.0))
        # Relay at 50 MB/s: 300 MB is 6,000 ms of transfer > 1,900 ms of re-prefill.
        d = teleport.decide(TABLE, 18_000, link="tailnet-relay")
        self.assertEqual((d["mode"], d["reason"], d["transfer_ms"]), ("replay", "reprefill_faster", 6_000.0))
        # One hop at 100 MB/s at 128K: 1,000 + 17,300 + 800 + 200 = 19,300 < 20,000.
        self.assertEqual(teleport.decide(TABLE, 128_000, link="tailnet-hop")["mode"], "state")
        self.assertEqual(teleport.decide(TABLE, 128_000, link="tailnet-relay")["mode"], "replay")
        self.assertEqual(teleport.decide(TABLE, 18_000, link_mbps=10_000.0)["mode"], "state")

    def test_contended_destination_ships_and_an_unknown_board_replays(self):
        d = teleport.decide(TABLE, 18_000, link="tailnet-relay", contended=True)
        self.assertEqual((d["mode"], d["reason"]), ("state", "destination_contended"))
        d = teleport.decide(TABLE, 18_000, board="strix-halo-vulkan", link="loopback")
        self.assertEqual((d["mode"], d["reason"]), ("replay", "no_estimate_for_board"))
        with self.assertRaises(teleport.TeleportError):
            teleport.decide(TABLE, 18_000, link="wan")
        with self.assertRaises(teleport.TeleportError):
            teleport.decide(TABLE, 18_000, link_mbps=0)

    def test_load_table_checks_the_schema_and_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.json"
            path.write_text(json.dumps(TABLE))
            self.assertEqual(teleport.load_table(path)["links"]["tailnet-hop"]["transfer_mb_per_s"], 100.0)
            bad = json.loads(json.dumps(TABLE))
            del bad["boards"]["rtx-pro-6000"]["rows"][0]["reprefill_ms"]
            path.write_text(json.dumps(bad))
            with self.assertRaises(teleport.TeleportError):
                teleport.load_table(path)
            path.write_text(json.dumps({**TABLE, "schema": "x"}))
            with self.assertRaises(teleport.TeleportError):
                teleport.load_table(path)

    def test_the_committed_table_is_measured_and_loads(self):
        table = teleport.load_table()
        rows = table["boards"][teleport.DEFAULT_BOARD]["rows"]
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(table["measured"]["report"].startswith("bench/reports/teleport_bench-"))
        self.assertTrue((ROOT / table["measured"]["report"]).is_file())
        self.assertEqual(set(table["links"]), {"loopback", "tailnet-hop", "tailnet-relay"})


class LoopbackApi(TempCase):
    def test_endpoints_must_be_loopback(self):
        for url in ("http://10.0.0.1:18309", "https://127.0.0.1:18309", "http://127.0.0.1"):
            with self.assertRaises(teleport.TeleportError) as e:
                teleport.Endpoint(url, "t", "m").connection(1)
            self.assertEqual(e.exception.code, "not_loopback")

    def test_export_body_shapes(self):
        self.assertEqual(teleport.export_body("m", "own", token_ids=[1, 2]),
                         {"tokens": [1, 2], "model": "m", "destination_tier": "own"})
        body = teleport.export_body("m", "team", messages={"messages": [], "tools": [], "stream": True})
        self.assertEqual(body, {"messages": [], "tools": [], "model": "m", "destination_tier": "team"})
        with self.assertRaises(teleport.TeleportError):
            teleport.export_body("m", "own")
        with self.assertRaises(teleport.TeleportError):
            teleport.export_body("m", "own", messages="text")

    def test_export_to_file_then_import_file_round_trip_with_the_key_header(self):
        blob = sealed_blob()
        key = teleport.load_key(self.key_file)
        out = self.tmp / "s.cstb"
        with FakeInstall(blob) as src, FakeInstall() as dst:
            meta = teleport.export_to_file(teleport.Endpoint(src.url, "ta", "m"), key,
                                           teleport.export_body("m", "own", token_ids=[5]), out)
            self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)
            self.assertEqual(out.read_bytes(), blob)
            self.assertEqual((meta["blob_bytes"], meta["tokens"], meta["image_bytes"]), (len(blob), 1000, 300000))
            got = teleport.import_file(teleport.Endpoint(dst.url, "tb", "my model"), key, out)
        self.assertEqual(dst.received, blob)
        self.assertEqual(got["imported_tokens"], 1000)
        self.assertEqual(src.requests[0]["key"], key)
        self.assertEqual(src.requests[0]["auth"], "Bearer ta")
        self.assertEqual(dst.requests[0]["path"], "/v1/cere/state/import?model=my%20model")
        self.assertEqual(teleport.read_header(out)["tokens"], 1000)

    def test_a_refused_export_leaves_no_file(self):
        out = self.tmp / "s.cstb"
        with FakeInstall(export_refusal=(404, "state_no_prefix")) as src:
            with self.assertRaises(teleport.Refused) as e:
                teleport.export_to_file(teleport.Endpoint(src.url, "t", "m"), "0" * 64,
                                        teleport.export_body("m", "own", token_ids=[1]), out)
        self.assertEqual((e.exception.op, e.exception.status, e.exception.code), ("export", 404, "state_no_prefix"))
        self.assertFalse(out.exists())

    def test_import_file_refuses_a_non_blob_before_sending(self):
        path = self.tmp / "x.cstb"
        path.write_bytes(b"CERESNP1" + b"\0" * 64)
        with FakeInstall() as dst:
            with self.assertRaises(teleport.TeleportError):
                teleport.import_file(teleport.Endpoint(dst.url, "t", "m"), "0" * 64, path)
        self.assertEqual(dst.requests, [])

    def test_move_direct_streams_the_export_into_the_import(self):
        blob = sealed_blob(image_bytes=9 << 20)
        with FakeInstall(blob) as src, FakeInstall() as dst:
            code, out = self.run_tool("move", "--source", src.url, "--dest", dst.url, "--source-token-env", "TOK_A",
                                      "--dest-token-env", "TOK_B", "--model", "m", "--messages", str(self.messages),
                                      "--dest-tier", "own", "--prompt-tokens", "18000", "--table", str(self.table))
        self.assertEqual(code, 0)
        self.assertEqual((out["mode"], out["decision"]["mode"], out["decision"]["link"]), ("state", "state", "loopback"))
        self.assertEqual(dst.received, blob)
        self.assertEqual(dst.requests[0]["length"], len(blob))
        self.assertEqual(src.requests[0]["body"]["destination_tier"], "own")
        self.assertEqual(src.requests[0]["auth"], "Bearer token-a")
        self.assertEqual(dst.requests[0]["auth"], "Bearer token-b")
        self.assertNotIn(teleport.load_key(self.key_file), json.dumps(out))

    def test_move_answers_replay_when_the_rule_says_so_without_touching_either_install(self):
        with FakeInstall(sealed_blob()) as src, FakeInstall() as dst:
            code, out = self.run_tool("move", "--source", src.url, "--dest", dst.url, "--source-token-env", "TOK_A",
                                      "--dest-token-env", "TOK_B", "--model", "m", "--messages", str(self.messages),
                                      "--dest-tier", "own", "--prompt-tokens", "18000", "--table", str(self.table),
                                      "--via", "vault", "--vault", str(self.tmp / "vault"))
        self.assertEqual(code, 0)
        self.assertEqual((out["mode"], out["reason"], out["decision"]["link"]),
                         ("replay", "reprefill_faster", "tailnet-relay"))
        self.assertEqual(src.requests + dst.requests, [])

    def test_move_needs_a_token_count_for_the_rule(self):
        with FakeInstall() as src:
            code, out = self.run_tool("move", "--source", src.url, "--dest", src.url, "--source-token-env", "TOK_A",
                                      "--dest-token-env", "TOK_B", "--model", "m", "--messages", str(self.messages),
                                      "--dest-tier", "own", "--table", str(self.table))
        self.assertEqual((code, out["code"]), (2, "usage"))

    def test_move_completes_as_replay_naming_the_refusal(self):
        """D-T1: a refusal by either install answers replay with its code."""
        for src_kw, dst_kw, reason, by in (({"export_refusal": (409, "state_busy")}, {}, "state_busy", "export"),
                                           ({}, {"import_refusal": (409, "state_compat_mismatch")},
                                            "state_compat_mismatch", "import")):
            with FakeInstall(sealed_blob(), **src_kw) as src, FakeInstall(**dst_kw) as dst:
                code, out = self.run_tool("move", "--source", src.url, "--dest", dst.url, "--source-token-env",
                                          "TOK_A", "--dest-token-env", "TOK_B", "--model", "m", "--messages",
                                          str(self.messages), "--dest-tier", "own", "--force-state")
            self.assertEqual(code, 0)
            self.assertEqual((out["mode"], out["reason"], out["refused_by"]), ("replay", reason, by))

    def test_move_through_a_local_vault_leaves_nothing_behind(self):
        blob = sealed_blob()
        vault = self.tmp / "vault"
        with FakeInstall(blob) as src, FakeInstall() as dst:
            code, out = self.run_tool("move", "--source", src.url, "--dest", dst.url, "--source-token-env", "TOK_A",
                                      "--dest-token-env", "TOK_B", "--model", "m", "--messages", str(self.messages),
                                      "--dest-tier", "own", "--force-state", "--via", "vault", "--vault", str(vault))
        self.assertEqual((code, out["mode"]), (0, "state"))
        self.assertEqual(dst.received, blob)
        self.assertEqual(list(vault.iterdir()), [])
        self.assertEqual(list(self.spool.iterdir()), [])

    def test_export_to_vault_then_import_from_vault(self):
        blob = sealed_blob()
        vault = self.tmp / "vault"
        with FakeInstall(blob) as src, FakeInstall() as dst:
            code, out = self.run_tool("export", "--source", src.url, "--token-env", "TOK_A", "--model", "m",
                                      "--messages", str(self.messages), "--dest-tier", "own", "--to-vault", str(vault))
            self.assertEqual(code, 0)
            name = out["name"]
            self.assertEqual(stat.S_IMODE((vault / f"{name}.cstb").stat().st_mode), 0o600)
            self.assertEqual(list(self.spool.iterdir()), [])
            code, out = self.run_tool("import", "--dest", dst.url, "--token-env", "TOK_B", "--model", "m",
                                      "--dest-tier", "own", "--from-vault", str(vault), "--name", name)
        self.assertEqual(code, 0)
        self.assertEqual(out["import"]["imported_tokens"], 1000)
        self.assertEqual(dst.received, blob)
        self.assertEqual(list(vault.iterdir()), [])

    def test_an_unset_bearer_env_is_a_usage_error(self):
        with FakeInstall(sealed_blob()) as src:
            code, out = self.run_tool("export", "--source", src.url, "--token-env", "TOK_NONE", "--model", "m",
                                      "--messages", str(self.messages), "--dest-tier", "own",
                                      "--out", str(self.tmp / "o"))
        self.assertEqual((code, out["code"]), (2, "usage"))
        self.assertEqual(src.requests, [])


class Tunnel(unittest.TestCase):
    def test_tunnel_argv_forwards_loopback_only_and_fails_closed(self):
        argv = teleport.tunnel_argv("gpu-box", 40000, 18309)
        self.assertEqual(argv[:2], ["ssh", "-N"])
        self.assertIn("ExitOnForwardFailure=yes", argv)
        self.assertIn("BatchMode=yes", argv)
        self.assertEqual(argv[-3:], ["-L", "127.0.0.1:40000:127.0.0.1:18309", "gpu-box"])

    def test_no_host_is_the_url_itself_and_a_dead_tunnel_is_a_transfer_failure(self):
        with teleport.tunnel("http://127.0.0.1:1", None) as url:
            self.assertEqual(url, "http://127.0.0.1:1")

        def dead(argv, **kw):
            return subprocess.Popen([sys.executable, "-c", "raise SystemExit(255)"], **kw)

        with self.assertRaises(teleport.TeleportError) as e:
            with teleport.tunnel("http://127.0.0.1:18309", "host", popen=dead):
                pass
        self.assertEqual(e.exception.code, "transfer_failed")
        with self.assertRaises(teleport.TeleportError):
            with teleport.tunnel("http://10.1.1.1:18309", "host", popen=dead):
                pass


def as_local(argv, input=None, capture_output=True, **kw):
    """A fake ssh/scp runner: ssh runs the remote command here with `sh -c`, scp copies with the `host:` dropped."""
    if argv[0] == "ssh":
        return subprocess.run(["sh", "-c", argv[-1]], input=input, capture_output=True)
    if argv[0] == "scp":
        src, dst = (a.split(":", 1)[1] if ":" in a else a for a in argv[-2:])
        shutil.copyfile(src, dst)
        return subprocess.CompletedProcess(argv, 0, b"", b"")
    raise AssertionError(argv)


class Vault(TempCase):
    def test_parse(self):
        self.assertEqual(tv.Vault.parse("node-b-wsl:/var/lib/cere/v/"), tv.Vault("node-b-wsl", "/var/lib/cere/v"))
        self.assertEqual(tv.Vault.parse("/tmp/v"), tv.Vault(None, "/tmp/v"))
        for bad in ("host:relative", ":/x", "/a b", "host:/a/../b", "/x/.."):
            with self.assertRaises(tv.VaultError):
                tv.Vault.parse(bad)
        for bad in ("../x", "", "a" * 65, "a.b"):
            with self.assertRaises(tv.VaultError):
                tv.check_name(bad)
        self.assertRegex(tv.new_name(), r"^[0-9a-f]{32}$")

    def test_scan_passes_a_sealed_blob(self):
        path = self.tmp / "a.cstb"
        path.write_bytes(sealed_blob(image_bytes=(9 << 20) + 5))
        got = tv.scan_file(path)
        self.assertTrue(got["sealed"], got)
        self.assertEqual((got["image_magic"], got["token_id_runs"], got["entropy_windows"]), (0, 0, 1))
        self.assertGreater(got["entropy_min"], 7.99)

    def test_scan_finds_plaintext(self):
        blob = sealed_blob(image_bytes=3 << 20)
        cases = {
            # The image magic anywhere, also across the 16 MiB read boundary.
            "magic": blob[:2000] + b"CERESNP1" + blob[2008:],
            "token_run": blob[:5000] + struct.pack("<20I", *range(100_000, 100_020)) + blob[5080:],
            "low_entropy": blob[:200] + b"\x00" * (len(blob) - 200),
            "size": blob[:-1],
            "not_a_blob": b"CERESNP1" + blob[8:],
        }
        for name, data in cases.items():
            path = self.tmp / f"{name}.cstb"
            path.write_bytes(data)
            self.assertFalse(tv.scan_file(path)["sealed"], name)
        across = bytearray(os.urandom(tv.SCAN_CHUNK + 4096))
        across[tv.SCAN_CHUNK - 4:tv.SCAN_CHUNK + 4] = b"CERESNP1"
        path = self.tmp / "across.cstb"
        path.write_bytes(bytes(across))
        self.assertEqual(tv.scan_file(path)["image_magic"], 1)
        run = bytearray(os.urandom(tv.SCAN_CHUNK + 4096))
        run[tv.SCAN_CHUNK - 30:tv.SCAN_CHUNK + 50] = struct.pack("<20I", *range(7, 27))
        path.write_bytes(bytes(run))
        self.assertEqual(tv.scan_file(path)["token_id_runs"], 1)

    def test_remote_vault_runs_this_file_over_ssh_stdin(self):
        """put/get/list/scan/remove through the ssh and scp argv, executed here (the remote is this host)."""
        root = self.tmp / "remote"
        vault = tv.Vault("vault-host", str(root))
        local = self.tmp / "s.cstb"
        local.write_bytes(sealed_blob())
        put = vault.put(local, "abc", run=as_local)
        self.assertEqual(put["bytes"], local.stat().st_size)
        self.assertEqual(stat.S_IMODE((root / "abc.cstb").stat().st_mode), 0o600)
        self.assertEqual([e["name"] for e in vault.list(run=as_local)], ["abc"])
        scan = vault.scan(["abc"], run=as_local)
        self.assertTrue(scan["pass"], scan)
        back = self.tmp / "back.cstb"
        vault.get("abc", back, run=as_local)
        self.assertEqual(back.read_bytes(), local.read_bytes())
        self.assertEqual(stat.S_IMODE(back.stat().st_mode), 0o600)
        self.assertFalse(vault.scan(["absent"], run=as_local)["pass"])
        vault.remove_dir(run=as_local)
        self.assertFalse(root.exists())

    def test_a_failed_ssh_is_a_vault_error(self):
        def fail(argv, **kw):
            return subprocess.CompletedProcess(argv, 255, b"", b"no route")

        with self.assertRaisesRegex(tv.VaultError, "no route"):
            tv.Vault("h", "/x").list(run=fail)

    def test_main_lists_scans_and_removes(self):
        (self.tmp / "v").mkdir()
        (self.tmp / "v" / "n1.cstb").write_bytes(sealed_blob())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(tv.main(["--scan", str(self.tmp / "v")]), 0)
            self.assertEqual(tv.main(["--rm", str(self.tmp / "v"), "n1"]), 0)
            self.assertEqual(tv.main(["--list", str(self.tmp / "v")]), 0)
        scan, removed, listed = (json.loads(line) for line in out.getvalue().splitlines())
        self.assertTrue(scan["pass"])
        self.assertEqual((removed, listed), (["n1"], []))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tv.main(["--rm", str(self.tmp / "v"), "../x"]), 2)
            self.assertEqual(tv.main(["--scan", "relative"]), 2)

    def test_probe_link_through_a_local_vault(self):
        got = teleport.cmd_probe_link(argparse.Namespace(vault=str(self.tmp / "v"), mib=2), {})
        self.assertEqual(got["bytes"], 2 << 20)
        self.assertGreater(got["relay_mb_per_s"], 0)
        self.assertEqual(list((self.tmp / "v").iterdir()), [])
        self.assertEqual(list(self.spool.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
