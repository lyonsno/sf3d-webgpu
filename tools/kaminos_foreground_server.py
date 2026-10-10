"""Owned loopback adapter over the exact Kaminos serving implementation.

No copied endpoints. Stores are caller-owned; the ordinary handler and runtime
config are imported from the explicitly selected checkout.
"""
import importlib.util
import http.server
import json
import os
import re
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
if sys.argv[3:] not in ([], ["--sf3d-source-hold-head"]):
    raise ValueError("unknown selected-consumer adapter policy")
source_hold_head = sys.argv[3:] == ["--sf3d-source-hold-head"]
spec = importlib.util.spec_from_file_location("mini_actual_kaminos_server", root / "serve.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
arguments, module.SHARED_BASIN_STORE = module.split_shared_basin_store_arguments([
    "0", "--no-shared-basin-store",
    "--volume-settings-store", str(out / "settings"),
    "--volume-basin-session-store", str(out / "sessions"),
    "--volume-cockpit-layout-store", str(out / "layouts"),
], module.SHARED_BASIN_STORE_DEFAULT)
module.PORT, module.VOLUME_SETTINGS_STORE, module.VOLUME_BASIN_SESSION_STORE, module.VOLUME_COCKPIT_LAYOUT_STORE = module.parse_server_arguments(arguments)
class SelectedConsumerHandler(module.KaminosHandler):
    # Apply the held full-model/source boundary at the private server too, so
    # learning its ephemeral loopback port does not bypass the proxy refusal.
    def held(self):
        decoded = urlparse(self.path).path
        while unquote(decoded) != decoded:
            decoded = unquote(decoded)
        if re.search(r"\.(bin|glb|gltf|safetensors|npy|npz|pt|gguf)$", decoded, re.I) or re.match(r"/api/(read|delete|job)", decoded):
            return True
        return False

    def do_GET(self):
        if self.held():
            self.send_json({"error": "unadmitted model/data/mutation route held by selected consumer"}, 409)
            return
        super().do_GET()

    def do_HEAD(self):
        decoded = urlparse(self.path).path
        while unquote(decoded) != decoded:
            decoded = unquote(decoded)
        if source_hold_head and decoded == "/lib/sf3d/weights.bin":
            # Delegate only the refusal, never static file access. If this
            # source is not held, or the actual handler lacks this capability,
            # the selected consumer's original409 restriction still applies.
            refuse = getattr(self, "refuse_sf3d_source", None)
            if callable(refuse) and refuse(module.ROOT / "lib/sf3d/weights.bin"):
                return
        if self.held():
            self.send_response(409)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_HEAD()

    def do_POST(self):
        if urlparse(self.path).path == "/api/volume-cockpit-layouts":
            # The actual cockpit publishes its source layout during startup.
            # This store was explicitly routed into this run's output directory.
            super().do_POST()
            return
        self.send_json({"error": "selected consumer permits only caller-owned cockpit layout writes"}, 409)

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SelectedConsumerHandler)
module.PORT = server.server_address[1]
print(json.dumps({"origin": f"http://127.0.0.1:{module.PORT}", "pid": os.getpid(), "sourceHoldHead": source_hold_head}), flush=True)
try:
    server.serve_forever()
finally:
    server.server_close()
