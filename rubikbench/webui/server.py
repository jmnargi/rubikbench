"""Local web server that replays a RubikBench run file in the browser.

``rubikbench view results.jsonl`` starts the server, opens the page, and blocks
until interrupted. The run is embedded into the page as JSON; the 3D replay is
powered by Three.js loaded from a CDN (the cube still renders the data offline,
but the scene itself needs a network connection for the library).
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..aggregate import aggregate_files

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8321


def build_replay_document(run_path: str | Path) -> str:
    """Render the full HTML page with the run dataset embedded."""
    dataset = aggregate_files([run_path])
    index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    payload = json.dumps(dataset)
    if "/*__RUN_DATA__*/" not in index:
        raise RuntimeError("webui template is missing the run-data marker")
    return index.replace("/*__RUN_DATA__*/", payload)


class _Handler(BaseHTTPRequestHandler):
    server_version = "RubikBenchView/1.0"

    def log_message(self, *args: object) -> None:  # silence request logging
        pass

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("", "/index.html"):
            body = build_replay_document(self.server.run_path).encode()  # type: ignore[attr-defined]
            ctype = "text/html; charset=utf-8"
        elif self.path.split("?")[0] == "/app.js":
            body = (STATIC_DIR / "app.js").read_bytes()
            ctype = "text/javascript; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_run(run_path: str | Path, port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Serve the replay page until interrupted (Ctrl+C)."""
    run_path = Path(run_path)
    if not run_path.exists():
        raise FileNotFoundError(f"run file not found: {run_path}")
    # Validate up front so errors surface before the server starts.
    aggregate_files([run_path])

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.run_path = str(run_path)  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"RubikBench view: {run_path}")
    print(f"Open {url} (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001,S110 - headless environments have no browser
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
