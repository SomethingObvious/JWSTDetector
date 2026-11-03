#!/usr/bin/env python3
"""Opens a run in the web viewer, or exports it as a static site.

    python viewer.py results_miri datasets/miri
    python viewer.py results_miri datasets/miri --export site/

The first serves the viewer on localhost and saves verdicts to verdicts.csv in the run
folder. The second writes the viewer and the whole run to a folder that any static
host can serve, GitHub Pages included, where verdicts stay in each visitor's browser.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from jwstdetector.bundle import RunBundle, add_site_meta, export, save_verdict

logger = logging.getLogger("viewer")

APP_DIR = Path(__file__).resolve().parent / "jwstdetector" / "viewer_static"


def make_handler(bundle: RunBundle):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send(status, json.dumps({"error": message}).encode(), "application/json")

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
            if path.startswith("run/"):
                try:
                    body, ctype = bundle.read(path[4:])
                except (FileNotFoundError, OSError):
                    self._error(HTTPStatus.NOT_FOUND, f"There is nothing at {path}")
                    return
                static = not path.endswith(("index.json", "verdicts.json"))
                self._send(HTTPStatus.OK, body, ctype, cache=static)
                return

            target = (APP_DIR / path).resolve()
            if not target.is_relative_to(APP_DIR) or not target.is_file():
                self._error(HTTPStatus.NOT_FOUND, f"There is nothing at {path}")
                return
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, target.read_bytes(), ctype, cache=path != "index.html")

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/run/api/verdict":
                self._error(HTTPStatus.NOT_FOUND, "Only run/api/verdict takes writes")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(min(length, 1 << 20)))
                key = str(body["key"])
                if key not in {t.key for t in bundle.tiles}:
                    raise KeyError(f"{key} isn't a tile in this run")
                with lock:
                    saved = save_verdict(
                        bundle.run_dir,
                        key,
                        body.get("verdict"),
                        bool(body.get("star")),
                        str(body.get("note") or ""),
                    )
            except (ValueError, KeyError, TypeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send(HTTPStatus.OK, json.dumps({"saved": saved}).encode(), "application/json")

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="A run_query_bootstrap.py output folder.")
    ap.add_argument("data_root", type=Path, help="The prep_data.py --out folder it scored.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--no-browser", dest="browser", action="store_false", help="Don't open a browser tab."
    )
    ap.add_argument("--export", type=Path, default=None, help="Write a static site here instead.")
    ap.add_argument(
        "--cutouts",
        type=int,
        default=None,
        help="With --export, keep the FITS cutouts of only the top N tiles, as they're big.",
    )
    ap.add_argument("--name", default=None, help="The run's name in the viewer, its tag otherwise.")
    ap.add_argument(
        "--site_url",
        default=None,
        help="With --export, where the site will be hosted, for its social card and sitemap.",
    )
    ap.add_argument("--verbose", action="store_true", help="Log every request.")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not (APP_DIR / "index.html").is_file():
        raise SystemExit(
            f"The viewer isn't built, as {APP_DIR / 'index.html'} is missing. "
            "Run `npm ci && npm run build` in web/."
        )
    bundle = RunBundle(args.run_dir, args.data_root, writable=args.export is None)
    if args.name:
        bundle.index["run"]["name"] = args.name

    if args.export is not None:
        n = export(bundle, args.export, APP_DIR, cutouts=args.cutouts)
        if args.site_url:
            add_site_meta(bundle, args.export, args.site_url)
        logger.info("Wrote the viewer and %d run files to %s", n, args.export)
        return 0

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bundle))
    url = f"http://127.0.0.1:{server.server_port}/"
    logger.info("Serving %d tiles at %s, Ctrl+C to stop", len(bundle.tiles), url)
    if args.browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
