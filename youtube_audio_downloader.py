#!/usr/bin/env python3
"""Small, standalone browser UI for the YouTube audio import script."""

import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import errno
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
from urllib.parse import urlparse
from urllib.request import urlopen
import uuid


YOUTUBE_HOSTS = {"youtube.com", "youtu.be", "youtube-nocookie.com"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def valid_youtube_url(value):
    if not isinstance(value, str) or len(value) > 4096:
        return False
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https") and any(
        host == allowed or host.endswith("." + allowed) for allowed in YOUTUBE_HOSTS
    )


class DownloadQueue:
    """FIFO queue with one worker and no dependency on the YuE web application."""

    def __init__(self, importer=None):
        self.importer = Path(importer or Path(__file__).with_name("import_youtube_audio.py")).resolve()
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.jobs = {}
        self.pending = deque()
        self.active = None
        self.process = None
        self.closing = False
        self.worker = threading.Thread(target=self._work, daemon=True, name="youtube-audio-queue")
        self.worker.start()

    def add(self, urls, output_dir):
        if not isinstance(urls, list) or not urls or len(urls) > 100:
            raise ValueError("Add between 1 and 100 YouTube URLs.")
        urls = [value.strip() if isinstance(value, str) else value for value in urls]
        if any(not valid_youtube_url(value) for value in urls):
            raise ValueError("Every entry must be a valid YouTube URL.")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("Choose a download directory.")
        directory = Path(output_dir.strip()).expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            raise ValueError("The download destination is not a directory.")

        added = []
        with self.changed:
            if self.closing:
                raise ValueError("The downloader is shutting down.")
            for url in urls:
                identifier = uuid.uuid4().hex
                job = {
                    "id": identifier,
                    "url": url,
                    "output_dir": str(directory),
                    "status": "queued",
                    "progress": "Waiting in queue",
                    "created": utc_now(),
                }
                self.jobs[identifier] = job
                self.pending.append(identifier)
                added.append(dict(job))
            self.changed.notify()
        return added

    def snapshot(self):
        with self.lock:
            positions = {identifier: index + 1 for index, identifier in enumerate(self.pending)}
            jobs = sorted(self.jobs.values(), key=lambda job: job["created"], reverse=True)
            return {
                "app": "youtube-audio-downloader",
                "busy": self.active is not None,
                "queued": len(self.pending),
                "jobs": [dict(job, queue_position=positions.get(job["id"])) for job in jobs],
            }

    def _work(self):
        while True:
            with self.changed:
                while not self.pending and not self.closing:
                    self.changed.wait()
                if self.closing:
                    return
                identifier = self.pending.popleft()
                job = self.jobs[identifier]
                job.update(status="running", progress="Starting download…", started=utc_now())
                self.active = identifier
            self._run(job)
            with self.lock:
                self.active = None

    def _run(self, job):
        command = [
            sys.executable,
            "-u",
            str(self.importer),
            "--url",
            job["url"],
            "--output-dir",
            job["output_dir"],
        ]
        last_lines = []
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            with self.lock:
                self.process = process
            for raw in process.stdout:
                line = raw.strip()
                if not line:
                    continue
                last_lines.append(line)
                last_lines = last_lines[-5:]
                try:
                    event = json.loads(line)
                except ValueError:
                    event = {}
                with self.lock:
                    if event.get("event") == "progress":
                        job["progress"] = event.get("message", "Downloading…")
                    if event.get("ok") is True:
                        job["output_path"] = event.get("output_path")
                        job["title"] = event.get("import_name")
                    elif event.get("ok") is False:
                        job["error"] = event.get("error", "Download failed.")
            code = process.wait()
            process.stdout.close()
            with self.lock:
                self.process = None
                if code == 0 and job.get("output_path"):
                    job.update(status="done", progress="Download complete")
                else:
                    message = job.get("error") or "\n".join(last_lines) or f"Downloader exited with code {code}."
                    job.update(status="failed", progress=message[-4000:])
                job["finished"] = utc_now()
        except OSError as exc:
            with self.lock:
                self.process = None
                job.update(status="failed", progress=f"Could not start downloader: {exc}", finished=utc_now())

    def close(self):
        with self.changed:
            self.closing = True
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self.changed.notify_all()


def make_handler(queue):
    page = Path(__file__).with_name("youtube_audio_downloader.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _local(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed:
                raise ValueError("Use the local downloader address.")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + item for item in allowed}:
                raise ValueError("Cross-origin requests are not allowed.")

        def _json(self, value, status=200):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            try:
                self._local()
                if urlparse(self.path).path == "/api/state":
                    return self._json(queue.snapshot())
                if urlparse(self.path).path == "/":
                    payload = page.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self._json({"error": "Not found"}, 404)
            except (ValueError, OSError) as exc:
                self._json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                self._local()
                if urlparse(self.path).path != "/api/jobs":
                    return self._json({"error": "Not found"}, 404)
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 512_000:
                    raise ValueError("Request body is empty or too large.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                self._json({"jobs": queue.add(data.get("urls"), data.get("output_dir"))}, 201)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Standalone YouTube audio download queue")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the downloader in your browser")
    args = parser.parse_args()
    queue = DownloadQueue()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(queue))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE and args.open:
            try:
                with urlopen(f"http://127.0.0.1:{args.port}/api/state", timeout=2) as response:
                    state = json.load(response)
                if state.get("app") == "youtube-audio-downloader":
                    import webbrowser
                    webbrowser.open(f"http://127.0.0.1:{args.port}")
                    print("YouTube Audio Downloader is already running. Opened it in your browser.")
                    queue.close()
                    return
            except (OSError, ValueError):
                pass
        queue.close()
        parser.exit(1, f"Could not start the downloader on port {args.port}: {exc}\n")
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"YouTube Audio Downloader → {address}", flush=True)
    if args.open:
        import webbrowser
        threading.Timer(.4, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        queue.close()
        server.server_close()


if __name__ == "__main__":
    main()
