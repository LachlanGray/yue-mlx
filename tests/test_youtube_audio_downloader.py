import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from youtube_audio_downloader import DownloadQueue, make_handler, ThreadingHTTPServer, valid_youtube_url


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    assert predicate()


@pytest.fixture
def fake_importer(tmp_path):
    script = tmp_path / "importer.py"
    script.write_text(
        """import argparse, json, pathlib, time
p=argparse.ArgumentParser(); p.add_argument('--url'); p.add_argument('--output-dir'); a=p.parse_args()
print(json.dumps({'event':'progress','message':'Downloading test audio'}), flush=True)
time.sleep(.04)
name=a.url.rsplit('=',1)[-1]
path=pathlib.Path(a.output_dir)/(name+'.wav')
print(json.dumps({'ok':True,'output_path':str(path),'import_name':name}), flush=True)
"""
    )
    return script


def test_youtube_url_validation():
    assert valid_youtube_url("https://www.youtube.com/watch?v=abc")
    assert valid_youtube_url("https://youtu.be/abc")
    assert not valid_youtube_url("https://youtube.com.example.org/watch?v=abc")
    assert not valid_youtube_url("file:///tmp/video")


def test_fifo_queue_and_title_result(tmp_path, fake_importer):
    queue = DownloadQueue(fake_importer)
    try:
        jobs = queue.add(
            ["https://youtube.com/watch?v=first", "https://youtube.com/watch?v=second"],
            str(tmp_path / "audio"),
        )
        wait_until(lambda: all(queue.jobs[item["id"]]["status"] == "done" for item in jobs))
        assert queue.jobs[jobs[0]["id"]]["title"] == "first"
        assert queue.jobs[jobs[1]["id"]]["output_path"].endswith("second.wav")
        assert queue.active is None
    finally:
        queue.close()


def test_http_api_is_separate_and_rejects_bad_urls(tmp_path, fake_importer):
    queue = DownloadQueue(fake_importer)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(queue))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urlopen(base + "/api/state"))["app"] == "youtube-audio-downloader"
        request = Request(
            base + "/api/jobs",
            data=json.dumps({"urls": ["https://example.com/video"], "output_dir": str(tmp_path)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 400
    finally:
        queue.close()
        server.shutdown()
        server.server_close()
