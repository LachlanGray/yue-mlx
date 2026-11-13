"""Queue lifecycle tests with controlled, real subprocesses; no model inference."""
import subprocess
import sys
import time

from yue2_mlx.web import Studio


def until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    assert predicate()


def setup(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'manifest.json').write_text('{}')
    launched = []
    real_popen = subprocess.Popen

    def launch(cmd, **kwargs):
        assert all(p.poll() is not None for _, p in launched), 'Overlapping inference'
        style = cmd[cmd.index('--style') + 1]
        if style == 'launch-error':
            raise OSError('test launch failure')
        process = real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        launched.append((style, process))
        return process

    monkeypatch.setattr('yue2_mlx.web.subprocess.Popen', launch)
    return Studio(tmp_path / 'sessions', model, model), launched


def request(style):
    return dict(kind='generate', style=style, lyrics='', seed=123)


def test_fifo_failure_cancel_and_saved_settings(tmp_path, monkeypatch):
    studio, launched = setup(tmp_path, monkeypatch)
    try:
        first = studio.start(request('first'))
        bad = studio.start(request('launch-error'))
        second = studio.start(request('second'))
        removed = studio.start(request('removed'))
        third = studio.start(request('third'))
        studio.cancel(removed['id'])
        positions = {j['title']: j['queue_position'] for j in studio.snapshot()['jobs']}
        assert positions['second'] == 2 and positions['third'] == 3
        assert len(launched) == 1
        # Unexpected nonzero exit also advances the queue, skipping launch errors.
        launched[0][1].terminate()
        until(lambda: len(launched) == 2)
        assert studio.jobs[first['id']]['status'] == 'failed'
        assert studio.jobs[bad['id']]['status'] == 'failed'
        assert launched[1][0] == 'second'
        assert studio.jobs[second['id']]['settings']['style'] == 'second'
        studio.cancel(second['id'])
        until(lambda: len(launched) == 3)
        assert launched[2][0] == 'third'
        studio.cancel(third['id'])
        until(lambda: studio.active is None)
        assert studio.snapshot()['queued'] == 0
    finally:
        studio.close()


def test_shutdown_retains_queue_and_explicit_resume(tmp_path, monkeypatch):
    studio, launched = setup(tmp_path, monkeypatch)
    studio.start(request('first'))
    second = studio.start(request('second'))
    studio.close()
    until(lambda: studio.active is None)
    assert len(launched) == 1
    restored = Studio(studio.root, studio.model, studio.vae)
    try:
        assert len(launched) == 1  # Constructing a second launcher cannot run jobs.
        assert restored.jobs[second['id']]['status'] == 'queued'
        restored.resume_queue()
        assert launched[-1][0] == 'second'
        restored.cancel(second['id'])
        until(lambda: restored.active is None)
    finally:
        restored.close()


def test_success_advances_queue(tmp_path, monkeypatch):
    original = subprocess.Popen
    studio, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr('yue2_mlx.web.subprocess.Popen', lambda cmd, **kwargs: original(
        [sys.executable, '-c', 'pass'], **kwargs))
    try:
        with studio.lock:
            first = studio.start(request('first'))
            second = studio.start(request('second'))
            assert second['status'] == 'queued'
        until(lambda: studio.jobs[second['id']]['status'] == 'done')
        assert studio.jobs[first['id']]['status'] == 'done'
    finally:
        studio.close()
