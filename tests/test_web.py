import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import soundfile as sf

from yue2_mlx.web import Studio, ThreadingHTTPServer, handler


@pytest.fixture
def server(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text("{}")
    studio = Studio(tmp_path / "sessions", model, model)
    http = ThreadingHTTPServer(("127.0.0.1", 0), handler(studio))
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield studio, f"http://127.0.0.1:{http.server_port}"
    studio.close()
    http.shutdown()
    http.server_close()


@pytest.mark.parametrize('kind',['generate','flow-edit'])
def test_acoustic_checkpoint_selection(tmp_path,monkeypatch,kind):
    model=tmp_path/'model';model.mkdir();(model/'manifest.json').write_text('{}')
    studio=Studio(tmp_path/'jobs',model,model,tokenizer_model=model)
    monkeypatch.setattr(studio,'launch_next',lambda:None)
    upload='a'*32;sf.write(studio.root/'uploads'/upload,np.zeros((4800,2)),48000,format='WAV')
    data=dict(kind=kind,style='piano',upload=upload,acoustic_choice='mothersuperior')
    job=studio.start(data)
    assert job['settings']['acoustic_choice']=='mothersuperior'
    cmd=job['command'];assert cmd[cmd.index('--acoustic-checkpoint')+1]==str(model)
    stock=studio.start(dict(data,acoustic_choice='stock'))
    assert '--acoustic-checkpoint' not in stock['command']
    with pytest.raises(ValueError):studio.start(dict(data,acoustic_choice='unknown'))


def send(url, data, headers=None):
    raw = data if isinstance(data, bytes) else json.dumps(data).encode()
    return urlopen(Request(url, data=raw, headers=headers or {"Content-Type": "application/json"}), timeout=5)


def test_page_validation_and_origin(server):
    studio, base = server
    assert b"YuE2 Studio" in urlopen(base).read()
    assert b"YuE2 Studio" in urlopen(base + "/history").read()
    assert json.load(urlopen(base + "/api/state"))["generation_ready"]
    with pytest.raises(HTTPError) as exc:
        send(base + "/api/jobs", {"kind": "generate", "style": "", "lyrics": "hi"})
    assert exc.value.code == 400
    with pytest.raises(HTTPError):
        send(base + "/api/jobs", {}, {"Origin": "https://unrelated.example"})
    with pytest.raises(HTTPError):
        urlopen(base + "/files/" + "a" * 32 + "/%2e%2e")


def test_audio_upload_and_bad_file(server):
    studio, base = server
    data = io.BytesIO()
    sf.write(data, np.zeros((4800, 2)), 48000, format="WAV")
    result = json.load(send(base + "/api/upload", data.getvalue()))
    assert result["seconds"] == .1
    assert (studio.root / "uploads" / result["id"]).is_file()
    with pytest.raises(HTTPError) as exc:
        send(base + "/api/upload", b"not audio")
    assert exc.value.code == 400
    assert len(list((studio.root / "uploads").iterdir())) == 1


def test_semantic_flow_request(server, monkeypatch):
    studio, base = server
    studio.tokenizer_model = studio.model
    monkeypatch.setattr(studio, 'launch_next', lambda: None)
    data = io.BytesIO()
    sf.write(data, np.zeros((4800, 2)), 48000, format='WAV')
    upload = json.load(send(base+'/api/upload', data.getvalue()))['id']
    request = dict(kind='semantic-audio', upload=upload, mode='anchored', start_time=.25,
                   noise_amount=.1, reference_upload=upload, reference_placement='audio',
                   reference_strength=1.5, anchor_strength=2, nar_adapter=True)
    job = json.load(send(base+'/api/jobs', request))
    assert job['settings']['noise_amount']==.1
    assert job['settings']['reference_upload']==upload
    assert job['settings']['mode']=='anchored'
    cmd=job['command']
    for flag,value in [('--noise-amount','0.1'),('--start-time','0.25'),('--reference-placement','audio')]:
        assert cmd[cmd.index(flag)+1]==value
    with pytest.raises(HTTPError):
        send(base+'/api/jobs',dict(request,noise_amount=1.1))


def test_single_job_cancel_and_history(server, monkeypatch):
    studio, base = server
    real_popen = subprocess.Popen
    monkeypatch.setattr("yue2_mlx.web.subprocess.Popen", lambda cmd, **kwargs: real_popen(
        [sys.executable, "-u", "-c", "import time; print('semantic: 1/200', flush=True); time.sleep(20)"], **kwargs))
    request = {"kind": "generate", "style": "piano", "lyrics": "hi", "seconds": 8}
    job = json.load(send(base + "/api/jobs", request))
    queued = json.load(send(base + "/api/jobs", request))
    assert queued['status'] == 'queued'
    assert studio.snapshot()['queued'] == 1
    assert json.load(send(base + '/api/cancel', {'id': queued['id']}))['ok']
    assert studio.jobs[queued['id']]['status'] == 'cancelled'
    assert json.load(send(base + "/api/cancel", {"id": job["id"]}))["ok"]
    deadline = time.monotonic() + 5
    while studio.active and time.monotonic() < deadline:
        time.sleep(.02)
    assert studio.jobs[job["id"]]["status"] == "cancelled"
    restored = Studio(studio.root, studio.model, studio.vae)
    assert restored.jobs[job["id"]]["status"] == "cancelled"


@pytest.mark.parametrize("lyrics", [None, "", "   "])
def test_style_only_generation(server, monkeypatch, lyrics):
    studio, base = server
    real_popen = subprocess.Popen
    commands = []

    def launch(cmd, **kwargs):
        commands.append(cmd)
        return real_popen([sys.executable, "-c", "pass"], **kwargs)

    monkeypatch.setattr("yue2_mlx.web.subprocess.Popen", launch)
    request = {"kind": "generate", "style": "instrumental piano"}
    if lyrics is not None:
        request["lyrics"] = lyrics
    job = json.load(send(base + "/api/jobs", request))
    assert (studio.root / job["id"] / "lyrics.txt").read_text() == ""
    cmd = commands[0]
    assert Path(cmd[cmd.index("--lyrics-file") + 1]).read_text() == ""


@pytest.mark.parametrize("kind", ["generate", "flow-edit"])
def test_no_artificial_caps(server, monkeypatch, kind):
    studio, base = server
    real_popen = subprocess.Popen
    monkeypatch.setattr("yue2_mlx.web.subprocess.Popen", lambda cmd, **kwargs: real_popen(
        [sys.executable, "-c", "pass"], **kwargs))
    request = dict(kind=kind, style="piano " * 25000, lyrics="", seconds=500, steps=96)
    if kind == "flow-edit":
        data = io.BytesIO()
        sf.write(data, np.zeros((601 * 1000, 2)), 1000, format="WAV")
        upload = json.load(send(base + "/api/upload", data.getvalue()))
        request.update(upload=upload['id'], reference_upload=upload['id'],
                       reference_placement="prefix", reference_strength=6,
                       mode="anchored", anchor_strength=25)
    job = json.load(send(base + "/api/jobs", request))
    assert job['settings']['steps'] == 96
    assert job['settings']['style'] == request['style'].strip()
    if kind == "flow-edit":
        assert job['settings']['reference_strength'] == 6
        assert job['settings']['anchor_strength'] == 25
    else:
        assert job['settings']['seconds'] == 500


def test_audio_ranges(server):
    studio, base = server
    output = studio.root / ("b" * 32) / "output"
    output.mkdir(parents=True)
    (output / "audio.wav").write_bytes(bytes(range(100)))
    response = urlopen(Request(base + "/files/" + "b" * 32 + "/audio.wav", headers={"Range": "bytes=10-19"}))
    assert response.status == 206
    assert response.read() == bytes(range(10, 20))
    assert response.headers["Content-Range"] == "bytes 10-19/100"


def test_flow_edit_validation_and_command(server, monkeypatch):
    studio, base = server
    audio = io.BytesIO()
    sf.write(audio, np.zeros((4800, 2)), 48000, format="WAV")
    uploaded = json.load(send(base + "/api/upload", audio.getvalue()))
    request = {"kind": "flow-edit", "upload": uploaded["id"], "start_time": .12, "mode": "direct", "steps": 3}
    for override in [{"start_time": -1}, {"start_time": float("nan")}, {"mode": "bad"}, {"steps": 0},
                     {"anchor_strength": -1}, {"anchor_strength": float("nan")}, {"independent_channels": "yes"},
                     {"reference_placement":"bad"}, {"reference_placement":"prefix"}, {"reference_strength":float("nan")}]:
        with pytest.raises(HTTPError) as exc:
            send(base + "/api/jobs", {**request, **override})
        assert exc.value.code == 400
    real_popen, commands = subprocess.Popen, []

    def launch(cmd, **kwargs):
        commands.append(cmd)
        return real_popen([sys.executable, "-c", "pass"], **kwargs)

    monkeypatch.setattr("yue2_mlx.web.subprocess.Popen", launch)
    request.update(mode="anchored", anchor_strength=2.5, independent_channels=True,
                   reference_upload=uploaded["id"], reference_placement="audio", reference_strength=.5)
    job = json.load(send(base + "/api/jobs", request))
    assert job["kind"] == "flow-edit"
    assert job["settings"]["start_time"] == .12
    assert job["settings"]["independent_channels"] is True
    assert job["settings"]["anchor_strength"] == 2.5
    saved = json.loads((studio.root / job["id"] / "job.json").read_text())
    assert saved["settings"] == job["settings"]
    restored = Studio(studio.root, studio.model, studio.vae)
    assert restored.jobs[job["id"]]["settings"] == job["settings"]
    cmd = commands[0]
    assert cmd[cmd.index("--model")+1] == str(studio.model)
    assert cmd[cmd.index("--start-time")+1] == "0.12"
    assert cmd[cmd.index("--mode")+1] == "anchored"
    assert cmd[cmd.index("--anchor-strength")+1] == "2.5"
    assert "--independent-channels" in cmd
    assert cmd[cmd.index("--reference")+1] == str(studio.root / 'uploads' / uploaded['id'])
    assert cmd[cmd.index("--reference-placement")+1] == 'audio'
    assert job['settings']['reference_upload'] == uploaded['id']


def test_recover_older_settings(tmp_path):
    root = tmp_path / "sessions"
    folder = root / ("a"*32)
    (folder / "output").mkdir(parents=True)
    job = {"id": "a"*32, "kind": "flow-edit", "status": "done", "title": "recording.wav"}
    (folder / "job.json").write_text(json.dumps(job))
    (folder / "output/result.json").write_text(json.dumps({"mode":"anchored", "start_time":.2,
        "steps":8, "seed":42, "style":"piano", "channel_mode":"independent"}))
    studio = Studio(root, tmp_path / "model", tmp_path / "vae")
    restored = studio.jobs[job["id"]]
    assert restored["settings_recovered"]
    assert restored["settings"]["mode"] == "anchored"
    assert restored["settings"]["independent_channels"] is True
    assert "anchor_strength" not in restored["settings"]
