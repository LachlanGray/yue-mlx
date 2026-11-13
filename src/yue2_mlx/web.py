"""Local studio. Each job runs in its own process; only one may use Metal."""
import argparse
import errno
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
import uuid

ID = re.compile(r"[a-f0-9]{32}\Z")


class BusyError(Exception):
    pass


class Studio:
    def __init__(self, root, model, vae, memory=10, tokenizer_model="models/realaudio-v4"):
        self.root, self.model, self.vae = Path(root).resolve(), Path(model).resolve(), Path(vae).resolve()
        self.memory = memory
        self.tokenizer_model = Path(tokenizer_model).resolve()
        self.adapter_root = self.model.parent / "loras"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "uploads").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.active = None
        self.closing = False
        self.jobs = {}
        for file in self.root.glob("*/job.json"):
            try:
                job = json.loads(file.read_text())
                if job["status"] in ("running", "cancelling"):
                    job.update(status="interrupted", progress="Server stopped before this session finished.")
                if "settings" not in job:
                    job["settings"] = self.recover_settings(file.parent, job)
                    job["settings_recovered"] = True
                self.jobs[job["id"]] = job
            except (ValueError, KeyError):
                continue

    def tokenizers(self):
        result=[dict(id='mothersuperior',name='Mothersuperior v4',path=str(self.tokenizer_model))]
        for manifest in sorted(self.adapter_root.glob('*/mlx/manifest.json')):
            try:
                value=json.loads(manifest.read_text());folder=manifest.parent
                if value.get('format')!='yue2-realaudio-v4':continue
                if not all((folder/n).is_file() for n in ('tokenizer_head_joint_v4.safetensors','nar_lora_joint_v4.safetensors')):continue
                result.append(dict(id='bundle:'+manifest.parent.parent.name,
                                   name=value.get('name',manifest.parent.parent.name),path=str(folder),
                                   ar=bool(value.get('ar_adapter') and (folder/value['ar_adapter']).is_file())))
            except (OSError,ValueError,KeyError):continue
        with self.lock:
            jobs=list(self.jobs.values())
        for job in jobs:
            if job.get('kind')!='train':continue
            for manifest in sorted((self.root/job['id']/'output/checkpoints').glob('step-*/manifest.json')):
                try:
                    value=json.loads(manifest.read_text())
                    if value.get('format')!='yue2-realaudio-v4':continue
                    folder=manifest.parent
                    if not all((folder/n).is_file() for n in ('tokenizer_head_joint_v4.safetensors','nar_lora_joint_v4.safetensors')):continue
                    result.append(dict(id=job['id']+':'+folder.name,name=value.get('name',folder.name),path=str(folder)))
                except (OSError,ValueError):continue
        return result

    def ar_adapters(self):
        result=[dict(id=item['id'],name=item['name'],path=item['path']) for item in self.tokenizers() if item.get('ar')]
        seen={item['path'] for item in result}
        for manifest in sorted(self.adapter_root.glob('*/mlx/manifest.json')):
            try:
                value=json.loads(manifest.read_text());folder=manifest.parent
                if value.get('format')!='yue2-ar-adapter' or not (folder/value.get('ar_adapter','ar_lora.safetensors')).is_file():continue
                path=str(folder)
                if path not in seen:
                    result.append(dict(id='ar:'+manifest.parent.parent.name,
                                       name=value.get('name',manifest.parent.parent.name),path=path));seen.add(path)
            except (OSError,ValueError,KeyError):continue
        return result

    def ar_path(self, choice):
        for item in self.ar_adapters():
            if item['id']==choice:return item['path']
        raise ValueError('AR checkpoint is unavailable; choose another checkpoint')

    def tokenizer_path(self, choice):
        for item in self.tokenizers():
            if item['id']==choice:return item['path']
        raise ValueError('Tokenizer checkpoint is unavailable; choose another checkpoint')

    def training_songs(self):
        file=self.root/'training-songs.json'
        return json.loads(file.read_text()) if file.exists() else []

    def set_training_songs(self, songs):
        if not isinstance(songs,list):raise ValueError('Expected a song list')
        clean=[];seen=set()
        for song in songs:
            identifier=song.get('id','')
            if not isinstance(identifier,str) or not ID.fullmatch(identifier) or not (self.root/'uploads'/identifier).is_file():
                raise ValueError('Uploaded song was not found')
            if identifier not in seen:
                clean.append(dict(id=identifier,name=str(song.get('name','Recording'))));seen.add(identifier)
        with self.lock:
            temp=self.root/'training-songs.tmp';temp.write_text(json.dumps(clean,indent=2));temp.replace(self.root/'training-songs.json')
        return clean

    def start_training(self, data):
        from dataclasses import asdict
        from .training_config import Settings
        if data.get('resume_job'):
            return self.resume_training(data['resume_job'])
        parameters=asdict(Settings.parse(data.get('parameters',{})))
        choice=data.get('tokenizer_choice','mothersuperior');source=self.tokenizer_path(choice)
        if not (Path(source)/'manifest.json').is_file() or not (self.model/'manifest.json').is_file():
            raise ValueError('Install generation and tokenizer models first')
        name=str(data.get('name','My tokenizer')).strip()
        if not name:raise ValueError('Name this training run')
        # Snapshot uploads now: removing songs later never changes a queued run.
        collection=self.training_songs()
        if not collection:raise ValueError('Upload at least one song first')
        songs=[dict(path=str(self.root/'uploads'/s['id']),**s) for s in collection]
        if any(not Path(s['path']).is_file() for s in songs):raise ValueError('A training upload is missing')
        settings=dict(kind='train',name=name,parameters=parameters,songs=songs,tokenizer_choice=choice,
                      tokenizer_model=source,model=str(self.model),memory_gib=self.memory,
                      cache=str(self.root/'training-cache'))
        return self.queue_training(settings)

    def training_progress(self, identifier):
        with self.lock:
            job=self.jobs.get(identifier)
            if not job or job.get('kind')!='train':raise ValueError('Training run not found')
        output=self.root/identifier/'output';rows=[];checkpoints=[]
        if (output/'metrics.jsonl').exists():
            for line in (output/'metrics.jsonl').read_text().splitlines():
                try:rows.append(json.loads(line))
                except ValueError:pass  # Writer may be halfway through the latest line.
        for path in sorted((output/'checkpoints').glob('step-*/manifest.json')):
            try:
                manifest=json.loads(path.read_text());training=manifest.get('training',{})
                if training.get('optimizer_resume') and all((path.parent/n).is_file() for n in ('resume.json','optimizer.safetensors','tokenizer_head_joint_v4.safetensors','nar_lora_joint_v4.safetensors')):
                    checkpoints.append((path.parent,training))
            except (OSError,ValueError):pass
        latest=checkpoints[-1] if checkpoints else None
        resumable=bool(latest and latest[1]['step']<latest[1]['settings']['steps'] and job['status'] not in ('running','queued','cancelling'))
        return dict(metrics=rows,resumable=resumable,step=latest[1]['step'] if latest else None,
                    checkpoint=str(latest[0]) if latest else None)

    def resume_training(self, identifier):
        with self.lock:
            progress=self.training_progress(identifier)
            if not progress['resumable']:raise ValueError('No unfinished resumable checkpoint; older checkpoints support warm starts only')
            if any(j.get('settings',{}).get('resume_job')==identifier and j['status'] in ('queued','running','cancelling') for j in self.jobs.values()):
                raise ValueError('This run already has a queued or running continuation')
            settings=dict(self.jobs[identifier]['settings'])
            if settings['model']!=str(self.model):raise ValueError('Resume requires the original base model')
            if any(not Path(s['path']).is_file() for s in settings['songs']):raise ValueError('An original training upload is missing')
            settings.update(resume_checkpoint=progress['checkpoint'],resume_job=identifier)
            return self.queue_training(settings)

    def queue_training(self, settings):
        with self.lock:
            if self.closing:raise BusyError('The server is shutting down')
            identifier=uuid.uuid4().hex;folder=self.root/identifier;folder.mkdir()
            (folder/'training.json').write_text(json.dumps(settings,indent=2))
            cmd=[sys.executable,'-u','-m','yue2_mlx.training','--config',str(folder/'training.json'),
                 '--output',str(folder/'output'),'--model',settings['model'],'--memory-gib',str(self.memory)]
            job=dict(id=identifier,kind='train',title=settings['name'],status='queued',created=datetime.now(timezone.utc).isoformat(),
                     progress='Waiting in queue',files=[],settings=settings,command=cmd)
            self.jobs[identifier]=job;self.save(job);self.launch_next();return dict(job)

    def resume_queue(self):
        # Only after binding the server: a second launcher must not run jobs.
        with self.lock:
            self.launch_next()

    def save(self, job):
        folder = self.root / job["id"]
        temp = folder / "job.tmp"
        temp.write_text(json.dumps(job, indent=2))
        temp.replace(folder / "job.json")

    def recover_settings(self, folder, job):
        """Recover only evidenced settings from older artifacts; never invent values."""
        def read(name):
            try:
                value = json.loads((folder / name).read_text())
                return value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                return {}
        result, plan = read("output/result.json"), read("output/plan.json")
        settings = {"kind": job["kind"]}
        for key in ("style", "lyrics", "seed", "steps", "mode", "start_time", "anchor_strength",
                    "reference_placement", "reference_strength"):
            if key in result:
                settings[key] = result[key]
        if job["kind"] == "generate":
            for key in ("style", "lyrics", "seed", "cot"):
                if key in plan:
                    settings[key] = plan[key]
            sampling = read("output/semantic.json").get("sampling", {})
            if "max_tokens" in sampling:
                settings["seconds"] = sampling["max_tokens"] / 25
            settings["abc"] = (folder / "score-input.abc").read_text() if (folder / "score-input.abc").is_file() else ""
        else:
            settings["name"] = job["title"]
            if "channel_mode" in result:
                settings["independent_channels"] = result["channel_mode"] == "independent"
            elif result:
                settings["independent_channels"] = False
            source_id = Path(result.get("input", "")).name
            if ID.fullmatch(source_id) and (self.root / "uploads" / source_id).is_file():
                settings["upload"] = source_id
            ref_id = Path(result.get("reference_path") or "").name
            if ID.fullmatch(ref_id) and (self.root / "uploads" / ref_id).is_file():
                settings["reference_upload"] = ref_id
            if not result.get("reference_path"):
                settings["reference_placement"] = "off"
        if "lyrics" not in settings and (folder / "lyrics.txt").is_file():
            settings["lyrics"] = (folder / "lyrics.txt").read_text()
        return settings

    def snapshot(self):
        with self.lock:
            positions = {j['id']: i + 1 for i, j in enumerate(self.pending())}
            return {"app": "yue2-mlx-studio", "jobs": [dict(j, queue_position=positions.get(j['id'])) for j in sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)],
                    "tokenizers": self.tokenizers(),
                    "ar_adapters": self.ar_adapters(),
                    "queued": len(positions),
                    "busy": self.active is not None,
                    "generation_ready": (self.model / "manifest.json").is_file(),
                    "semantic_audio_ready": (self.model / "manifest.json").is_file() and (self.tokenizer_model / "manifest.json").is_file(),
                    "audio_ready": (self.vae / "manifest.json").is_file()}

    def start(self, data):
        kind = data.get("kind")
        if kind=='train':return self.start_training(data)
        if kind not in ("generate", "reconstruct", "flow-edit", "semantic-audio"):
            raise ValueError("Choose generation, audio reconstruction, or flow editing.")
        model = self.vae if kind == "reconstruct" else self.model
        if not (model / "manifest.json").is_file():
            raise ValueError(f"Converted model not found: {model}")
        cmd = [sys.executable, "-u", "-m", "yue2_mlx.cli", kind, "--model", str(model),
               "--memory-gib", str(self.memory)]
        if kind == "generate":
            style, lyrics, abc = (data.get(k, "").strip() for k in ("style", "lyrics", "abc"))
            if not style:
                raise ValueError("Add a style; lyrics are optional.")
            seconds, steps, seed = (int(data.get(k, d)) for k, d in (("seconds", 8), ("steps", 32), ("seed", 831001)))
            cot = data.get("cot", "off")
            if seconds < 1 or steps < 1 or not 0 <= seed < 2**32:
                raise ValueError("Use positive integer seconds and steps, and a seed below 2³².")
            if cot not in ("off", "full", "melody") or (abc and cot == "off"):
                raise ValueError("Choose melody or full planning when supplying an ABC score.")
            cmd += ["--style", style, "--cot", cot, "--seed", str(seed), "--steps", str(steps),
                    "--max-semantic-tokens", str(seconds * 25), "--min-semantic-tokens", str(min(200, seconds * 25))]
            title = style[:85]
        else:
            upload = data.get("upload", "")
            if not ID.fullmatch(upload):
                raise ValueError("Drop an audio file first.")
            source = self.root / "uploads" / upload
            if not source.is_file():
                raise ValueError("Uploaded audio was not found.")
            cmd += ["--input", str(source)]
            independent = data.get("independent_channels", False)
            if type(independent) is not bool:
                raise ValueError("Independent channels must be true or false.")
            if independent:
                if kind == "semantic-audio":
                    raise ValueError("Semantic reconstruction currently uses mono semantics to stereo output.")
                cmd += ["--independent-channels"]
            title = str(data.get("name", "Audio reconstruction"))[:120]
            if kind == "semantic-audio":
                tokenizer_choice=data.get('tokenizer_choice','mothersuperior')
                selected_tokenizer=self.tokenizer_path(tokenizer_choice)
                if not (Path(selected_tokenizer) / "manifest.json").is_file():
                    raise ValueError("Install tokenizer weights first: python -m yue2_mlx.cli install-realaudio")
                style, lyrics = (data.get(k, "").strip() for k in ("style", "lyrics"))
                steps, seed = int(data.get("steps", 32)), int(data.get("seed", 831001))
                nar_adapter = data.get("nar_adapter", True)
                if steps < 1 or not 0 <= seed < 2**32 or type(nar_adapter) is not bool:
                    raise ValueError("Use positive steps, uint32 seed and a boolean adapter option.")
                cmd += ["--tokenizer-model", selected_tokenizer, "--style", style,
                        "--steps", str(steps), "--seed", str(seed)]
                if not nar_adapter:
                    cmd += ["--no-nar-adapter"]
            if kind in ("flow-edit", "semantic-audio"):
                import soundfile as sf
                style, lyrics = (data.get(k, "").strip() for k in ("style", "lyrics"))
                start_time = float(data.get("start_time", 1. if kind == "semantic-audio" else .15))
                mode = data.get("mode", "fresh" if kind == "semantic-audio" else "mix")
                anchor_strength = float(data.get("anchor_strength", 2.))
                steps, seed = int(data.get("steps", 32 if kind == "semantic-audio" else 16)), int(data.get("seed", 831001))
                modes = ("fresh", "mix", "direct", "anchored") if kind == "semantic-audio" else ("mix", "direct", "anchored")
                if (not math.isfinite(start_time) or not 0 <= start_time <= 1 or mode not in modes
                        or not math.isfinite(anchor_strength) or not anchor_strength >= 0
                        or steps < 1 or not 0 <= seed < 2**32):
                    raise ValueError("Use start time 0–1, mix/direct/anchored mode, nonnegative source pull, positive steps, and uint32 seed.")
                if kind == "semantic-audio":
                    noise_amount = data.get("noise_amount")
                    if noise_amount is not None:
                        noise_amount = float(noise_amount)
                        if not math.isfinite(noise_amount) or not 0 <= noise_amount <= 1:
                            raise ValueError("Noise amount must be 0–1 or automatic")
                        cmd += ["--noise-amount", str(noise_amount)]
                cmd += ["--style", style, "--start-time", str(start_time), "--mode", mode,
                        "--steps", str(steps), "--seed", str(seed), "--anchor-strength", str(anchor_strength)]
                reference_upload = data.get("reference_upload", "")
                reference_placement = data.get("reference_placement", "off")
                reference_strength = float(data.get("reference_strength", 1.))
                reference_name = str(data.get("reference_name", "Reference audio"))[:120]
                if reference_placement not in ("off", "prefix", "audio") or not math.isfinite(reference_strength) or not reference_strength >= 0:
                    raise ValueError("Use off/prefix/audio reference placement and finite nonnegative strength.")
                if reference_placement != "off":
                    if not isinstance(reference_upload, str) or not ID.fullmatch(reference_upload):
                        raise ValueError("Drop reference audio or turn reference placement off.")
                    ref_path = self.root / "uploads" / reference_upload
                    if not ref_path.is_file():
                        raise ValueError("Reference audio was not found. Upload it again.")
                    ref_info = sf.info(ref_path)
                    if not ref_info.duration >= .04 or ref_info.channels not in (1, 2):
                        raise ValueError("Reference audio must be mono/stereo and at least 0.04 seconds.")
                    cmd += ["--reference", str(ref_path), "--reference-placement", reference_placement,
                            "--reference-strength", str(reference_strength)]
        settings = {"kind": kind, "model": str(model), "memory_gib": self.memory}
        if kind in ('generate','flow-edit'):
            choice=data.get('acoustic_choice','stock')
            acoustic_scale=float(data.get('acoustic_scale',1.))
            if not math.isfinite(acoustic_scale) or acoustic_scale < 0:raise ValueError('Acoustic strength must be finite and nonnegative')
            acoustic=None if choice=='stock' else self.tokenizer_path(choice)
            if acoustic:
                if not (Path(acoustic)/'manifest.json').is_file():raise ValueError('Acoustic checkpoint is unavailable')
                cmd += ['--acoustic-checkpoint',acoustic,'--acoustic-scale',str(acoustic_scale)]
            settings.update(acoustic_choice=choice,acoustic_checkpoint=acoustic,acoustic_scale=acoustic_scale)
        if kind=='semantic-audio':
            acoustic_scale=float(data.get('acoustic_scale',1.))
            if not math.isfinite(acoustic_scale) or acoustic_scale < 0:raise ValueError('Acoustic strength must be finite and nonnegative')
            cmd += ['--acoustic-scale',str(acoustic_scale)]
            settings.update(acoustic_scale=acoustic_scale)
        if kind in ('generate','flow-edit','semantic-audio'):
            ar_choice=data.get('ar_choice','stock');ar_scale=float(data.get('ar_scale',1.))
            if not math.isfinite(ar_scale) or ar_scale < 0:raise ValueError('AR strength must be finite and nonnegative')
            ar_checkpoint=None if ar_choice=='stock' else self.ar_path(ar_choice)
            if ar_checkpoint:cmd += ['--ar-checkpoint',ar_checkpoint,'--ar-scale',str(ar_scale)]
            settings.update(ar_choice=ar_choice,ar_checkpoint=ar_checkpoint,ar_scale=ar_scale)
        if kind == "generate":
            settings.update(style=style, lyrics=lyrics, abc=abc, cot=cot, seconds=seconds, steps=steps, seed=seed)
        else:
            settings.update(upload=upload, name=title, independent_channels=independent)
            if kind == "semantic-audio":
                settings.update(style=style, lyrics=lyrics, steps=steps, seed=seed,
                                nar_adapter=nar_adapter, tokenizer_model=selected_tokenizer, tokenizer_choice=tokenizer_choice, noise_amount=noise_amount)
            if kind in ("flow-edit", "semantic-audio"):
                settings.update(style=style, lyrics=lyrics, mode=mode, start_time=start_time,
                                steps=steps, seed=seed, anchor_strength=anchor_strength,
                                reference_upload=reference_upload if reference_placement != "off" else "",
                                reference_name=reference_name, reference_placement=reference_placement,
                                reference_strength=reference_strength)
        with self.lock:
            if self.closing:
                raise BusyError("The server is shutting down.")
            identifier = uuid.uuid4().hex
            folder = self.root / identifier
            folder.mkdir()
            cmd += ["--output", str(folder / "output")]
            if kind in ("generate", "flow-edit", "semantic-audio"):
                (folder / "lyrics.txt").write_text(lyrics)
                cmd += ["--lyrics-file", str(folder / "lyrics.txt")]
                if kind == "generate" and abc:
                    (folder / "score-input.abc").write_text(abc)
                    cmd += ["--abc-file", str(folder / "score-input.abc")]
            job = {"id": identifier, "kind": kind, "title": title, "status": "queued",
                   "created": datetime.now(timezone.utc).isoformat(), "progress": "Waiting in queue", "files": [],
                   "settings": settings, "command": cmd}
            self.jobs[identifier] = job
            self.save(job)
            self.launch_next()
            return dict(job)

    def pending(self):
        return sorted((j for j in self.jobs.values() if j['status'] == 'queued'), key=lambda j: j['created'])

    def launch_next(self):
        """Caller holds lock. Persist before launch; never overlap inference processes."""
        if self.closing or self.active is not None:
            return
        for job in self.pending():
            job.update(status="running", progress="Starting…", started=datetime.now(timezone.utc).isoformat())
            self.save(job)
            try:
                process = subprocess.Popen(job['command'], stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
            except (OSError, KeyError, ValueError) as exc:
                job.update(status="failed", progress=f"Could not start inference: {exc}",
                           finished=datetime.now(timezone.utc).isoformat())
                self.save(job)
                continue
            self.active = (job['id'], process)
            threading.Thread(target=self.watch, args=(job, process), daemon=True).start()
            break

    def watch(self, job, process):
        folder = self.root / job["id"]
        lines = []
        with (folder / "run.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.strip():
                    lines.append(line.strip())
                    lines = lines[-12:]
                    with self.lock:
                        job["progress"] = line.strip()
        code = process.wait()
        with self.lock:
            cancelled = job["status"] == "cancelling"
            job["status"] = "cancelled" if cancelled else "done" if code == 0 else "failed"
            job["progress"] = "Stopped. Partial files are retained." if cancelled else "Ready to listen" if code == 0 else "\n".join(lines[-4:])
            output = folder / "output"
            job["files"] = [p.name for p in sorted(output.glob("*")) if p.is_file()]
            if (output / "result.json").is_file():
                try:
                    result = json.loads((output / "result.json").read_text())
                    job["truncated"] = result.get("truncated", False)
                    job["peak_gib"] = max((e["peak_gib"] for e in result.get("memory", [])), default=0)
                except (ValueError, KeyError):
                    pass
            job["finished"] = datetime.now(timezone.utc).isoformat()
            self.save(job)
            self.active = None
            self.launch_next()
        process.stdout.close()

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job and job['status'] == 'queued':
                job.update(status="cancelled", progress="Removed from queue.",
                           finished=datetime.now(timezone.utc).isoformat())
                self.save(job)
                return
            if self.active is None or self.active[0] != identifier:
                raise ValueError("This session is not running.")
            job = self.jobs[identifier]
            job.update(status="cancelling", progress="Stopping…")
            self.save(job)
            self.active[1].terminate()

    def close(self):
        with self.lock:
            self.closing = True
            if self.active:
                job = self.jobs[self.active[0]]
                job.update(status="cancelling", progress="Server stopping; queued jobs will resume on restart.")
                self.save(job)
                self.active[1].terminate()


def handler(studio):
    static = Path(__file__).parent / "static"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def json(self, value, status=200):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def local(self):
            host = self.headers.get("Host", "")
            if host not in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"):
                raise ValueError("Use the local studio address.")
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host:
                raise ValueError("Cross-origin requests are not allowed.")

        def do_GET(self):
            try:
                self.local()
                path = unquote(urlparse(self.path).path)
                if path == "/api/state":
                    return self.json(studio.snapshot())
                if path == '/api/training/songs':return self.json(studio.training_songs())
                match=re.fullmatch(r'/api/training/([a-f0-9]{32})',path)
                if match:return self.json(studio.training_progress(match[1]))
                if path == '/training':return self.file(static/'training.html')
                if path == '/training.js':return self.file(static/'training.js')
                if path in ("/", "/history", "/app.js", "/realaudio.js", "/style.css"):
                    return self.file(static / ("index.html" if path in ("/", "/history") else path[1:]))
                match = re.fullmatch(r"/files/([a-f0-9]{32})/([^/]+)", path)
                if match and match[2] not in (".", ".."):
                    target = studio.root / match[1] / "output" / match[2]
                    if target.resolve().parent == (studio.root / match[1] / "output").resolve():
                        return self.file(target)
                match = re.fullmatch(r"/uploads/([a-f0-9]{32})", path)
                if match:
                    return self.file(studio.root / "uploads" / match[1], mime="application/octet-stream")
                self.json({"error": "Not found"}, 404)
            except (ValueError, FileNotFoundError) as exc:
                self.json({"error": str(exc)}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                self.local()
                length = int(self.headers.get("Content-Length", "0"))
                path = urlparse(self.path).path
                if path == "/api/upload":
                    if length <= 0:
                        raise ValueError("Choose a nonempty audio file.")
                    identifier = uuid.uuid4().hex
                    target = studio.root / "uploads" / identifier
                    try:
                        with target.open("xb") as f:
                            remaining = length
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError("Upload was interrupted.")
                                f.write(chunk)
                                remaining -= len(chunk)
                        import soundfile as sf
                        info = sf.info(target)
                        if info.channels not in (1, 2) or info.duration <= 0 or info.frames < info.samplerate / 25:
                            raise ValueError("Choose mono/stereo audio with at least 0.04 seconds.")
                    except Exception:
                        target.unlink(missing_ok=True)
                        raise ValueError("Could not read this audio. Try WAV, FLAC, MP3, or OGG; mono/stereo audio required.")
                    return self.json({"id": identifier, "seconds": info.duration, "sample_rate": info.samplerate}, 201)
                if length <= 0:
                    raise ValueError("Request body must not be empty.")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                if path == "/api/jobs":
                    return self.json(studio.start(data), 201)
                if path == '/api/training/songs':return self.json(studio.set_training_songs(data.get('songs')))
                if path == "/api/cancel":
                    studio.cancel(data.get("id"))
                    return self.json({"ok": True})
                self.json({"error": "Not found"}, 404)
            except BusyError as exc:
                self.json({"error": str(exc)}, 409)
            except (ValueError, TypeError, AttributeError, OSError) as exc:
                self.json({"error": str(exc)}, 400)

        def file(self, path, mime=None):
            if not path.is_file():
                raise FileNotFoundError("File not found")
            size = path.stat().st_size
            start, end, partial = 0, size - 1, False
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]), end) if match[2] else end
                else:
                    start = max(0, size - int(match[2]))
                if start > end or start >= size:
                    self.send_error(416)
                    return
                partial = True
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = f.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local YuE2 studio")
    parser.add_argument("--model", default="models/yue2-8bit")
    parser.add_argument("--vae", default="models/yue2-vae")
    parser.add_argument("--output", default="outputs/web")
    parser.add_argument("--memory-gib", type=float, default=10)
    parser.add_argument("--tokenizer-model", default="models/realaudio-v4")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="Open the studio in your browser")
    args = parser.parse_args()
    studio = Studio(args.output, args.model, args.vae, args.memory_gib, args.tokenizer_model)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(studio))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE and args.open:
            try:
                with urlopen(f"http://127.0.0.1:{args.port}/api/state", timeout=2) as response:
                    existing = json.load(response)
                if existing.get("app") == "yue2-mlx-studio":
                    import webbrowser
                    webbrowser.open(f"http://127.0.0.1:{args.port}")
                    print("YuE2 Studio is already running. Opened it in your browser.")
                    return
            except (OSError, ValueError):
                pass
        parser.exit(1, f"Could not start the studio on port {args.port}: {exc}\n")
    studio.resume_queue()
    print(f"YuE2 Studio → http://127.0.0.1:{server.server_port}", flush=True)
    if args.open:
        import webbrowser
        threading.Timer(.5, lambda: webbrowser.open(f"http://127.0.0.1:{server.server_port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.close()
        server.server_close()


if __name__ == "__main__":
    main()
