#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def emit(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def safe_stem(text):
    cleaned = []
    for ch in text.strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    stem = "".join(cleaned).strip("_")
    return stem or "youtube_audio"


def unique_output_path(directory, stem):
    candidate = directory / f"{stem}.wav"
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{suffix}.wav"
        suffix += 1
    return candidate


def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return result.returncode == 0, output


def yt_dlp_base_cmd(url, temp_dir):
    return [
        "yt-dlp",
        "--no-update",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--user-agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "--extractor-args",
        "youtube:player_client=web_embedded,tv",
        "--paths",
        f"home:{temp_dir}",
        "--paths",
        f"temp:{temp_dir}",
        "--output",
        "%(title)s.%(ext)s",
        url,
    ]


def try_download(url, temp_dir, cookies_file=None, browser=None):
    cmd = yt_dlp_base_cmd(url, temp_dir)
    label = "without cookies"
    if cookies_file:
        cmd[1:1] = ["--cookies", str(cookies_file)]
        label = "with saved cookies"
    if browser:
        insert_at = 1
        if cookies_file:
            insert_at = 3
        cmd[insert_at:insert_at] = ["--cookies-from-browser", browser]
        label = f"with {browser} browser cookies"

    emit(event="progress", message=f"Downloading {label}")
    return run_command(cmd)


def newest_wav(temp_dir):
    wavs = list(Path(temp_dir).glob("*.wav"))
    if not wavs:
        return None
    return max(wavs, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio into a project WAV.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="", help="Output filename stem (defaults to the video title)")
    parser.add_argument("--cookies-file", default="")
    parser.add_argument("--browser", default="none")
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        emit(ok=False, error="yt-dlp not found in PATH")
        return 1
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        emit(ok=False, error="ffmpeg and ffprobe are required by yt-dlp audio extraction")
        return 1

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cookie_path = Path(args.cookies_file).expanduser() if args.cookies_file else None
    browser = args.browser.strip().lower()
    if browser == "none":
        browser = ""

    if cookie_path:
        cookie_path.parent.mkdir(parents=True, exist_ok=True)
        if not cookie_path.exists():
            cookie_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            try:
                os.chmod(cookie_path, 0o600)
            except OSError:
                pass

    failures = []
    with tempfile.TemporaryDirectory(prefix="ladidaw-youtube-") as temp_dir:
        downloaded = False

        if cookie_path and cookie_path.exists() and cookie_path.stat().st_size > 0:
            ok, output = try_download(args.url, temp_dir, cookies_file=cookie_path)
            if ok:
                downloaded = True
            else:
                failures.append(("saved cookies", output))

        if not downloaded and browser:
            ok, output = try_download(args.url, temp_dir, cookies_file=cookie_path, browser=browser)
            if cookie_path and cookie_path.exists():
                try:
                    os.chmod(cookie_path, 0o600)
                except OSError:
                    pass
            if ok:
                downloaded = True
            else:
                failures.append((f"{browser} browser cookies", output))

        if not downloaded and not browser:
            ok, output = try_download(args.url, temp_dir)
            if ok:
                downloaded = True
            else:
                failures.append(("no cookies", output))

        if not downloaded:
            detail = failures[-1][1] if failures else "download failed"
            emit(ok=False, error=detail[-4000:])
            return 1

        wav = newest_wav(temp_dir)
        if wav is None:
            emit(ok=False, error="yt-dlp completed but no WAV file was produced")
            return 1

        stem = safe_stem(args.name or wav.stem)
        output_path = unique_output_path(output_dir, stem)
        shutil.move(str(wav), str(output_path))

    emit(ok=True, output_path=str(output_path), import_name=output_path.stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
