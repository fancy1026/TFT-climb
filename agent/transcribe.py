"""A strong player's words in a video when the site gives no captions (YouTube answers caption requests with HTTP 429
for long stretches; Twitch has none): the audio transcribed on this PC with faster-whisper (the owner approved the
install, 2026-10-29), written as game.whisper.vtt next to the read - agent/comp_coach.py reads it like captions, by
seconds into the video, matched to the rounds.

    python agent/transcribe.py runs/vod-yt-<id> [--url <the video>] [--section 0:00:00-1:00:00]

The video's own file is used while it is there (agent/study_queue.py calls this before deleting it); for a game read
earlier, only the audio is downloaded (the same section of a long stream), transcribed and deleted.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = 'small.en'                    # ~460 MB, downloaded once into the Hugging Face cache



def yt_dlp():
    """yt-dlp: on PATH, else the one requirements.txt put beside this Python (.venv/Scripts, which is not on PATH
    when the climb runs as .venv/Scripts/python - a fresh install found none)."""
    return shutil.which('yt-dlp') or next((str(p) for p in (Path(sys.executable).with_name('yt-dlp.exe'),
                                                              Path(sys.executable).with_name('yt-dlp')) if p.exists()), None)


def stamp(s):
    h, rem = divmod(max(0.0, s), 3600)
    m, sec = divmod(rem, 60)
    return f'{int(h):02d}:{int(m):02d}:{sec:06.3f}'


def audio(url, folder, section=None):
    """The video's audio only (yt-dlp), for a game whose video is gone. -> the file, or None."""
    exe = yt_dlp()
    if not exe or not url:
        return None
    cmd = [exe, '-f', 'bestaudio', '--no-playlist', '-o', str(folder / 'audio.%(ext)s')]
    if section:
        cmd += ['--download-sections', f'*{section}']
    subprocess.run(cmd + [url], capture_output=True, text=True, encoding='utf-8', errors='replace')
    return next(iter(sorted(folder.glob('audio.*'))), None)


def transcribe(folder, url=None, section=None, model=MODEL):
    """-> what happened, in a few words."""
    folder = Path(folder)
    if list(folder.glob('game*.vtt')):
        return 'captions already there'
    src, fetched = folder / 'game.mp4', False
    if not src.exists():
        src, fetched = audio(url, folder, section), True
        if src is None:
            return 'no video and no audio to transcribe'
    t0 = time.time()
    try:
        from faster_whisper import WhisperModel
        w = WhisperModel(model, device='cpu', compute_type='int8')
        segments, info = w.transcribe(str(src), language='en', vad_filter=True, beam_size=1)
        cues = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    finally:
        if fetched and src and src.exists():
            src.unlink()                                 # the owner's terms: derived data only
    out = folder / 'game.whisper.vtt'
    out.write_text('WEBVTT\n\n' + ''.join(f'{stamp(a)} --> {stamp(b)}\n{t}\n\n' for a, b, t in cues), encoding='utf-8')
    return f'{len(cues)} lines in {time.time() - t0:.0f} s -> {out.name}'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--url')
    ap.add_argument('--section')
    ap.add_argument('--model', default=MODEL)
    a = ap.parse_args()
    print(transcribe(a.folder, a.url, a.section, a.model))


if __name__ == '__main__':
    main()
