"""The pro-game pipeline into the Strategist (the owner's structure v3, step 9): the strong players' games the
Strategist asked to study (its "study" requests, agent/sources/requests.jsonl) are found, downloaded, read and played
along with, so its next run finds them in agent/model/playalong/.

    python agent/study_queue.py            work through the open requests

- a query: agent/sources/search.py --query -> agent/sources/candidates.jsonl (the Strategist picks a video next run);
- a Twitch or YouTube video ("url"; the owner, 2026-10-29: "there are great US players as well ... download the
  twitch recording directly"): downloaded by yt-dlp at up to 1080p ("section" "0:00:00-1:00:00" for part of a long
  stream) into runs/vod-tw-<id>/ or runs/vod-yt-<id>/;
- a Bilibili video ("bvid"): Bilibili answers scripts with HTTP 412, so it is read only once runs/vod-<bvid>/game.mp4
  is there (a browser step, agent/sources/fetch.py); until then it waits as "needs download".
A downloaded video: agent/vod_read.py, agent/cases.py vod (its decisions; a stream is split into its games), and
agent/model/playalong.py on the games that went to the train split - a test game is the Judge's held-out set and is
never played along with. The video is deleted after (derived data only).
"""
import json
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / 'agent'
QUEUE = AGENT / 'sources' / 'requests.jsonl'
CANDIDATES = AGENT / 'sources' / 'candidates.jsonl'
SPLITS = AGENT / 'decisions' / 'splits.json'
PY = sys.executable


def rows():
    return [json.loads(l) for l in QUEUE.read_text(encoding='utf-8').splitlines() if l.strip()] if QUEUE.exists() else []


def save(rs):
    QUEUE.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs), encoding='utf-8')


def candidate(bvid):
    for l in CANDIDATES.read_text(encoding='utf-8').splitlines() if CANDIDATES.exists() else []:
        if l.strip() and json.loads(l).get('bvid') == bvid:
            return json.loads(l)
    return {}


def run(cmd):
    r = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return r.returncode == 0, (r.stdout + r.stderr)[-600:]


def splits():
    return json.loads(SPLITS.read_text(encoding='utf-8')) if SPLITS.exists() else {'test': [], 'train': []}


VIDEOS = AGENT / 'sources' / 'videos.jsonl'
MIN_GAME_S = 15 * 60                     # shorter is a guide or a clip, not a full game
MIN_STAGED_READS = 100                   # frames with a stage read: fewer is no game read (a full game has ~1000)


def search_youtube(query, n=12):
    """Full-length videos for a query on YouTube (yt-dlp search, no login): [{"url", "id", "duration", "channel",
    "title"}], also merged into agent/sources/videos.jsonl. A strong player's game of the comp, for the reflection
    (the owner, 2026-10-29: "learn from how others play the same comp via videos")."""
    exe = shutil.which('yt-dlp')
    if not exe:
        return []
    r = subprocess.run([exe, '--flat-playlist', '--print', '%(id)s\t%(duration)s\t%(channel)s\t%(title)s',
                        f'ytsearch{n}:{query}'], capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=180)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        try:
            dur = int(float(parts[1]))
        except ValueError:
            dur = 0
        if dur >= MIN_GAME_S:
            out.append({'url': f'https://www.youtube.com/watch?v={parts[0]}', 'id': parts[0], 'duration': dur,
                        'channel': parts[2], 'title': parts[3], 'query': query})
    have = {json.loads(l)['id'] for l in VIDEOS.read_text(encoding='utf-8').splitlines() if l.strip()} \
        if VIDEOS.exists() else set()
    with open(VIDEOS, 'a', encoding='utf-8') as f:
        for v in out:
            if v['id'] not in have:
                f.write(json.dumps(v, ensure_ascii=False) + '\n')
    return out


def url_folder(url):
    """runs/vod-tw-<id> for a Twitch VOD, runs/vod-yt-<id> for a YouTube video, or None."""
    m = re.search(r'twitch\.tv/videos/(\d+)', url)
    if m:
        return ROOT / 'runs' / f'vod-tw-{m.group(1)}'
    m = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})', url)
    if m:
        return ROOT / 'runs' / f'vod-yt-{m.group(1)}'
    return None


def download(url, folder, section=None):
    """yt-dlp, at up to 1080p, into folder/game.mp4 (with English captions where the site has them)."""
    exe = shutil.which('yt-dlp')
    if not exe:
        return False, 'yt-dlp is not installed'
    import imageio_ffmpeg
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [exe, '-f', 'bv*[height<=1080]+ba/b[height<=1080]', '--merge-output-format', 'mp4',
           '--ffmpeg-location', imageio_ffmpeg.get_ffmpeg_exe(), '-o', str(folder / 'game.%(ext)s'),
           '--no-playlist', '--sleep-requests', '1']
    if section:
        cmd += ['--download-sections', f'*{section}']
    r = subprocess.run(cmd + [url], capture_output=True, text=True, encoding='utf-8', errors='replace')
    # the captions apart, and their failure is not the video's: a 429 on the captions aborted both downloads of game
    # 40's reflection (YouTube limits caption requests sooner than video ones)
    for wait in (0, 60):                    # captions: once more after a minute (YouTube's 429 hits them first)
        time.sleep(wait)
        subprocess.run([exe, '--skip-download', '--write-auto-subs', '--sub-langs', 'en', '--no-playlist', '-o',
                        str(folder / 'game.%(ext)s'), url], capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
        if list(folder.glob('game*.vtt')):
            break
    return (folder / 'game.mp4').exists(), (r.stdout + r.stderr)[-400:]


def work():
    """-> {'done': [...], 'needs_download': [...], 'failed': [...]}"""
    rs, out = rows(), {'done': [], 'needs_download': [], 'failed': []}
    done_urls = {r.get('url') for r in rs if r.get('status') == 'done' and r.get('url')}
    for r in rs:
        if r.get('status') not in ('open', 'needs download'):
            continue
        if r.get('url') and r['url'] in done_urls:      # asked again: read already (or earlier in this run)
            r['status'] = 'duplicate'
            continue
        if r.get('query') and not (r.get('bvid') or r.get('url')) and r.get('site') == 'youtube':
            found = search_youtube(r['query'])
            r['status'] = 'searched'
            out['done'].append(f"search {r['query']} ({len(found)} videos)")
            continue
        if r.get('query') and not (r.get('bvid') or r.get('url')):
            ok, log = run([str(AGENT / 'sources' / 'search.py'), '--query', r['query']])
            r['status'] = 'searched' if ok else 'failed'
            (out['done'] if ok else out['failed']).append(f"search {r['query']}")
            continue
        if r.get('url'):
            folder = url_folder(r['url'])
            if folder is None:
                r.update(status='failed', error='not a Twitch VOD or YouTube video link')
                out['failed'].append(r['url'])
                continue
            seen = next((json.loads(l) for l in VIDEOS.read_text(encoding='utf-8').splitlines() if l.strip()
                         and json.loads(l).get('url') == r['url']), {}) if VIDEOS.exists() else {}
            name, client, multi = folder.name, 'NA en', 'twitch' in r['url']
            player = r.get('player') or seen.get('channel') or 'pro'
            if not (folder / 'game.mp4').exists():
                ok, log = download(r['url'], folder, r.get('section'))
                if not ok and '429' in log:              # YouTube's rate limit (game 40's reflection): once more later
                    import time
                    time.sleep(90)
                    ok, log = download(r['url'], folder, r.get('section'))
                if not ok:
                    r.update(status='failed', error=log)
                    out['failed'].append(f"{name}: download")
                    continue
        else:
            bvid = r.get('bvid')
            folder = ROOT / 'runs' / f'vod-{bvid}'
            name, client, player, multi = bvid, 'CN zh', candidate(bvid).get('uploader') or 'pro', False
            if not (folder / 'game.mp4').exists():
                r['status'] = 'needs download'
                out['needs_download'].append(bvid)
                continue
        video = folder / 'game.mp4'
        before = set(splits().get('train', [])) | set(splits().get('test', []))
        steps = [[str(AGENT / 'vod_read.py'), str(video), '--workers', '4'],
                 [str(AGENT / 'cases.py'), 'vod', str(folder), '--player', player, '--client', client]
                 + (['--multi'] if multi else [])]
        failed = None
        for cmd in steps:
            ok, log = run(cmd)
            if not ok:
                failed = f'{Path(cmd[0]).name}: {log}'
                break
            if Path(cmd[0]).name == 'vod_read.py':
                # a read with no stage is no game read (CurSEdzxz's video: 0 of 623 frames, marked studied, an empty
                # play-along): failed, with why, not done
                reads = folder / 'read' / 'reads.jsonl'
                staged = sum('"stage": "' in l for l in reads.read_text(encoding='utf-8').splitlines()) \
                    if reads.exists() else 0
                if staged < MIN_STAGED_READS:
                    failed = (f'vod_read.py read a stage in {staged} frames: the HUD was not found (another layout, '
                              f'an overlay, or not a game)')
                    break
        if not failed:
            s = splits()
            new_train = sorted(set(s.get('train', [])) - before)
            r['games'] = {'train': new_train, 'test': sorted(set(s.get('test', [])) - before)}
            for g in new_train:                          # the held-out (test) games are never played along with
                ok, log = run([str(AGENT / 'model' / 'playalong.py'), g])
                if not ok:
                    failed = f'playalong.py {g}: {log}'
                    break
        if failed:
            r.update(status='failed', error=failed[-500:])
            out['failed'].append(name)
            continue
        r['status'] = 'done'
        out['done'].append(name)
        if r.get('url'):
            done_urls.add(r['url'])
        try:                                             # the player's words, from the video while it is here
            import transcribe
            r['words'] = transcribe.transcribe(folder)
        except Exception as e:                           # no words is no failed study
            r['words'] = f'transcription failed: {e.__class__.__name__}: {e}'[:300]
        video.unlink()                                   # the owner's terms: derived data only (captions stay)
    save(rs)
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(json.dumps(work(), ensure_ascii=False))
