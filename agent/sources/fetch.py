"""Download picked videos from the signed links the logged-in browser page returned (links valid ~2 h).

    python agent/sources/fetch.py <links.json>
links.json: [{"bvid", "part", "video": [urls], "audio": [urls], "subtitle": url or null}, ...]
-> runs/vod-<bvid>/game.mp4 (a video in parts is joined into one) + its AI subtitles. Resumes partial files;
tries each mirror.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36'


CHUNK, WORKERS = 16 * 2 ** 20, 6


def size_of(u):
    """Total bytes, from a 1-byte range request's Content-Range."""
    r = subprocess.run(['curl', '-s', '-D', '-', '-o', '/dev/null', '-m', '20', '-r', '0-0', '-A', UA, '-H',
                        'Referer: https://www.bilibili.com/', u], capture_output=True, text=True)
    m = [l for l in r.stdout.splitlines() if l.lower().startswith('content-range')]
    return int(m[0].rsplit('/', 1)[1]) if m else None


def curl(urls, out):
    """Download in 16 MB ranges, several at once: one long transfer from this CDN stalls after a few tens of MB
    (25 Oct: 30 MB then nothing), while 10 MB ranges came at ~2 MB/s each. Each range retries on every mirror."""
    from concurrent.futures import ThreadPoolExecutor
    total = next((s for s in (size_of(u) for u in urls) if s), None)
    if not total:
        return False
    parts = [(i, a, min(a + CHUNK, total) - 1) for i, a in enumerate(range(0, total, CHUNK))]
    tmp = out.with_suffix(out.suffix + '.parts')
    tmp.mkdir(exist_ok=True)

    def one(p):
        i, a, b = p
        f = tmp / f'{i:05d}'
        if f.exists() and f.stat().st_size == b - a + 1:
            return True
        for attempt in range(6):
            u = urls[attempt % len(urls)]
            r = subprocess.run(['curl', '-s', '-f', '-m', '120', '-r', f'{a}-{b}', '-A', UA, '-H',
                                'Referer: https://www.bilibili.com/', '-o', str(f), u])
            if r.returncode == 0 and f.exists() and f.stat().st_size == b - a + 1:
                return True
        return False
    with ThreadPoolExecutor(WORKERS) as ex:
        ok = all(ex.map(one, parts))
    if not ok:
        return False
    with open(out, 'wb') as w:
        for i, _, _ in parts:
            w.write((tmp / f'{i:05d}').read_bytes())
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return out.stat().st_size == total


def main():
    import imageio_ffmpeg
    from collections import defaultdict
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    videos = defaultdict(list)
    for x in json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')):
        videos[x['bvid']].append(x)
    for bvid, parts in videos.items():
        d = ROOT / 'runs' / f'vod-{bvid}'
        d.mkdir(parents=True, exist_ok=True)
        if (d / 'game.mp4').exists() or (d / 'read' / 'reads.jsonl').exists():
            print(bvid, 'already there')
            continue
        done = []
        for x in sorted(parts, key=lambda x: x['part']):
            k = x['part']
            out = d / f'part{k}.mp4'
            if not out.exists():
                if not (curl(x['video'], d / f'video{k}.m4s') and curl(x['audio'], d / f'audio{k}.m4s')):
                    print(bvid, k, 'DOWNLOAD FAILED', flush=True)
                    break
                subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(d / f'video{k}.m4s'), '-i',
                                str(d / f'audio{k}.m4s'), '-c', 'copy', str(out)], check=True)
                (d / f'video{k}.m4s').unlink()
                (d / f'audio{k}.m4s').unlink()
            done.append(out)
            if x.get('subtitle'):
                try:
                    req = urllib.request.Request(x['subtitle'], headers={'User-Agent': UA})
                    (d / ('subtitles-ai-zh.json' if len(parts) == 1 else f'subtitles-ai-zh-p{k}.json')).write_bytes(
                        urllib.request.urlopen(req, timeout=30).read())
                except Exception as ex:
                    print(bvid, k, 'no subtitles:', ex)
        if len(done) != len(parts):
            continue
        if len(done) == 1:
            done[0].rename(d / 'game.mp4')
        else:                                            # a stream cut into hour-long parts: one video, games stay whole
            (d / 'parts.txt').write_text(''.join(f"file '{p.name}'\n" for p in done), encoding='utf-8')
            subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i',
                            str(d / 'parts.txt'), '-c', 'copy', str(d / 'joining.mp4')], check=True)
            (d / 'joining.mp4').rename(d / 'game.mp4')   # game.mp4 appears only whole: a reader may be waiting for it
            for p in done:
                p.unlink()
        print(bvid, 'ok', len(parts), 'part(s)', round((d / 'game.mp4').stat().st_size / 2 ** 20), 'MB', flush=True)


if __name__ == '__main__':
    main()
