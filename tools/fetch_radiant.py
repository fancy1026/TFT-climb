"""The Radiant item icons the item reader needs, downloaded at install instead of shipped in the repository: the
list (names and addresses on CommunityDragon) is tft_player/assets/radiant-items/catalogue.json.

    python tools/fetch_radiant.py          (install.bat runs it before fetch_assets.py; files already there are kept)
"""
import json
import urllib.request
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent / 'tft_player' / 'assets' / 'radiant-items'


def main():
    items = json.loads((FOLDER / 'catalogue.json').read_text(encoding='utf-8'))['items']
    got = failed = 0
    for it in items:
        path = FOLDER / it['file']
        if path.exists():
            continue
        try:
            req = urllib.request.Request(it['url'], headers={'User-Agent': 'Mozilla/5.0'})
            path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            got += 1
        except Exception as e:
            failed += 1
            print(f"{it['name']}: {e}")
    print(f'Radiant item icons: {got} new, {len(list(FOLDER.glob("*.png")))} in all'
          + (f', {failed} failed' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
