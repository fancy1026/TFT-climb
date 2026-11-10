"""Game settings the player relies on, applied before each match.

TFT is an Unreal client; its display settings live in GameUserSettings.ini and
are read at launch (and rewritten on exit), so we set them while the game is
not running:
- borderless fullscreen at the screen's own resolution (the HUD positions scale from 1080p)
- 100% render resolution: at 50% text and icons are blurred upscales
A backup of the user's original file is kept next to it.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

INI = Path(os.environ.get('LOCALAPPDATA', '')) / 'TFT' / 'Saved' / 'Config' / 'WindowsClient' / 'GameUserSettings.ini'

WANT = {
    # borderless fullscreen at the panel's native resolution: sharp (no 1080p
    # upscaling blur), not capped like windowed mode at 180% scaling, and
    # overlays (MetaTFT via Overwolf) can draw on it - exclusive fullscreen blocks them
    'FullscreenMode': '1', 'LastConfirmedFullscreenMode': '1',     # 1 = windowed fullscreen (borderless)
    'sg.ResolutionQuality': '100',
}


def native_resolution():
    """The primary screen's current resolution in physical pixels (Windows display scaling does not change it), or
    None. It was a constant 2560x1600 - the development PC's screen - which another player's screen would not have."""
    try:
        import win32api
        mode = win32api.EnumDisplaySettings(None, -1)           # ENUM_CURRENT_SETTINGS
        return int(mode.PelsWidth), int(mode.PelsHeight)
    except Exception:
        return None


def ensure_settings(path: Path = INI):
    """Returns a list of the keys changed (empty if already right)."""
    if not path.exists():
        return []
    want = dict(WANT)
    res = native_resolution()
    if res:
        want.update({'ResolutionSizeX': str(res[0]), 'ResolutionSizeY': str(res[1]),
                     'LastUserConfirmedResolutionSizeX': str(res[0]), 'LastUserConfirmedResolutionSizeY': str(res[1])})
    text = path.read_text(encoding='utf-8')
    changed = []
    for key, val in want.items():
        pat = re.compile(rf'^{re.escape(key)}=.*$', re.M)
        if pat.search(text):
            new = pat.sub(f'{key}={val}', text)
            if new != text:
                changed.append(key)
                text = new
    if changed:
        backup = path.with_suffix('.ini.user-backup')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text, encoding='utf-8')
    return changed


def launch_tft():
    """Start the TFT client through the Riot Client's own product launcher (its lockfile gives the port and
    password; no user password is typed). -> True once the request is accepted."""
    import base64
    import json
    import ssl
    import urllib.request
    lock = Path(os.environ.get('LOCALAPPDATA', '')) / 'Riot Games' / 'Riot Client' / 'Config' / 'lockfile'
    try:
        _, _, port, password, proto = lock.read_text(encoding='utf-8').strip().split(':')
    except (OSError, ValueError):
        return False
    req = urllib.request.Request(f'{proto}://127.0.0.1:{port}/product-launcher/v1/products/teamfighttactics/'
                                 f'patchlines/live', data=b'', method='POST')
    req.add_header('Authorization', 'Basic ' + base64.b64encode(f'riot:{password}'.encode()).decode())
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            return 200 <= r.status < 300
    except Exception as e:
        print(json.dumps({'tft_launch_failed': str(e)}), flush=True)
        return False


def mouse_accepted() -> bool:
    """Windows takes an injected zero-distance mouse move (SendInput returns 1). With the TFT client open it has refused
    every injected mouse event at times - 0, error 87, while keys still went through (11-06 game 66; 11-08 after game
    77, when the results screen's clicks never landed) - and then no click or drag of ours lands. Checked before every
    mouse input of ours: the structures are built once."""
    if not _ZERO_MOVE:
        _ZERO_MOVE.append(_zero_move())
    send, m, size = _ZERO_MOVE[0]
    return send(1, m, size) == 1


_ZERO_MOVE = []


def mouse_share(seconds=3.0) -> float:
    """The share of zero-distance mouse moves Windows takes over `seconds`, one every 0.1 s. Single refusals come and
    go with the mouse working (11-08: 1 of 595 over a minute, and one refused check stopped game 78 for nothing); a
    block refuses them all."""
    import time
    n = ok = 0
    end = time.time() + seconds
    while time.time() < end:
        n, ok = n + 1, ok + mouse_accepted()
        time.sleep(0.1)
    return ok / max(n, 1)


def _zero_move():
    import ctypes
    from ctypes import wintypes

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('mouseData', wintypes.DWORD),
                    ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD), ('dwFlags', wintypes.DWORD),
                    ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]

    class U(ctypes.Union):
        _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [('type', wintypes.DWORD), ('u', U)]

    m = INPUT(type=0, u=U(mi=MOUSEINPUT(0, 0, 0, 0x0001, 0, 0)))          # INPUT_MOUSE, MOUSEEVENTF_MOVE by 0,0
    return ctypes.WinDLL('user32').SendInput, ctypes.byref(m), ctypes.sizeof(INPUT)


MWB_EXES = ('mousewithoutborders.exe', 'mousewithoutbordershelper.exe')
_MWB_ASK = []            # the elevated close still waiting on its Windows prompt (one at a time)


def close_mwb(wait=60.0) -> list:
    """The owner (2026-11-08): when Windows refuses our mouse input, Mouse Without Borders is closed automatically and
    the input tried again. Its helper is ended here; its main processes run as SYSTEM (started by its service), so they
    are ended by an elevated taskkill that Windows asks the owner to allow - one prompt at a time, never stacked.
    -> the names still running after up to `wait` s ([]: closed)."""
    import subprocess
    import time
    import psutil

    def running():
        return [p for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() in MWB_EXES]

    for p in running():
        try:
            p.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    time.sleep(0.3)
    if running() and not (_MWB_ASK and _MWB_ASK[0].poll() is None):
        _MWB_ASK[:] = [subprocess.Popen(
            ['powershell', '-NoProfile', '-Command',
             "Start-Process taskkill -ArgumentList '/F','/IM','MouseWithoutBorders.exe','/IM',"
             "'MouseWithoutBordersHelper.exe' -Verb RunAs -WindowStyle Hidden"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x08000000)]     # CREATE_NO_WINDOW
    end = time.time() + wait
    while running() and time.time() < end:
        time.sleep(0.5)
    return sorted({p.info['name'] for p in running()})


def fullscreen_client(path: Path = INI, wait=180.0):
    """The owner (x18.3, 2026-11-06): the game is set to full screen automatically when its window is not the
    screen's resolution. Since 18.6 the TFT client stays open between games and writes its own (windowed) settings
    back on exit, so ensure_settings alone did nothing (games 65-66 ran in a 1920x1200 window on a 2560x1600
    screen): the client is closed, the settings are written, and it is started again. -> what was done, or None."""
    import time
    import psutil
    import win32con
    import win32gui
    from tft_player.game import TFT_EXES, find_client
    c, res = find_client(), native_resolution()
    if c is None or not res or (c.width, c.height) == tuple(res):
        return None
    was = (c.width, c.height)
    win32gui.PostMessage(c.hwnd, win32con.WM_CLOSE, 0, 0)
    procs = [p for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() in TFT_EXES]
    for p in psutil.wait_procs(procs, timeout=60)[1]:
        p.kill()
    changed = ensure_settings(path)
    # the Riot Client answers 423 Locked while the closed session is still being released, and a launch that went
    # through can answer after the read timed out (game 66, 2026-11-06): it is asked again, and the window waited for
    end = time.time() + wait
    started, now, tries = False, None, 0
    while time.time() < end:
        now = find_client()
        if now and (now.width, now.height) == tuple(res):
            break
        if not started and now is None and tries < 6:
            tries += 1
            started = launch_tft()
            time.sleep(1 if started else 5)
            continue
        time.sleep(1)
    return {'window_was': was, 'screen': res, 'changed': changed, 'launched': started or now is not None,
            'launch_tries': tries, 'window_now': (now.width, now.height) if now else None}

