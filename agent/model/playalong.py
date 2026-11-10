"""Step 1 - play along with a strong player's game (the owner's pipeline, 2026-10-26): our player decides every round
from his screen, the HUD and the calculator, as if it were playing; its action is compared with his. Only comparisons
are collected; the Strategist reads them (agent/roles/strategist.md).

    python agent/model/playalong.py <game> [<game> ...]   -> agent/model/playalong/<game>.json and .md
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decide  # noqa: E402
import modelfile  # noqa: E402

OUT = HERE / 'playalong'


def said(subs, a, b):
    return ' / '.join(s['content'] for s in subs if a <= s['from'] < b)


def play_along(game, workers=4):
    model = modelfile.read()
    mem_dir = decide.memory_dir(model)
    memory = decide.judge.memory_text(f'dir:{mem_dir}')
    rs = decide.rounds(game)
    folder = decide.ROOT / 'runs' / game.split('-g')[0]
    sub = next(folder.glob('subtitles-ai-zh*.json'), None)
    subs = json.loads(sub.read_text(encoding='utf-8'))['body'] if sub else []
    with ThreadPoolExecutor(workers) as ex:
        answers = list(ex.map(lambda r: decide.ask(memory, r[0], r[1]), rs))
    rows = []
    for i, ((c, start, his), rec) in enumerate(zip(rs, answers)):
        nxt = rs[i + 1][0] if i + 1 < len(rs) else None
        bot = rec['answer'] if isinstance(rec['answer'], dict) else {}
        rows.append({'round': c['stage'], 'case': c['id'], 'start': start[:3] if start else None,
                     'ours': {k: bot.get(k) for k in ('level_end', 'gold_end', 'xp_buys', 'roll_gold', 'buy', 'sell', 'why',
                                                      'principles')},
                     'his': his, 'his_words': said(subs, c['t'] - 4, nxt['t'] if nxt else c['t'] + 60),
                     'screens': [c['state']['keyframe'], nxt['state']['keyframe'] if nxt else None],
                     'differences': decide.differences(c, start, bot, his),
                     'agreement': decide.points(c, bot, his)})
    OUT.mkdir(exist_ok=True)
    agree = sum(r['agreement'][0] for r in rows), sum(r['agreement'][1] for r in rows)
    res = {'game': game, 'model_principles': [p['id'] for p in modelfile.principles(model) if p['status'] == 'active'],
           'rounds': rows, 'agreement': agree, 'rounds_with_differences': sum(bool(r['differences']) for r in rows),
           'cost_usd': round(sum(a.get('cost') or 0 for a in answers), 2)}
    (OUT / f'{game}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    md = [f"# Play-along: {game}\n", f"Agreement with his choices: {agree[0]} of {agree[1]} points; "
          f"{res['rounds_with_differences']} of {len(rows)} rounds differ.\n",
          '| round | we | he | differences | his words |', '|---|---|---|---|---|']
    for r in rows:
        o, h = r['ours'], r['his']
        md.append(f"| {r['round']} | L{o.get('level_end')} {o.get('gold_end')}g xp{o.get('xp_buys')} roll{o.get('roll_gold')} "
                  f"{', '.join(o.get('buy') or [])} | L{h['level_end']} {h['gold_end']}g xp{h['xp_buys']} other{h['other_gold']} | "
                  f"{'; '.join(r['differences']) or '-'} | {r['his_words'][:120]} |")
    (OUT / f'{game}.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return res


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    for g in sys.argv[1:]:
        r = play_along(g)
        print(f"{g}: agreement {r['agreement'][0]}/{r['agreement'][1]}, {r['rounds_with_differences']} of "
              f"{len(r['rounds'])} rounds differ, ${r['cost_usd']}")


if __name__ == '__main__':
    main()
