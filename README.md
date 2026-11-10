# TFT Climb

A multi-agent framework that plays Teamfight Tactics Ranked and learns from its own games and from strong
players' games. Built on [TFT Harness](https://github.com/fancy1026/tft-harness).

- **Learns from its own games.** After every five games a Reviewer grades every decision, the five games side by
  side; a Strategist rewrites the stage strategies that went wrong, and an Editor fixes the harness faults the review
  found.
- **Learns from strong players.** It finds strong players' games of its comp on YouTube and Twitch and reads them
  round by round from the video. The player plays along with them, and every new strategy is tested against their
  decisions.
- **Every change is tested on the game that found it.** A fault becomes a round case: the round's recorded question
  and a check. A fix lands only when its case passes and no other case breaks; a new strategy ships only if a Judge
  in code scores it no lower, on games no agent has read.
- **An expert in the loop.** A domain expert's feedback becomes rules that come first for the player and for every
  agent. An agent may disagree, but only with reasons and evidence, and the rule holds until the expert changes it.

<p align="center">
  <img src="media/framework.svg" alt="Figure 1: during a game the player model plans each round and the harness carries it out on TFT and records everything; between games the Reviewer grades the record, the Editor fixes harness faults through round cases, the Strategist rewrites stage strategies through the Judge, strong players' games read from video feed the Judge, the play-along and the comp notes, and the domain expert's rules come first in what the player reads" width="100%">
</p>

**Figure 1.** During a game (top), the player model plans each round and the harness carries the plan out. Between
games (bottom), the agents review the record, learn from strong players' games, and change what the player reads and
the harness it plays through.

## How it works

- **The player.** A language model plays every round through
  [TFT Harness](https://github.com/fancy1026/tft-harness), which reads the screen and carries out its plans. It plays
  with what the agents have learned so far: the expert's rules, the comp's notes and a strategy for each stage.
- **Review.** After every five games, a Reviewer goes through them stage by stage, the five side by side, and
  judges each decision by what was best at that moment, not by how the fight turned out.
- **Strategy.** When a stage goes wrong, the Strategist rewrites how to play it. A Judge lets the new version in only
  if it does at least as well on decisions already recorded.
- **Harness fixes.** When the harness itself made the mistake, the Editor fixes it, and the fix counts only once it
  works on the round where the mistake happened.
- **Strong players.** The agents watch strong players' games of the same comp: they learn the comp from them, set
  their own play beside them, and use their decisions to judge new strategies.
- **Guards.** Each agent can change only its own part, and games start only when you start them.

Each part as a diagram: [docs/framework.md](docs/framework.md).

## Running it

Windows 10 or 11, the TFT client in English and signed in, Python 3.11, Git, Claude Code signed in (the agents run as
Claude Code sessions) and the Codex app signed in for a GPT player (when its usage runs out the player falls back to
Claude). Before each match the harness sets TFT to borderless fullscreen (your settings are backed up).

```powershell
git clone https://github.com/fancy1026/tft-climb C:\tft-climb
cd C:\tft-climb
.\install.bat
.venv\Scripts\python agent\climb.py --games 5
```

It starts from the strategy our climb has learned so far (in `agent/`: the expert's rules, the comp's notes and
stage strategies, its code rules and the playbook), with a history of your own: your games, reviews and logs. To
start from nothing instead, so that the first climb learns the comp itself, run
`.venv\Scripts\python tools\start_empty.py` before it (`git checkout -- agent` brings the strategy back).

`agent/config.json` sets the comp, the player's model and effort, the agents' model and the video. After every five
games the agents review the five together, fix the harness and change the strategies before the next game. Options:
`--until 23:00` starts no game after that time; `--roles runs\<game>` runs the review and the fixes on a recorded
game. Your own rules for a comp go in `agent/expert/rules/<comp>.json`. **F7** pauses, **F8** stops all input, and an
`agent/STOP` file stops the loop. Every game is saved in `runs/` with each prompt, answer and screenshot, its video
and its report page.

## License

The code is under the [MIT license](LICENSE). Game data and icons are downloaded from CommunityDragon at install;
reference pictures cut from the game ship only as numeric fingerprints, except the four loot orb images in
`data/loot`. Those and the downloaded files belong to Riot Games and are not covered by the license. Other players'
names in the game records are replaced with Player1, Player2 and so on.
