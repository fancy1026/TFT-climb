# TFT Climb

A team of language-model agents that plays Teamfight Tactics Ranked and learns from every game. The player plays
through the harness in `tft_player/`; the agents that work between games are described in `agent/FRAMEWORK.md`, and
each one's instructions are in `agent/roles/`.

- `agent/climb.py` runs the loop: a game, then its review, harness fixes and strategy changes, then the next game.
- An agent changes only the files its role allows; the loop takes back anything else.
- Games start only when the person running this starts them. Never push.
