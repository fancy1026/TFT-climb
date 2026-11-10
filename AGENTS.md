# TFT Climb

A team of language-model agents that plays Teamfight Tactics Ranked and learns from its games. The player plays
through the harness in `tft_player/`; the agents that work between games are described in `agent/FRAMEWORK.md`, and
each one's instructions are in `agent/roles/`.

- `agent/climb.py` runs the loop: games in batches of five; after each batch, its review, harness fixes and strategy
  changes, then the next batch.
- An agent changes only the files its role allows; the loop takes back anything else.
- Games start only when the person running this starts them. Never push.
