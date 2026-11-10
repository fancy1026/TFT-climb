import os

# numpy's OpenBLAS on one thread, set before numpy loads (every game entry imports tft_player first). Its 16 worker
# threads, woken for each small matrix product (bench containers, item icons), waited on the game for the CPU: one
# bench.container_match took 106-129 ms at p50 with 16 threads and 9-10 ms with one, a game running, and the bench
# check held 130-540 s of ticks over 1 s in each batch-2 game (experiments/README.md E3.4)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
