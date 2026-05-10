"""Headless throughput benchmark for RacerEnv.

Runs the env with random actions and reports how fast we can step it. The number
here directly determines how long RL training will take in Stage 4+.
"""

import random
import time

from environment.env import RacerEnv
from environment.action import Action


def random_action():
    return Action(
        steer=random.uniform(-1.0, 1.0),
        throttle=1.0 if random.random() < 0.9 else 0.0,
    )


def main(n_steps=50_000):
    env = RacerEnv()
    env.reset()

    episodes = 0
    finishes = 0
    timeouts = 0

    start = time.perf_counter()
    for _ in range(n_steps):
        _, _, done, info = env.step(random_action())
        if done:
            episodes += 1
            if info["crossed_finish"]:
                finishes += 1
            elif info["timed_out"]:
                timeouts += 1
            env.reset()
    elapsed = time.perf_counter() - start

    steps_per_sec = n_steps / elapsed
    realtime_multiplier = steps_per_sec / 60  # env logic targets 60 FPS

    print(f"Ran {n_steps:,} steps in {elapsed:.2f}s")
    print(f"Throughput: {steps_per_sec:,.0f} steps/sec ({realtime_multiplier:.1f}x real-time)")
    print(f"Episodes: {episodes}  (finishes: {finishes}, timeouts: {timeouts})")
    print(f"1M-step training would take ~{1_000_000 / steps_per_sec:.0f}s")


if __name__ == "__main__":
    main()
