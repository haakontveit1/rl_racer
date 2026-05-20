"""Train a PPO policy on the racer.

Saves:
- models/best/best_model.zip                    — best policy by deterministic eval reward.
- models/checkpoints/ppo_racer_{N}_steps.zip    — periodic snapshots (for evolution viewer).
- models/ppo_racer_final.zip                    — final-step policy.

After training, prints a clearly delimited summary block — copy that block
back to chat to share results.
"""

import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from environment.env import RacerEnv
from gym_env import GymRacer


# Track-scoped paths so different track designs don't overwrite each other's runs.
# Old simple-L-track models live in models/simple_track/.
MODELS_DIR = "models/chicane"
BEST_MODEL_DIR = "models/chicane/best"
CHECKPOINTS_DIR = "models/chicane/checkpoints"
LOG_DIR = "logs/chicane"
TOTAL_TIMESTEPS = 1_500_000
EVAL_FREQ = 10_000
CHECKPOINT_FREQ = 10_000

# Linear LR schedule: 3e-4 at start, 1e-4 at end. Lets PPO take big steps early
# to find the basic strategy, then small steps late to refine the racing line.
LR_INITIAL = 3e-4
LR_FINAL = 1e-4


def linear_schedule(initial_value: float, final_value: float):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Training env uses random spawn points to densify the finish-bonus signal
    # on long tracks. Eval env uses the deterministic standard start so eval
    # numbers are a consistent baseline ("can the policy do a full lap from rest?").
    env = GymRacer(randomize_start=True)
    eval_env = GymRacer(randomize_start=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINTS_DIR,
        name_prefix="ppo_racer",
        verbose=0,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(LR_INITIAL, LR_FINAL),
        # Default ent_coef=0.0 lets the action std collapse — exploration dies and
        # the policy gets stuck. A small positive bonus keeps the distribution wide.
        ent_coef=0.01,
        seed=42,
        verbose=1,
    )

    # CSV logger writes every metric to disk so we can summarize the run at the end.
    model.set_logger(configure(LOG_DIR, ["stdout", "csv"]))

    model.save(os.path.join(CHECKPOINTS_DIR, "ppo_racer_0_steps"))
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    model.save(os.path.join(MODELS_DIR, "ppo_racer_final"))

    print_run_summary()


def print_run_summary():
    csv_path = os.path.join(LOG_DIR, "progress.csv")
    if not os.path.exists(csv_path):
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    rollout_rows = [r for r in rows if r.get("rollout/ep_rew_mean", "").strip()]
    eval_rows = [r for r in rows if r.get("eval/mean_reward", "").strip()]

    print()
    print("=" * 72)
    print("COPY EVERYTHING BELOW (paste this whole block back in chat):")
    print("=" * 72)
    print()
    print(f"Run: TIME_PENALTY={RacerEnv.TIME_PENALTY}  "
          f"TOTAL_TIMESTEPS={TOTAL_TIMESTEPS:,}  "
          f"LR={LR_INITIAL}->{LR_FINAL}")
    print()

    # Training trajectory (10 evenly-spaced samples across the whole run).
    print("Training rollout trajectory:")
    print(f"  {'step':>10}  {'ep_rew':>10}  {'ep_len':>8}  {'std':>8}  {'entropy':>10}")
    for r in _evenly_spaced(rollout_rows, n=10):
        step = int(float(r.get("time/total_timesteps") or 0))
        print(f"  {step:>10,}  {_safe_float(r, 'rollout/ep_rew_mean'):>10.1f}  "
              f"{_safe_float(r, 'rollout/ep_len_mean'):>8.0f}  "
              f"{_safe_float(r, 'train/std'):>8.3f}  "
              f"{_safe_float(r, 'train/entropy_loss'):>10.3f}")

    print()
    # Eval trajectory (10 evenly-spaced samples).
    print("Eval trajectory:")
    print(f"  {'step':>10}  {'mean_reward':>12}  {'mean_ep_length':>14}")
    for r in _evenly_spaced(eval_rows, n=10):
        step = int(float(r.get("time/total_timesteps") or 0))
        print(f"  {step:>10,}  {_safe_float(r, 'eval/mean_reward'):>12.1f}  "
              f"{_safe_float(r, 'eval/mean_ep_length'):>14.0f}")

    print()
    # Peak/final summaries.
    if rollout_rows:
        best_ro = min(rollout_rows, key=lambda r: _safe_float(r, "rollout/ep_len_mean"))
        best_ro_step = int(float(best_ro.get("time/total_timesteps") or 0))
        best_ro_len = _safe_float(best_ro, "rollout/ep_len_mean")
        print(f"Best rollout lap: {best_ro_len:.0f} steps ({best_ro_len/60:.2f}s) "
              f"at step {best_ro_step:,}")
    if eval_rows:
        best_ev = min(eval_rows, key=lambda r: _safe_float(r, "eval/mean_ep_length"))
        best_ev_step = int(float(best_ev.get("time/total_timesteps") or 0))
        best_ev_len = _safe_float(best_ev, "eval/mean_ep_length")
        best_ev_rew = _safe_float(best_ev, "eval/mean_reward")
        print(f"Best eval lap:    {best_ev_len:.0f} steps ({best_ev_len/60:.2f}s) "
              f"at step {best_ev_step:,}, reward {best_ev_rew:.1f}")

        final_ev = eval_rows[-1]
        final_step = int(float(final_ev.get("time/total_timesteps") or 0))
        final_len = _safe_float(final_ev, "eval/mean_ep_length")
        final_rew = _safe_float(final_ev, "eval/mean_reward")
        print(f"Final eval lap:   {final_len:.0f} steps ({final_len/60:.2f}s) "
              f"at step {final_step:,}, reward {final_rew:.1f}")

    print()
    print(f"Best model:  {BEST_MODEL_DIR}/best_model.zip")
    print(f"Final model: {MODELS_DIR}/ppo_racer_final.zip")
    print(f"Checkpoints: {CHECKPOINTS_DIR}/  ({TOTAL_TIMESTEPS // CHECKPOINT_FREQ + 1} files)")
    print("=" * 72)


def _evenly_spaced(rows, n=10):
    if len(rows) <= n:
        return rows
    return [rows[int(i * (len(rows) - 1) / (n - 1))] for i in range(n)]


def _safe_float(row, key):
    val = row.get(key, "")
    return float(val) if val.strip() else float("nan")


if __name__ == "__main__":
    main()
