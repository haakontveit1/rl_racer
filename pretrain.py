"""Behavior cloning: pretrain the PPO policy on human demonstration laps.

Reads all .npz files from demos/ (recorded by main.py with R toggled on),
trains the policy network via supervised MSE on the recorded actions, and
saves a checkpoint that train.py picks up automatically.

The point: PPO from scratch keeps converging to the rush-and-tangent local
optimum. By initializing the policy near the human racing line, fine-tuning
PPO has a much better starting basin.
"""

import glob
import os

import numpy as np
import torch

from stable_baselines3 import PPO

from gym_env import GymRacer


DEMOS_DIR = "demos"
PRETRAINED_PATH = "models/pretrained.zip"
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3


def load_demos():
    files = sorted(glob.glob(os.path.join(DEMOS_DIR, "*.npz")))
    if not files:
        raise SystemExit(f"No demos in {DEMOS_DIR}/. Record laps via main.py (press R).")

    obs_list, act_list = [], []
    for f in files:
        data = np.load(f)
        obs_list.append(data["obs"])
        act_list.append(data["actions"])
    obs = np.concatenate(obs_list, axis=0).astype(np.float32)
    acts = np.concatenate(act_list, axis=0).astype(np.float32)
    print(f"Loaded {len(files)} demos: {obs.shape[0]} frames total")
    return obs, acts


def main():
    obs_np, acts_np = load_demos()

    env = GymRacer()
    # Match train.py's PPO config so the saved model is plug-compatible.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        ent_coef=0.01,
        seed=42,
        verbose=0,
    )

    policy = model.policy
    device = policy.device

    obs_t = torch.from_numpy(obs_np).to(device)
    acts_t = torch.from_numpy(acts_np).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    n = obs_t.shape[0]
    print(f"Pretraining: {n} samples, {EPOCHS} epochs, batch size {BATCH_SIZE}")

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            obs_batch = obs_t[idx]
            acts_batch = acts_t[idx]

            # deterministic=True returns the policy's mean action (Gaussian mean),
            # which is what we want to align with the demo actions. log_std and
            # value_net are not in this graph, so they won't be updated.
            pred_actions, _, _ = policy(obs_batch, deterministic=True)
            loss = ((pred_actions - acts_batch) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1:3d}/{EPOCHS}  avg MSE: {avg_loss:.6f}")

    # Final fit metrics over the whole dataset.
    with torch.no_grad():
        pred_all, _, _ = policy(obs_t, deterministic=True)
        final_mse = ((pred_all - acts_t) ** 2).mean().item()
        final_mae = (pred_all - acts_t).abs().mean().item()
        steer_mae = (pred_all[:, 0] - acts_t[:, 0]).abs().mean().item()
        throttle_mae = (pred_all[:, 1] - acts_t[:, 1]).abs().mean().item()

    print()
    print(f"Final MSE:      {final_mse:.6f}")
    print(f"Final MAE:      {final_mae:.6f}  (avg action error)")
    print(f"  steer MAE:    {steer_mae:.6f}  (range [-1, 1])")
    print(f"  throttle MAE: {throttle_mae:.6f}  (range [0, 1])")

    os.makedirs(os.path.dirname(PRETRAINED_PATH), exist_ok=True)
    model.save(PRETRAINED_PATH)
    print()
    print(f"Saved: {PRETRAINED_PATH}")
    print("Run train.py — it will detect this checkpoint and continue from it.")


if __name__ == "__main__":
    main()
