"""Replay a trained PPO policy in the pygame viewer.

Usage:
    python play_model.py
    python play_model.py models/checkpoints/ppo_racer_260000_steps.zip
    python play_model.py --stochastic                # sample actions instead of deterministic

Watch the agent drive so you can see what it's actually doing wrong:
oversteering, cutting corners into grass, stalling, etc. Lap time is
measured in simulation steps / FPS so it matches the training summary.
"""

import argparse
import os
import sys

import pygame
from stable_baselines3 import PPO

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, GREEN
from gym_env import GymRacer


DEFAULT_MODEL = "models/best/best_model.zip"


def main():
    parser = argparse.ArgumentParser(description="Replay a trained PPO policy.")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Path to a model .zip (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of taking the mean action.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    model = PPO.load(args.model_path)
    gym_env = GymRacer()
    # Underlying RacerEnv — the gym wrapper doesn't expose track/car for rendering.
    env = gym_env.env

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"RL Racer — {os.path.basename(args.model_path)}")
    history_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    font = pygame.font.SysFont("Arial", 22, bold=True)
    clock = pygame.time.Clock()

    obs, _ = gym_env.reset()
    episode_reward = 0.0
    last_reward = 0.0
    lap_count = 0
    best_lap_steps = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        episode_reward += reward
        last_reward = reward

        elapsed = env.steps / FPS

        if info["crossed_finish"]:
            if len(env.car.trajectory) > 1:
                pygame.draw.lines(history_surface, (0, 255, 255, 80), False, env.car.trajectory, 1)
            lap_count += 1
            lap_steps = env.steps
            if best_lap_steps is None or lap_steps < best_lap_steps:
                best_lap_steps = lap_steps
            print(f"Lap {lap_count}: {lap_steps} steps ({lap_steps / FPS:.2f}s)  "
                  f"reward={episode_reward:.1f}")

        if terminated or truncated:
            obs, _ = gym_env.reset()
            episode_reward = 0.0
            last_reward = 0.0

        screen.fill(GREEN)
        env.track.draw(screen)
        pygame.draw.lines(screen, (255, 0, 255), False, env.track.centerline, 2)
        screen.blit(history_surface, (0, 0))
        env.car.draw(screen)

        for start, end in env.get_ray_endpoints():
            pygame.draw.line(screen, (255, 255, 0), start, end, 1)
            pygame.draw.circle(screen, (255, 255, 0), (int(end.x), int(end.y)), 3)

        hud = [
            f"Model: {os.path.basename(args.model_path)}",
            f"Mode: {'stochastic' if args.stochastic else 'deterministic'}",
            f"Time: {elapsed:.2f}s",
            f"Episode reward: {episode_reward:.1f}",
            f"Step reward: {last_reward:+.2f}",
            f"Laps: {lap_count}",
        ]
        if best_lap_steps is not None:
            hud.append(f"Best lap: {best_lap_steps / FPS:.2f}s ({best_lap_steps} steps)")
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (255, 255, 255)), (10, 10 + i * 26))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
