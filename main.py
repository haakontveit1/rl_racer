import pygame
import sys
import os

import numpy as np

from config import *
from environment.action import Action
from environment.env import RacerEnv

def main():
    # 1. Start the Pygame engines
    pygame.init()

    # 2. Create the window (the "Canvas")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RL Race Track")

    history_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    font = pygame.font.SysFont("Arial", 24, bold=True)

    # 3. The Clock (controls the speed of the game)
    clock = pygame.time.Clock()

    # 4. Build the simulation environment. A human controller will feed it
    #    Actions from the keyboard; an RL policy would feed it Actions
    #    from a forward pass on env.reset()'s observation.
    env = RacerEnv()
    obs = np.array(env.reset(), dtype=np.float32)

    os.makedirs("demos", exist_ok=True)
    recording = False
    recording_buffer = []
    demo_count = 0

    start_time = pygame.time.get_ticks()

    episode_reward = 0.0
    last_step_reward = 0.0

    screenshot_count = 0
    running = True
    while running:
        # A. Event Handling (Check for clicks/quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # This triggers exactly ONCE when the key is first pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    # Build the path: "screenshots/lap_0.png"
                    filename = os.path.join("screenshots", f"lap_{screenshot_count}.png")

                    # Save the screen to that specific path
                    pygame.image.save(screen, filename)

                    print(f"Saved screenshot to {filename}")
                    screenshot_count += 1

                if event.key == pygame.K_r:
                    recording = not recording
                    if not recording:
                        recording_buffer = []
                    print(f"Recording: {'ON' if recording else 'OFF'}")
        


        # 1. Capture the Keys (The Brain)
        keys = pygame.key.get_pressed()

        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000

        # Translate keyboard state into a generic action, then step the env.
        # An RL policy will skip the keyboard and construct Action(...) directly,
        # then call env.step(action) the same way we do here.
        action = Action.from_keys(keys)
        if recording:
            recording_buffer.append((
                obs.copy(),
                np.array([action.steer, action.throttle], dtype=np.float32),
            ))
        obs_tuple, reward, done, info = env.step(action)
        obs = np.array(obs_tuple, dtype=np.float32)
        last_step_reward = reward
        episode_reward += reward

        if info["crossed_finish"]:
            # Bake the completed lap's trail into the persistent history layer
            if len(env.car.trajectory) > 1:
                pygame.draw.lines(history_surface, (0, 255, 255, 50), False, env.car.trajectory, 1)
            print(f"Finish Line Reached at {elapsed_time:.2f}s! Episode reward: {episode_reward:.1f}")
            if recording and recording_buffer:
                demo_count += 1
                obs_arr = np.stack([p[0] for p in recording_buffer])
                act_arr = np.stack([p[1] for p in recording_buffer])
                fname = f"demos/lap_{demo_count:03d}_{elapsed_time:.2f}s.npz"
                np.savez(fname, obs=obs_arr, actions=act_arr)
                print(f"Saved demo: {fname}  ({len(recording_buffer)} frames)")
                recording_buffer = []

        if done:
            # Episode boundary — finish line OR time cap. reset() restores the
            # car *and* clears the trajectory so no jump line can form.
            if recording and recording_buffer:
                print(f"Discarded {len(recording_buffer)} frames (didn't finish)")
            recording_buffer = []
            obs = np.array(env.reset(), dtype=np.float32)
            start_time = pygame.time.get_ticks()
            episode_reward = 0.0
            last_step_reward = 0.0

        # B. Fill the background (Wipe the canvas clean)
        screen.fill(GREEN) # Our grass/mud

        # Layer order: grass -> track -> centerline -> historical trails -> car -> debug rays on top
        env.track.draw(screen)
        pygame.draw.lines(screen, (255, 0, 255), False, env.track.centerline, 2)
        screen.blit(history_surface, (0, 0))
        env.car.draw(screen)

        # Debug: draw the raycasts so we can see what the policy will "see".
        for start, end in env.get_ray_endpoints():
            pygame.draw.line(screen, (255, 255, 0), start, end, 1)
            pygame.draw.circle(screen, (255, 255, 0), (int(end.x), int(end.y)), 3)

        timer_text = font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))
        ep_reward_text = font.render(f"Episode reward: {episode_reward:.1f}", True, (255, 255, 255))
        step_reward_text = font.render(f"Step reward: {last_step_reward:+.2f}", True, (255, 255, 255))
        screen.blit(timer_text, (10, 10))
        screen.blit(ep_reward_text, (10, 40))
        screen.blit(step_reward_text, (10, 70))

        if recording:
            rec_text = font.render(f"● REC  (saved: {demo_count})", True, (255, 50, 50))
            screen.blit(rec_text, (10, 100))
        elif demo_count > 0:
            rec_text = font.render(f"Recording off  (saved: {demo_count})", True, (180, 180, 180))
            screen.blit(rec_text, (10, 100))

        # C. Update the display (Draw the new frame)
        pygame.display.flip()

        
        # D. Maintain 60 Frames Per Second
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()