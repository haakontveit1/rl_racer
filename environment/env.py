from pygame.math import Vector2

from config import FPS
from environment.track import Track
from environment.car import Car
from environment.action import Action


class RacerEnv:
    """Gym-style wrapper around Track + Car. Owns the simulation, not the rendering.

    Training loop:
        env = RacerEnv()
        obs = env.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
    """

    START_POS = (200, 130)
    START_ANGLE = 90
    MAX_STEPS = FPS * 60   # 60-second episode time cap

    def __init__(self):
        self.track = Track()
        self.car = Car(*self.START_POS)
        self.steps = 0

    def reset(self):
        self.car.position = Vector2(*self.START_POS)
        self.car.velocity = Vector2(0, 0)
        self.car.angle = self.START_ANGLE
        self.car.trajectory = []
        self.steps = 0
        return self._observation()

    def step(self, action: Action):
        on_track = self.track.is_on_track(self.car.position)
        self.car.update(action, on_track)
        self.steps += 1

        # Placeholder reward — produces a "sit still on track" pathology if
        # trained against. Replaced with progress-along-centerline later.
        reward = 1.0 if on_track else 0.0

        crossed_finish = self.car.get_rect().colliderect(self.track.finish_line)
        timed_out = self.steps >= self.MAX_STEPS
        done = crossed_finish or timed_out

        info = {
            "on_track": on_track,
            "crossed_finish": crossed_finish,
            "timed_out": timed_out,
        }
        return self._observation(), reward, done, info

    def _observation(self):
        # Placeholder observation — a real policy can't drive on this alone.
        # Step 2 replaces this with raycasts + heading + lateral speed.
        forward_dir = Vector2(1, 0).rotate(self.car.angle)
        forward_speed = self.car.velocity.dot(forward_dir)
        return (forward_speed,)
