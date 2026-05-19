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

    MAX_STEPS = FPS * 10   # 10-second episode time cap (was 60s — too long; catastrophic timeouts dominated training variance)

    # Raycast layout — angles are relative to the car's heading (0 = straight ahead).
    RAY_ANGLES = (-90, -45, 0, 45, 90)
    RAY_MAX_DISTANCE = 300
    # Speed normalizer — divides forward/lateral speed so observations sit roughly in [-1, 1].
    SPEED_NORM = 10.0

    # Reward shaping.
    # No explicit off-track penalty — grass physics (terminal speed ~1.3 vs 10
    # on track) already punishes off-track heavily and consistently.
    OFF_TRACK_PENALTY = 0.0
    TIME_PENALTY = 3.0        # per step always — strictly: faster lap = higher reward.
    FINISH_BONUS = 100.0      # once on crossing the finish line

    def __init__(self):
        self.track = Track()
        self.car = Car(*self.track.start_position)
        self.steps = 0
        self.prev_progress = 0.0

    def reset(self):
        self.car.position = Vector2(self.track.start_position)
        self.car.velocity = Vector2(0, 0)
        self.car.angle = self.track.start_angle
        self.car.trajectory = []
        self.steps = 0
        self.prev_progress = self.track.progress_along_centerline(self.car.position)
        return self._observation()

    def step(self, action: Action):
        prev_position = Vector2(self.car.position)
        on_track = self.track.is_on_track(self.car.position)
        self.car.update(action, on_track)
        self.steps += 1

        # Dense reward: progress made along the centerline this step.
        progress = self.track.progress_along_centerline(self.car.position)
        reward = progress - self.prev_progress
        self.prev_progress = progress

        if not on_track:
            reward -= self.OFF_TRACK_PENALTY
        reward -= self.TIME_PENALTY

        crossed_finish = self.track.crossed_finish(prev_position, self.car.position)
        timed_out = self.steps >= self.MAX_STEPS
        done = crossed_finish or timed_out

        if crossed_finish:
            reward += self.FINISH_BONUS

        info = {
            "on_track": on_track,
            "crossed_finish": crossed_finish,
            "timed_out": timed_out,
            "progress": progress,
        }
        return self._observation(), reward, done, info

    def _observation(self):
        # 5 raycast distances (normalized) + forward speed + lateral speed (car frame).
        # This is the full state a policy needs to drive locally.
        forward_dir = Vector2(1, 0).rotate(self.car.angle)
        right_dir = Vector2(0, 1).rotate(self.car.angle)

        rays = tuple(
            self.track.cast_ray(
                self.car.position,
                Vector2(1, 0).rotate(self.car.angle + a),
                self.RAY_MAX_DISTANCE,
            ) / self.RAY_MAX_DISTANCE
            for a in self.RAY_ANGLES
        )
        forward_speed = self.car.velocity.dot(forward_dir) / self.SPEED_NORM
        lateral_speed = self.car.velocity.dot(right_dir) / self.SPEED_NORM
        return (*rays, forward_speed, lateral_speed)

    def get_ray_endpoints(self):
        # Debug helper: returns [(start, end), ...] in world coords for drawing.
        endpoints = []
        for a in self.RAY_ANGLES:
            direction = Vector2(1, 0).rotate(self.car.angle + a)
            distance = self.track.cast_ray(self.car.position, direction, self.RAY_MAX_DISTANCE)
            endpoints.append((Vector2(self.car.position), self.car.position + direction * distance))
        return endpoints
