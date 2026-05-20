import numpy as np
from pygame.math import Vector2

from config import FPS, CAR_ROTATION_SPEED
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

    MAX_STEPS = FPS * 15   # 15-second cap, tuned for the new ~6s-optimal track with margin for early-training laps.

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
    FINISH_BONUS = 100.0      # flat reward on crossing the finish line
    FINISH_SPEED_BONUS = 100.0  # additional reward on finishing, scaled by avg speed over the lap.
                                # Only paid on finish — so "drive fast without finishing" gets zero.

    def __init__(self, waypoints=None, randomize_start=False):
        # waypoints=None falls through to Track's own default (the chicane).
        # randomize_start=True: 90% of reset()s spawn the car at a random point
        # on the centerline with random forward speed. Used for training to
        # solve the sparse-finish-bonus problem on long tracks.
        self.track = Track(waypoints=waypoints)
        self.car = Car(*self.track.start_position)
        self.randomize_start = randomize_start
        self.steps = 0
        self.prev_progress = 0.0
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.velocity_sum = 0.0

    def reset(self):
        if self.randomize_start and np.random.random() < 0.9:
            # 90% random spawn during randomized training; 10% keeps the policy
            # familiar with the actual start state (at-rest, slightly behind wp0).
            position, angle, velocity = self._random_spawn()
        else:
            position = Vector2(self.track.start_position)
            angle = self.track.start_angle
            velocity = Vector2(0, 0)

        self.car.position = position
        self.car.velocity = velocity
        self.car.angle = angle
        self.car.angular_velocity = 0.0
        self.car.trajectory = []
        self.steps = 0
        self.prev_progress = self.track.progress_along_centerline(self.car.position)
        self.last_steer = 0.0
        self.last_throttle = 0.0
        self.velocity_sum = 0.0
        return self._observation()

    def _random_spawn(self):
        # Random segment, random t along it, random lateral offset within the
        # track width, random forward speed. Heading aligns with segment
        # direction so the car always points down the track at spawn.
        # Lateral randomization is what teaches the policy that off-centerline
        # positions are usable — without it, the agent learns to hug the centerline.
        waypoints = self.track.waypoints
        segment_idx = np.random.randint(0, len(waypoints) - 1)
        t = np.random.uniform(0.0, 0.95)  # avoid spawning exactly on the finish line
        a = waypoints[segment_idx]
        b = waypoints[segment_idx + 1]
        centerline_point = a + (b - a) * t
        segment_dir = (b - a).normalize()
        # Perpendicular = segment_dir rotated 90°. Lateral offset capped at
        # 0.3 × half_width — wide enough to teach the policy that off-centerline
        # positions are usable, narrow enough that the state distribution doesn't
        # explode and prevent convergence. (0.8 was too wide; PPO never converged.)
        perpendicular = segment_dir.rotate(90)
        lateral_offset = float(np.random.uniform(-0.3, 0.3)) * self.track.half_width
        position = Vector2(centerline_point + perpendicular * lateral_offset)
        angle = segment_dir.as_polar()[1]
        speed = float(np.random.uniform(0.0, 8.0))
        velocity = segment_dir * speed
        return position, angle, velocity

    def step(self, action: Action):
        prev_position = Vector2(self.car.position)
        on_track = self.track.is_on_track(self.car.position)
        self.car.update(action, on_track)
        self.steps += 1
        self.last_steer = action.steer
        self.last_throttle = action.throttle
        self.velocity_sum += self.car.velocity.length()

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
            avg_speed = self.velocity_sum / max(self.steps, 1)
            reward += self.FINISH_BONUS + self.FINISH_SPEED_BONUS * avg_speed

        info = {
            "on_track": on_track,
            "crossed_finish": crossed_finish,
            "timed_out": timed_out,
            "progress": progress,
        }
        return self._observation(), reward, done, info

    def _observation(self):
        # 11 dims, matched by GymRacer.observation_space.
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
        angular_velocity = self.car.angular_velocity / CAR_ROTATION_SPEED
        lateral_offset = self.track.signed_lateral_offset(self.car.position) / self.track.half_width
        return (
            *rays,
            forward_speed,
            lateral_speed,
            self.last_steer,
            self.last_throttle,
            angular_velocity,
            lateral_offset,
        )

    def get_ray_endpoints(self):
        # Debug helper: returns [(start, end), ...] in world coords for drawing.
        endpoints = []
        for a in self.RAY_ANGLES:
            direction = Vector2(1, 0).rotate(self.car.angle + a)
            distance = self.track.cast_ray(self.car.position, direction, self.RAY_MAX_DISTANCE)
            endpoints.append((Vector2(self.car.position), self.car.position + direction * distance))
        return endpoints
