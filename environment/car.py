import pygame
from pygame.math import Vector2
from config import *

class Car:
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0, 0)
        self.angle = 90
        self.angular_velocity = 0.0
        
        # Appearance settings
        self.width = 40
        self.height = 20

        # New Physics Constants (we will move these to config later)
        self.acceleration_power = CAR_ACCELERATION
        self.rotation_speed = CAR_ROTATION_SPEED
        self.friction = CAR_FRICTION
        self.grip = CAR_GRIP

        self.trajectory = []

    def update(self, action, on_track):
        if on_track:
            current_friction = self.friction
            current_accel = self.acceleration_power
        else:
            current_friction = GRASS_FRICTION
            current_accel = self.acceleration_power * GRASS_ACCEL_LIMIT

        # 1. Handle Turning — first-order lag (rotational inertia).
        # Steer commands a target angular velocity; the actual angular velocity
        # eases toward it. Set ANGULAR_RESPONSIVENESS=1.0 to recover the old
        # instant-rotation behavior.
        target_omega = action.steer * self.rotation_speed
        self.angular_velocity += (target_omega - self.angular_velocity) * ANGULAR_RESPONSIVENESS
        self.angle += self.angular_velocity

        # 2. Handle Acceleration
        if action.throttle > 0:
            forward = Vector2(1, 0).rotate(self.angle)
            self.velocity += forward * current_accel * action.throttle

        # 3. Apply Physics
        self.velocity *= current_friction  # Slowly slows down
        
        # 2. Kill Lateral Velocity (The "Grip" logic)
        # Find the forward and right-hand directions relative to the car
        forward_dir = Vector2(1, 0).rotate(self.angle)
        right_dir = Vector2(0, 1).rotate(self.angle)
        forward_velocity = self.velocity.dot(forward_dir)
        lateral_velocity = self.velocity.dot(right_dir)
        lateral_velocity *= (1 - self.grip)
        self.velocity = (forward_dir * forward_velocity) + (right_dir * lateral_velocity)
        self.position += self.velocity

        self.trajectory.append(Vector2(self.position))


    def draw(self, surface):
        # 1. Draw the current active trail (the breadcrumbs for THIS lap)
        if len(self.trajectory) > 1:
            pygame.draw.lines(surface, (0, 255, 255), False, self.trajectory, 2)
        
        # 2. Create the car's physical body
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, RED, (0, 0, self.width, self.height))
        
        # 3. Handle rotation and centering
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated_car.get_rect(center=(self.position.x, self.position.y))
        
        # 4. Paste the car onto the screen
        surface.blit(rotated_car, rect)

    def get_rect(self):
        # This returns a Rect centered at our current position
        # We use this for collision detection
        return pygame.Rect(
            self.position.x - self.width / 2,
            self.position.y - self.height / 2,
            self.width,
            self.height
        )

    