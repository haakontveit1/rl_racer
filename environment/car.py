import pygame
from pygame.math import Vector2
from config import *

class Car:
    def __init__(self, x, y):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0, 0)
        self.angle = 90
        
        # Appearance settings
        self.width = 40
        self.height = 20

        # New Physics Constants (we will move these to config later)
        self.acceleration_power = CAR_ACCELERATION
        self.rotation_speed = CAR_ROTATION_SPEED
        self.friction = CAR_FRICTION
        self.grip = CAR_GRIP

    def update(self, keys, on_track):
        if on_track:
            current_friction = self.friction
            current_accel = self.acceleration_power
        else:
            current_friction = GRASS_FRICTION
            current_accel = self.acceleration_power * GRASS_ACCEL_LIMIT
        
        # 1. Handle Turning
        if keys[pygame.K_LEFT]:
            self.angle -= self.rotation_speed
        if keys[pygame.K_RIGHT]:
            self.angle += self.rotation_speed

        # 2. Handle Acceleration
        if keys[pygame.K_UP]:
            forward = Vector2(1, 0).rotate(self.angle)
            self.velocity += forward * current_accel

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


    def draw(self, surface):
        # 1. Create a temporary 'Surface' (a blank canvas) just for the car
        # We use 'SRCALPHA' so the background of the car is transparent
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # 2. Draw a red rectangle onto that tiny car canvas
        pygame.draw.rect(car_surface, RED, (0, 0, self.width, self.height))
        
        # 3. Rotate the car canvas based on our current angle
        # Note: we use negative self.angle because Pygame's rotation is counter-intuitive
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        
        # 4. Get a new 'Rect' for the rotated image so it stays centered
        # Without this, the car would 'wobble' around its top-left corner when turning
        rect = rotated_car.get_rect(center=(self.position.x, self.position.y))
        
        # 5. Finally, 'blit' (paste) the rotated car onto the main screen
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

    