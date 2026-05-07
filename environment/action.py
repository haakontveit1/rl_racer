from dataclasses import dataclass
import pygame


@dataclass
class Action:
    steer: float = 0.0      # -1.0 = full left, +1.0 = full right, 0.0 = straight
    throttle: float = 0.0   #  0.0 = coast,     1.0 = full gas

    @classmethod
    def from_keys(cls, keys) -> "Action":
        steer = 0.0
        if keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_RIGHT]:
            steer += 1.0
        throttle = 1.0 if keys[pygame.K_UP] else 0.0
        return cls(steer=steer, throttle=throttle)
