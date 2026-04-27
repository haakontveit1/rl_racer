import pygame
from config import *

class Track:
    def __init__(self):

        self.segments = [
            pygame.Rect(100, 100, 200, 600), # Vertical segment
            pygame.Rect(100, 500, 700, 200)  # Horizontal segment
        ]

        self.start_line = pygame.Rect(150, 150, 100, 10)
        self.finish_line = pygame.Rect(750, 550, 10, 100)

    def draw(self, surface):
        # 1. Draw the main track segments
        for segment in self.segments:
            pygame.draw.rect(surface, GRAY, segment)
            
        # 2. Draw the Start and Finish lines
        pygame.draw.rect(surface, WHITE, self.start_line) # Start is White
        pygame.draw.rect(surface, BLACK, self.finish_line) # Finish is Black

    def is_on_track(self, car_pos):
        # We loop through every gray segment in our track
        for segment in self.segments:
            # collidepoint checks if the (x, y) is inside the rectangle
            if segment.collidepoint(car_pos.x, car_pos.y):
                return True # As soon as we find one match, we are 'on track'
        
        # If the loop finishes and we haven't found a match, we are in the mud
        return False