import pygame
import sys

from config import *
from environment.track import Track
from environment.car import Car

def main():
    # 1. Start the Pygame engines
    pygame.init()
    
    # 2. Create the window (the "Canvas")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("RL Race Track")
    
    # 3. The Clock (controls the speed of the game)
    clock = pygame.time.Clock()

    # 4. Instantiate: Create one 'instance' of our Track
    track = Track()

    car = Car(200,130)
    
    running = True
    while running:
        # A. Event Handling (Check for clicks/quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 1. Capture the Keys (The Brain)
        keys = pygame.key.get_pressed()
        
        # 2. Update the Car (The Muscle)
        on_track = track.is_on_track(car.position)
        
        # Update the car and pass in the on_track status
        car.update(keys,on_track)

        if car.get_rect().colliderect(track.finish_line):
            print("Finish Line Reached!")
            
            # 2. Reset the car to the starting position
            # In RL, this is the end of the 'Episode'
            car.position = pygame.math.Vector2(200, 130)
            car.velocity = pygame.math.Vector2(0, 0)
            car.angle = 90

        # B. Fill the background (Wipe the canvas clean)
        screen.fill(GREEN) # Our grass/mud

        # 3. Draw: Put the track on top of the grass
        track.draw(screen)

        # 3. Draw: Put the track on top of the grass
        car.draw(screen)

        # C. Update the display (Draw the new frame)
        pygame.display.flip()

        
        # D. Maintain 60 Frames Per Second
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()