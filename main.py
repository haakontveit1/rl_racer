import pygame
import sys
import os

from config import *
from environment.track import Track
from environment.car import Car
from environment.action import Action

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

    # 4. Instantiate: Create one 'instance' of our Track
    track = Track()

    car = Car(200,130)
    
    start_time = pygame.time.get_ticks()


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
        


        # 1. Capture the Keys (The Brain)
        keys = pygame.key.get_pressed()
        
        # 2. Update the Car (The Muscle)
        on_track = track.is_on_track(car.position)
        
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000

        # Translate keyboard state into a generic action, then drive the car.
        # An RL policy will skip the keyboard and construct Action(...) directly.
        action = Action.from_keys(keys)
        car.update(action, on_track)

        # Paint the path onto the history surface forever


        if car.get_rect().colliderect(track.finish_line):
            if len(car.trajectory) > 1:
                pygame.draw.lines(history_surface, (0, 255, 255, 50), False, car.trajectory, 1)

            # CLEAR the memory immediately so no "jump line" can form
            car.trajectory = []

            print(f"Finish Line Reached at {elapsed_time}!")
            start_time = pygame.time.get_ticks()
            # 2. Reset the car to the starting position
            # In RL, this is the end of the 'Episode'
            car.position = pygame.math.Vector2(200, 130)
            car.velocity = pygame.math.Vector2(0, 0)
            car.angle = 90

        # B. Fill the background (Wipe the canvas clean)
        screen.fill(GREEN) # Our grass/mud

        # 3. Draw: Put the track on top of the grass
        track.draw(screen)

        screen.blit(history_surface, (0, 0))

        # 3. Draw: Put the track on top of the grass
        car.draw(screen)

        timer_text = font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))
        screen.blit(timer_text, (10, 10))

        # C. Update the display (Draw the new frame)
        pygame.display.flip()

        
        # D. Maintain 60 Frames Per Second
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()