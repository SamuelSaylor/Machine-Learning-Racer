import pygame
import sys
from HELPERS.racecar import RaceCar
import os

pygame.init()
pygame.mixer.quit() #stop the sounds

WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Race Car")
clock = pygame.time.Clock()

# --- Load Images ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(BASE_DIR, 'ASSETS', 'CARS', 'BlueRacer.png')
background_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', 'BUDAPEST', 'COSMETIC.png')
car_img = pygame.image.load(image_path).convert_alpha()
car_img = pygame.transform.scale(car_img, (40, 55)) #change size based on what we need
#track_img = pygame.image.load(os.path.join(track_path, "BOUNDARY.png")).convert_alpha()
#deadzone_img = pygame.image.load(os.path.join(track_path, "DEADZONE.png")).convert_alpha()
background_img = pygame.image.load(background_path).convert()


car = RaceCar()

running = True

car_width, car_height = 40, 20    
    
while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    input_accel = 0
    input_dir = 0

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        input_accel = 1
    elif keys[pygame.K_DOWN]:
        input_accel = -1

    if keys[pygame.K_LEFT]:
        input_dir = -1
    elif keys[pygame.K_RIGHT]:
        input_dir = 1

    car.update(input_accel, input_dir, dt)

    screen.fill((30, 30, 30))

    rotated_car = pygame.transform.rotate(car_img, car.angle)
    car_rect = rotated_car.get_rect(center=(car.car_pos[0], car.car_pos[1]))

    screen.blit(background_img, (0, 0))
    screen.blit(rotated_car, car_rect.topleft)

    pygame.display.flip()

pygame.quit()
sys.exit()