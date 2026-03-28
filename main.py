import pygame
import sys
from HELPERS.racecar import RaceCar
import os
import pandas as pd
import numpy as np

pygame.init()
pygame.mixer.quit() #stop the sounds


# --- Get the track information --- #
df = pd.read_csv('ASSETS/DATA/track_data.csv')
track_info = df.loc[df['trackname'] == 'Budapest']

# --- Set up Screen --- #
WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(track_info['trackname'].iloc[0])
clock = pygame.time.Clock()

# --- Load Images --- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#uncomment the other images to see the boundaries : )

image_path = os.path.join(BASE_DIR, 'ASSETS', 'CARS', 'BlueRacer.png')
background_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', track_info['dirname'].iloc[0], 'COSMETIC.png')
track_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', 'BUDAPEST')
deadzone_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', 'BUDAPEST')
car_img = pygame.image.load(image_path).convert_alpha()
car_img = pygame.transform.scale(car_img, (20, 32)) #change size based on what we need
track_img = pygame.image.load(os.path.join(track_path, "BOUNDARY.png")).convert_alpha()
track_img = pygame.transform.scale(track_img, (1000,1000))
deadzone_img = pygame.image.load(os.path.join(track_path, "DEADZONE.png")).convert_alpha()
deadzone_img = pygame.transform.scale(deadzone_img, (1000,1000))
background_img = pygame.image.load(background_path).convert()
background_img = pygame.transform.scale(background_img,(1000,1000))


# --- Create the masks ---- #
track_mask = pygame.mask.from_surface(track_img) 
boundary_mask = pygame.mask.from_surface(deadzone_img) 
car_mask = pygame.mask.from_surface(car_img)
#who are we?
car = RaceCar(track_info['startingcoordx'].iloc[0], track_info['startingcordy'].iloc[0], track_info['angle'].iloc[0])

running = True

    
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

    

    screen.fill((30, 30, 30))

    rotated_car = pygame.transform.rotate(car_img, car.angle)
    car_rect = rotated_car.get_rect(center=(car.car_pos[0], car.car_pos[1]))
    car_mask = pygame.mask.from_surface(rotated_car)
    
    #collission offset
    offset = (car_rect.left, car_rect.top)
    
    if boundary_mask.overlap(car_mask, offset):
        car.update(input_accel, input_dir, dt,2)
        print("Boundary hit!")
    elif not track_mask.overlap(car_mask, offset):
        car.update(input_accel, input_dir, dt,1)
        print("Off track!")
    else: 
        car.update(input_accel, input_dir, dt,0)

    screen.blit(background_img, (0, 0))
    screen.blit(track_img, (0, 0)) #uncomment to see boundaries
    screen.blit(deadzone_img, (0, 0))
    screen.blit(rotated_car, car_rect.topleft)

    pygame.display.flip()

pygame.quit()
sys.exit()