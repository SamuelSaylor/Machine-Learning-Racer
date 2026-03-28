##################################################################################
ENABLE_GAME = True
# Set to FALSE if you only wish to watch the model training LIVE!
##################################################################################

import pygame
import sys
from HELPERS.racecar import RaceCar
import os
import pandas as pd

pygame.init()
pygame.mixer.quit()  # stop the sounds

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

# NPC car image
npc_car_img = pygame.image.load(os.path.join(BASE_DIR, 'ASSETS', 'CARS', 'BlueRacer.png')).convert_alpha()
npc_car_img = pygame.transform.scale(npc_car_img, (20, 32))

# Player car image
player_car_img = pygame.image.load(os.path.join(BASE_DIR, 'ASSETS', 'CARS', 'RedRacer.png')).convert_alpha()
player_car_img = pygame.transform.scale(player_car_img, (20, 32))

# Track / Deadzone / Background
track_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', 'BUDAPEST')
track_img = pygame.image.load(os.path.join(track_path, "BOUNDARY.png")).convert_alpha()
track_img = pygame.transform.scale(track_img, (1000, 1000))
deadzone_img = pygame.image.load(os.path.join(track_path, "DEADZONE.png")).convert_alpha()
deadzone_img = pygame.transform.scale(deadzone_img, (1000, 1000))
background_path = os.path.join(BASE_DIR, 'ASSETS', 'TRACKS', track_info['dirname'].iloc[0], 'COSMETIC.png')
background_img = pygame.image.load(background_path).convert()
background_img = pygame.transform.scale(background_img, (1000, 1000))

# --- Create Masks --- #
track_mask = pygame.mask.from_surface(track_img)
boundary_mask = pygame.mask.from_surface(deadzone_img)

# --- Create Cars --- #
npc_car = RaceCar(track_info['startingcoordx'].iloc[0],
                  track_info['startingcoordy'].iloc[0],
                  track_info['angle'].iloc[0])

player = None
if ENABLE_GAME:
    player = RaceCar(track_info['startingcoordx'].iloc[0],
                     track_info['startingcoordy'].iloc[0],
                     track_info['angle'].iloc[0])

# --- Main Loop --- #
running = True
while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Player input ---
    pinput_accel = 0
    pinput_dir = 0
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        pinput_accel = 1
    elif keys[pygame.K_s]:
        pinput_accel = -1
    if keys[pygame.K_a]:
        pinput_dir = -1
    elif keys[pygame.K_d]:
        pinput_dir = 1

    # --- AI input ---
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

    # --- Draw background ---
    screen.blit(background_img, (0, 0))
    #screen.blit(track_img, (0, 0))
    #screen.blit(deadzone_img, (0, 0))

    # --- NPC car update ---
    rotated_npc = pygame.transform.rotate(npc_car_img, npc_car.angle)
    npc_rect = rotated_npc.get_rect(center=(npc_car.car_pos[0], npc_car.car_pos[1]))
    npc_mask = pygame.mask.from_surface(rotated_npc)
    npc_offset = (npc_rect.left, npc_rect.top)
    
    npc_x = int(npc_car.car_pos[0])
    npc_y = int(npc_car.car_pos[1])
    
    npc_on_track = False
    if 0 <= npc_x < WIDTH and 0 <= npc_y < HEIGHT:
        npc_on_track = track_mask.get_at((npc_x, npc_y))

    if boundary_mask.overlap(npc_mask, npc_offset):
        npc_car.update(input_accel, input_dir, dt, 2)  # crash
    elif not npc_on_track:
        npc_car.update(input_accel, input_dir, dt, 1)  # off track
        #print("NPC off track!")
    else:
        npc_car.update(input_accel, input_dir, dt, 0)  # on track
        #print("NPC on track!")

    # --- Player car update ---
    if player:
        rotated_player = pygame.transform.rotate(player_car_img, player.angle)
        player_rect = rotated_player.get_rect(center=(player.car_pos[0], player.car_pos[1]))
        player_mask = pygame.mask.from_surface(rotated_player)
        player_offset = (player_rect.left, player_rect.top)
        friction = 0 if track_mask.overlap(player_mask, player_offset) else 1
        player.update(pinput_accel, pinput_dir, dt, friction)
        screen.blit(rotated_player, player_rect.topleft)
    screen.blit(rotated_npc, npc_rect.topleft)
    pygame.display.flip()

pygame.quit()
sys.exit()