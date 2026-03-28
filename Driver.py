import pygame
import sys
from racecar import RaceCar

pygame.init()

WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Race Car")
clock = pygame.time.Clock()

car = RaceCar()

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

    car.update(input_accel, input_dir, dt)

    screen.fill((30, 30, 30))

    car_width, car_height = 40, 20
    car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
    car_surface.fill((255, 0, 0))

    rotated_car = pygame.transform.rotate(car_surface, -car.angle)
    rect = rotated_car.get_rect(center=(car.car_pos[0], car.car_pos[1]))
    screen.blit(rotated_car, rect.topleft)

    pygame.display.flip()

pygame.quit()
sys.exit()