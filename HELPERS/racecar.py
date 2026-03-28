import math
import pandas as pd
import random
import os

class RaceCar:
    def __init__(self, x, y, angle):
        self.data = self.get_data()
        self.name = self.data['car_name']
        self.car_pos = [x, y]
        self.angle = angle
        self.speed = 0.0
        self.friction = 150
        self.max_speed = self.data['max_speed']
        self.acceleration = self.data['acceleration']
        self.braking = self.data['braking']
        self.turn_speed = self.data['turn_speed']

    def get_data(self):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'ASSETS', 'DATA', 'car_data.csv')
        df = pd.read_csv(csv_path)
        index = random.randint(0, len(df) - 1)
        return df.iloc[index]


    def update(self, input_accel, input_dir, dt, track):
        if track == 1:
            self.max_speed = 100
        elif track == 2:
            self.max_speed = 0
            print('car_broke')
        else:
            self.max_speed = self.data['max_speed']
            
        # update speed 
        if input_accel > 0:
            self.speed += self.acceleration * input_accel * dt
        elif input_accel < 0:
            self.speed += self.braking * (-input_accel) * dt
        else:
            if self.speed > 0:
                self.speed -= self.friction * dt
                if self.speed < 0:
                    self.speed = 0
            elif self.speed < 0:
                self.speed += self.friction * dt
                if self.speed > 0:
                    self.speed = 0

        # clamp speed
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_speed / 2:
            self.speed = -self.max_speed / 2

        # update angle
        if self.speed != 0:          
            self.angle += -input_dir * self.turn_speed * dt

        # update position
        rad = math.radians(-self.angle)
        self.car_pos[0] += math.sin(rad) * self.speed * dt
        self.car_pos[1] -= math.cos(rad) * self.speed * dt
