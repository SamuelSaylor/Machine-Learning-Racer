import math

class RaceCar:
    def __init__(self):
        self.car_pos = [200.0, 300.0]
        self.health = 100
        self.angle = 0.0
        self.speed = 0.0

        self.max_speed = 300
        self.acceleration = 200
        self.max_deacceleration = -100
        self.friction = 150
        self.turn_speed = 120

    def update(self, input_accel, input_dir, dt):
        # update speed
        if input_accel > 0:
            self.speed += self.acceleration * input_accel * dt
        elif input_accel < 0:
            self.speed += self.max_deacceleration * (-input_accel) * dt
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
            self.angle += input_dir * self.turn_speed * dt

        # update position
        rad = math.radians(self.angle)
        self.car_pos[0] += math.cos(rad) * self.speed * dt
        self.car_pos[1] += math.sin(rad) * self.speed * dt
