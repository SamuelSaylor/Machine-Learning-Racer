import math

class RaceCar:
    def __init__(self, x,y, angle):
        self.car_pos = [x,y]
        self.angle = angle 
        self.speed = 0.0

        self.max_speed = 150
        self.acceleration = 100
        self.breaking = -200
        self.turn_speed = 150

    def update(self, input_accel, input_dir, dt, friction):
        # update speed 
        if input_accel > 0:
            self.speed += self.acceleration * input_accel * dt
        elif input_accel < 0:
            self.speed += self.breaking * (-input_accel) * dt
        else:
            if self.speed > 0:
                self.speed -= friction * dt
                if self.speed < 0:
                    self.speed = 0
            elif self.speed < 0:
                self.speed += friction * dt
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
