
import math

class PID_angle(object):

    def __init__(self, p=0, i=0, d=0, target=0, time=0, angle_max=1):
        self.p = p
        self.i = i
        self.d = d
        self.target = 0
        self.prev_time = time 
        self.prev_error = 0
        self.error = None
        self.angle_max = angle_max
        self.integral = 0

    def angle_diff(self, a1, a2, a_max):
        #limitiert den Winkel auf einen +/-halben Kreis, z.B. auf max. -180°..180°
        diff = a2 - a1
        half = a_max / 2
        return (diff + half ) % a_max - half

    def __call__(self, feedback, time=None):
        error = self.error = self.angle_diff( self.target, feedback, self.angle_max)
        if time is None:
            dt = 1
        else:
            dt = time - self.prev_time
        err_diff = self.angle_diff( error, self.prev_error, self.angle_max)
        self.integral += error
        derivative = 0
        if dt != 0: 
            derivative = (err_diff) / dt
        self.prev_error = error
        out = self.p*error + self.i*self.integral + self.d*derivative
        self.prev_time = time
        return out

pid= PID_angle(0.2,0.2,0.1)

a=pid(0.2)