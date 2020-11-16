
import math
import numpy as np

def limit_angle( angle, max_angle):
        #limitiert den Winkel auf einen +/-halben Kreis, z.B. auf max. -180°..180°
        half = max_angle / 2
        return (angle + half ) % max_angle - half


class PID_angle(object):

    def __init__(self, p=0, i=0, d=0, target=0, time=0, angle_max=float('inf'), out_min=float('inf'), out_max=float('inf'), anti_windup = 1):
        self.tune(p, i, d)
        self.target = 0
        self._prev_time = time 
        self._prev_error = 0
        self.error = None
        self.angle_max = angle_max
        self._integral = 0
        self.out_min = out_min
        self.out_max = out_max
        self.anti_windup = anti_windup

    
    def __call__(self, feedback, time=None):
        if self.target is None: return
        error = self.error = limit_angle( self.target - feedback, self.angle_max)
        if time is None:
            dt = 1
        else:
            dt = time - self._prev_time
        err_diff = limit_angle( error - self._prev_error, self.angle_max)
        if np.sign(error) != np.sign(self._prev_error):
            self._integral = self._integral / self.anti_windup
        self._integral += error * dt
        derivative = 0
        if dt != 0: 
            derivative = (err_diff) / dt
        self._prev_error = error
        out = self._p*error + self._i*self._integral + self._d*derivative
        self._prev_time = time
        return out
    
     #bitte entfernen
    def get_val(self, target, outer, time):
        self.target = target
        ret = self.__call__ (feedback=outer, time = time)
        ret = np.clip(ret, self.out_min, self.out_max)
        return ret

    #bitte entfernen
    def set_pid (self, kp, ki, kd):
        self.tune(kp, ki, kd)

    def tune(self, p, i, d):
        self._p = p
        self._i = i
        self._d = d
