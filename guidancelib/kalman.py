from numpy import dot, zeros, eye, cos, sin, diag, pi, matrix, array
from math import atan2
from numpy.linalg import norm, inv

import guidancelib

# stearing could be know by autoguidance and course could be calculated with
# delta_course = v/Length *tan(stearing_angle)


class ExtendedKalmanFilter(object):
    """Extended Kalman Filter for vehicle tracking"""
    def __init__(self):
        self.dim_x = 4 # Number of states:5 with gyro, 3 without
        self.dim_z = 3 # Number of measurements: 2 gps, 3 with gyro.

        self._dt = 1.0 / guidancelib.config['kalman.frequency']
        self._gps_dt = 1.0 / guidancelib.config['kalman.gps_frequency']

        self._x = zeros((self.dim_x,1))        # state
        self._F = eye(self.dim_x)
        self._P = eye(self.dim_x) * 1000  # uncertainty covariance
        
        # noise covariance
        s_error_gps = 0.5 # Probably a function related to number of satelites could be ok
        s_error_speed = 3
        R = diag([s_error_gps**2, s_error_gps**2, s_error_speed**2])
        self._R = R 

        # process uncertainty
        s_gps      = 0.5 * 8.8 * self._dt**2 # 8.8m/s2 as maximum acceleration
        s_course   = 0.1 * self._dt # 0.1rad/s as maximum turn rate for the vehicle
        s_velocity = 8.8 * self._dt # 8.8m/s2 as maximum acceleration
        self._Q = diag([s_gps**2, s_gps**2, s_course**2, s_velocity**2]) 

        # jacobian of meassurement function (H)
        JH = zeros((self.dim_z, self.dim_x)) # 2 if only gps, more for additional sensors
        JH[0,0] = 1   # gps
        JH[1,1] = 1   # gps
        JH[2,3] = 1   # gps calculated speed 
        self._JH = matrix(JH)

        self._y = zeros((self.dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = eye(self.dim_x)

    def _f_function(self, x):
        next_x = x
        next_x[0] = x[0] + x[3]* self._dt * cos(x[2])
        next_x[1] = x[1] + x[3]* self._dt * sin(x[2])
        dx = [ next_x[i] - x[i] for i in range(2)]
        next_x[2] = 90 - atan2(dx[1], dx[0]) * 180/ pi

        next_x[3] = x[3]

        return next_x

    def _calculate_jacobian_f(self, x):
        jacobian_f = [[1, 0, -self._dt * x[3] * sin(x[2]), self._dt * cos(x[2])],
                      [0, 1,  self._dt * x[3] * cos(x[2]), self._dt * sin(x[2])],
                      [0, 0,                            1,                    0],
                      [0, 0,                            0,                    1]]

        return matrix(jacobian_f)

    def _predict(self):
        self._x = self._f_function(self._x)

        self._F = self._calculate_jacobian_f(self._x)

        self._P = self._F * self._P * self._F.T + self._Q

    def filter_update(self, z):
        z = array(z)
        # Compute the Kalman Gain
        S = self._JH * self._P * self._JH.T + self._R
        K = (self._P * self._JH.T) * inv(S)

        hx = array([self._x[i] for i in [0,1,3]])
        # Update the estimate via measurement
        self._x = self._x + K * (z.reshape(self._JH.shape[0],1) - hx.reshape(self._JH.shape[0],1))

        # Update the error covariance
        self._P = (self._I - (K * self._JH)) * self._P


        self._predict()

        return self._x, self._P
