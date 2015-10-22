from numpy import dot, zeros, ones, eye, cos, sin, diag, pi, matrix, array, tan, abs
from math import atan2
from numpy.linalg import norm, inv

import guidancelib


class ExtendedKalmanFilter(object):
    """Extended Kalman Filter for vehicle tracking"""
    def __init__(self):
        self.dim_x = 5 
        self.dim_z = 3 

        self._dt = 1.0 / guidancelib.config['gps.frequency']

        self._x = ones((self.dim_x,1))  # state
        self._F = eye(self.dim_x)
        self._P = eye(self.dim_x) * 10  # uncertainty covariance
        
        self._R = self._get_noise_covariance()

        self._Q = self._get_Q()

        self._JH = self._get_JH()

        self._I = eye(self.dim_x)

    def _get_noise_covariance(self):
        s_error_gps = 0.5 # Probably a function related to number of satelites could be ok
        s_error_speed = 1
        R = diag([s_error_gps**2, s_error_gps**2, s_error_speed**2])
        return R

    def _get_Q(self):
        # process uncertainty
        s_gps      = 0.5 * 8.8 * self._dt**2 # 8.8m/s2 as maximum acceleration
        s_course   = 0.1 * self._dt # 0.1rad/s as maximum turn rate for the vehicle
        s_velocity = 8.8 * self._dt # 8.8m/s2 as maximum acceleration
        s_stearing = 1.0 * self._dt # 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
        return diag([s_gps**2, s_gps**2, s_course**2, s_velocity**2, s_stearing**2])

    def _get_JH(self):
        # jacobian of meassurement function (H)
        JH = zeros((self.dim_z, self.dim_x)) # 2 if only gps, more for additional sensors
        JH[0,0] = 1   # gps x
        JH[1,1] = 1   # gps y
        JH[2,3] = 1   # gps calculated speed
        return matrix(JH) 

    def _f_function(self, x, u):
        next_x = x

        if abs(x[4]) < 0.0001:
            next_x[0] = x[0] + x[3]* self._dt * cos(x[2])
            next_x[1] = x[1] + x[3]* self._dt * sin(x[2])
            next_x[4] = 0.0000001 # avoid numerical issues in Jacobians
        else:
            next_x[0] = x[0] + (x[3]/x[4]) * ( sin(self._dt * x[4]) + x[2] - sin(x[2]))
            next_x[1] = x[1] + (x[3]/x[4]) * (-cos(self._dt * x[4]) + x[2] - cos(x[2]))

        next_x[2] = (x[2] + x[4] * self._dt + pi) % (2.0 * pi) - pi
        #next_x[2] = 90 - atan2(dx[1], dx[0]) * 180/ pi
        #next_x[3] = norm(dx)/ self._dt
    

        if u != 0:
            next_x[4] = next_x[3] / guidancelib.config['tractor.axis_distance'] * tan(u)

        return next_x

    def _calculate_jacobian_f(self, x):

        a13 = float((x[3]/x[4]) * (cos(x[4]*self._dt+x[2]) - cos(x[2])))
        a14 = float((1.0/x[4]) * (sin(x[4]*self._dt+x[2]) - sin(x[2])))
        a15 = float((self._dt*x[3]/x[4])*cos(x[4]*self._dt+x[2]) - (x[3]/x[4]**2)*(sin(x[4]*self._dt+x[2]) - sin(x[2])))
        a23 = float((x[3]/x[4]) * (sin(x[4]*self._dt+x[2]) - sin(x[2])))
        a24 = float((1.0/x[4]) * (-cos(x[4]*self._dt+x[2]) + cos(x[2])))
        a25 = float((self._dt*x[3]/x[4])*sin(x[4]*self._dt+x[2]) - (x[3]/x[4]**2)*(-cos(x[4]*self._dt+x[2]) + cos(x[2])))
  

        jacobian_f = [[1, 0, a13, a14,      a15],
                      [0, 1, a23, a24,      a25],
                      [0, 0,   1,   0, self._dt],
                      [0, 0,   0,   1,        0],
                      [0, 0,   0,   0,        1]]

        return matrix(jacobian_f)

    def _predict(self, u):
        self._x = self._f_function(self._x, u)

        self._F = self._calculate_jacobian_f(self._x)

        self._P = self._F * self._P * self._F.T + self._Q

    def filter_update(self, z, u=0):
        z = array(z)
        # Compute the Kalman Gain
        S = self._JH * self._P * self._JH.T + self._R
        K = (self._P * self._JH.T) * inv(S)

        # Update the estimate via measurement
        hx = array([self._x[i] for i in [0, 1, 3]])
        self._x = self._x + K * (z.reshape(self._JH.shape[0],1) - hx.reshape(self._JH.shape[0],1))

        # Update the error covariance
        self._P = (self._I - (K * self._JH)) * self._P

        self._predict(u)

        return self._x, self._P
