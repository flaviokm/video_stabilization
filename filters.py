import numpy as np
from copy import deepcopy

class KalmanFilter():
    def __init__(self,priori_value,shape,Q = 15e-2):
        self.process_variance = np.ones(shape)*Q
        self.priori_value = priori_value
        self.priori_error = np.zeros(shape)
        self.posteriori_value = np.zeros(shape)
        self.posteriori_error = np.zeros(shape)
        self.estimated_variance = np.ones(shape)*(1e-2)
        self.kalman_gain = np.zeros(shape)
    
    def update(self):
        self.priori_value = deepcopy(self.posteriori_value)
        self.priori_error = deepcopy(self.posteriori_error)
        self.kalman_gain = self.priori_error/(self.priori_error+self.estimated_variance)

    def estimate(self,measurement):
        self.posteriori_value = self.priori_value+ self.kalman_gain*(measurement-self.priori_value)
        self.posteriori_error = (np.ones(self.kalman_gain.shape)-self.kalman_gain)*self.priori_error + self.process_variance
        return self.posteriori_value
    
    def apply(self,measurement):
        self.update()
        return self.estimate(measurement)
