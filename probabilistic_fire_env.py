from abstract_fire_env import AbstractFireEnv
import numpy as np
import random
height = 100
width = 100
D = 2
K = 0.05


class ProbabilisticFireEnv(AbstractFireEnv):

  def next_observation(self):

    probability_map = np.zeros(shape=(height,width), dtype=float)

    self.fuel[(self.observation == 1) & (self.fuel > 0)] -= 1
    
    self.observation[(self.observation == 1) & (self.fuel == 0)] = 0

    for y, x in zip(*np.where((self.observation == 0) & (self.fuel > 0))):
      
      Y, X = np.ogrid[:height, :width]
      dist_from_cell = np.sqrt((X - x)**2 + (Y-y)**2)
      neighboring_burning_cells = (dist_from_cell <= D) & (self.observation == 1)
      if np.count_nonzero(neighboring_burning_cells) > 0:
        pnm = 1
        for (ny, nx) in list(zip(*np.where(neighboring_burning_cells))):
          dnmkl = np.array([y-ny, x-nx])
          norm = np.sum(dnmkl**2)
          pnmkl0 = K/norm
          pnmklw = K*(dnmkl @ self.wind)/norm 
          pnmkl  = max(0, min(1, (pnmkl0+pnmklw)))
          pnm *= (1-pnmkl)
        pmn = 1 - pnm
        probability_map[y, x] = pmn

    self.observation[probability_map > np.random.rand(height,width)] = 1
    return self.observation
    

  def reset_observation(self):
    center = [49, 49]
    self.observation = np.zeros(shape=(self.height, self.width), dtype=int)
    self.observation[center[0]-2:center[0]+2, center[1]-2:center[1]+2] = 1
    self.fuel = np.random.randint(low=15, high=20, size=(self.height, self.width))
    self.wind = np.random.uniform(low=-0.25, high=0.25, size=2)
    return self.observation