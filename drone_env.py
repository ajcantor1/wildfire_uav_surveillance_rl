from svgpath2mpl import parse_path
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import rotate, shift
import numpy as np
import math
import random
import random

height = width = 100
LAMBDA_1 = 0.35
LAMBDA_2 = 0.35
LAMBDA_3 = 0.15
LAMBDA_4 = 0.15 
C = 50
Cx = 50
Cy = 50

VELOCITY = 2
GRAVITY  = 0.981
MINRANGE = 15   # Minimium initial distance from wildfire seed
MAXRANGE = 30   # Maximum initial distance from wildfire seed
BANK_ANGLE_DELTA  = 5

plane_marker = parse_path('M 11.640625 15.0625 L 9.304688 13.015625 L 9.300781 9.621094 L 15.125 11.511719 L 15.117188 10.109375 L 9.257812 5.535156 L 9.25 2.851562 L 9.25 1.296875 C 9.253906 1.019531 9.140625 0.777344 8.960938 0.585938 C 8.738281 0.324219 8.410156 0.15625 8.039062 0.160156 C 8.027344 0.160156 8.011719 0.164062 8 0.164062 C 7.988281 0.164062 7.972656 0.160156 7.960938 0.160156 C 7.589844 0.15625 7.257812 0.324219 7.035156 0.585938 C 6.859375 0.777344 6.746094 1.019531 6.746094 1.296875 L 6.746094 2.851562 L 6.742188 5.535156 L 0.882812 10.109375 L 0.875 11.511719 L 6.699219 9.621094 L 6.691406 13.011719 L 4.359375 15.0625 L 4.355469 15.761719 L 4.628906 15.695312 L 4.628906 15.839844 L 7.511719 14.992188 L 8 14.875 L 8.484375 14.992188 L 11.371094 15.839844 L 11.375 15.695312 L 11.644531 15.761719 Z M 11.640625 15.0625 ')
plane_marker.vertices -= plane_marker.vertices.mean(axis=0)
#plane_marker = plane_marker.transformed(matplotlib.transforms.Affine2D().rotate_deg(180))

def euclidean_distance(x1, y1, x2, y2):
  return math.sqrt((x2-x1)**2+(y2-y1)**2)

def shift_matrix(matrix, x, y, padding_value=0):
  deltaX = Cx-x
  deltaY = Cy-y


  if deltaX==0 and deltaY==0:
    return matrix

  return shift(matrix, (deltaY, deltaX), cval = padding_value)
  

class Drone:

  def __init__(self, _droneEnv, _dt, _dti):
    self._bank_angle = 0
    self._droneEnv = _droneEnv
    self._trajectory = []
    self._otherDrone = None
    self.dt = _dt
    self.dti = _dti

  def reset(self):
    radius = random.random()*(MAXRANGE-MINRANGE) + MINRANGE
    angle = (random.random()-0.5)*2*np.pi
    self._x = radius*np.cos(angle) + 50
    self._y = radius*np.sin(angle) + 50
    self._bank_angle = 0
    self._trajectory = [(self.x, self.y)]
    self._heading_angle = (random.random()-0.5)*2*np.pi

  @property
  def otherDrone(self):
    return self._otherDrone

  @otherDrone.setter
  def otherDrone(self, _otherDrone):
    self._otherDrone = _otherDrone

  @property
  def trajectory(self):
    return self._trajectory

  @trajectory.setter
  def trajectory(self, _trajectory):
    self._trajectory = _trajectory

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, _x):
    self._x = _x

  @property
  def y(self):
    return self._y

  @y.setter
  def y(self, _y):
    self._y = _y


  @property
  def bank_angle(self):
    return self._bank_angle

  @bank_angle.setter
  def bank_angle(self, _bank_angle):
    self._bank_angle = _bank_angle

  @property
  def heading_angle(self):
    return self._heading_angle

  @heading_angle.setter
  def heading_angle(self, _heading_angle):
    self._heading_angle = _heading_angle
  
  @property
  def rho(self):
    return euclidean_distance(self.x, self.y, self.otherDrone.x, self.otherDrone.y)

  @property
  def theta(self):
    _theta = np.arctan2((self.otherDrone.y-self.y),(self.otherDrone.x-self.x)) - self.heading_angle
    
    if (_theta > math.pi):
      _theta -= 2*math.pi
    elif (_theta<-math.pi):
      _theta+= 2*math.pi

    return _theta

  @property
  def psi(self):
    _psi = self.otherDrone.heading_angle - self.heading_angle

    if (_psi > math.pi):
      _psi -= 2*math.pi
    elif (_psi<-math.pi):
      _psi += 2*math.pi

    return _psi
    
  @property
  def state(self):
    return np.array([
        self.bank_angle, 
        self.rho,
        self.theta,
        self.psi,
        self.otherDrone.bank_angle
    ])[np.newaxis,...]

  @property
  def belief_map(self):
    return self._transform_map(self._droneEnv.belief_map_channel.copy())
  
  @property
  def time_elasped_map(self):
    return self._transform_map(self._droneEnv.time_map_channel.copy(), 250.0)/250.0

  def _transform_map(self, map, padding_value=0):
    return rotate(shift_matrix(map, self.x, self.y, padding_value), angle=np.rad2deg(self.heading_angle), reshape=False, cval=padding_value)

  @property
  def observation(self):
    return np.stack((self.time_elasped_map, self.belief_map), axis=0)[np.newaxis,...]
    
  def step(self, input):

    self.x +=  VELOCITY*math.cos(self.heading_angle)
    self.y +=  VELOCITY*math.sin(self.heading_angle)
    self.trajectory.append((self.x, self.y))  
    self.heading_angle += GRAVITY*np.tan(self.bank_angle)/(VELOCITY)

    if (self.heading_angle>np.pi):
      self.heading_angle-=2*np.pi
    elif (self.heading_angle<-math.pi):
      self.heading_angle+=2*np.pi

    action =  5.0*np.pi/180.0 if input==1 else -5.0*np.pi/180.0
    self.bank_angle += action
    

    if self.bank_angle >  50.0*np.pi/180.0 or self.bank_angle < -50.0*np.pi/180.0:
      self.bank_angle -= action

  @property
  def reward(self):

    return self._reward1()+self._reward2()+self._reward3()+self._reward4()
      

  def _reward1(self):
    
    fire_points = np.argwhere(self._droneEnv.belief_map_channel == 1)

    if len(fire_points) == 0:
      return -LAMBDA_1*141.42
    else:
      euclidean_distances = [euclidean_distance(self.x, self.y, fire_point[1], fire_point[0]) for fire_point in fire_points]
      return -LAMBDA_1*min(euclidean_distances) 


  def _reward2(self):

    min_drone_y = int(max(self.y-self._droneEnv.scan_radius,0))
    max_drone_y = int(min(self.y+self._droneEnv.scan_radius, height))

    min_drone_x = int(max(self.x-self._droneEnv.scan_radius,0))
    max_drone_x = int(min(self.x+self._droneEnv.scan_radius, width))
    
    radar = self._droneEnv.belief_map_channel[min_drone_y:max_drone_y,min_drone_x:max_drone_x]
    return -LAMBDA_2*(np.count_nonzero(radar==0))

  def _reward3(self):
    return -LAMBDA_3*(np.deg2rad(self.bank_angle)**2)

  def _reward4(self):
    return -LAMBDA_4*math.exp(-self.rho/C)

  def plot_time_elapsed(self, fig, ax):

    ax.axis(xmin=0, xmax=width)
    ax.axis(ymin=0, ymax=height)
    time_elasped_plot = ax.imshow(self.time_elasped_map*250.0, cmap='gray', vmin=0, vmax=250)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(time_elasped_plot, cax=cax)
    return time_elasped_plot 

  def plot_belief_map(self, fig, ax):

    ax.axis(xmin=0, xmax=width)
    ax.axis(ymin=0, ymax=height)
    belief_map_plot = ax.imshow(self.belief_map, cmap='gray_r', vmin=0, vmax=1)  
    return belief_map_plot


class DronesEnv:
  def __init__(self, _height, _width, _dt, _dti, _scan_radius=10):
    self._drones = [Drone(self, _dt, _dti), Drone(self, _dt, _dti)]
    self._drones[0].otherDrone = self._drones[1]
    self._drones[1].otherDrone = self._drones[0]
    self._height = _height 
    self._width  = _width
    self._scan_radius = _scan_radius

  @property 
  def scan_radius(self):
    return self._scan_radius

  @scan_radius.setter
  def scan_radius(self, _scan_radius):
    self._scan_radius = _scan_radius
        
  def reset(self, fireMap=np.nan):


    self._drones[0].reset()
    self._drones[1].reset()

    self._belief_map_channel = fireMap.copy()
    #self._belief_map_channel = np.zeros(shape=(self._height, self._width))
    self._time_elapsed_channel = np.full(shape=(self._height, self._width), fill_value=250)

    self._drone_scan(self._drones[0], fireMap)
    self._drone_scan(self._drones[1], fireMap)

  def _drone_scan(self, drone, fireMap=np.nan):

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - drone.x)**2 + (Y-drone.y)**2)
    mask = dist_from_center <= self.scan_radius

    reward = np.count_nonzero(mask & (self._belief_map_channel==0) & (fireMap==1))
    self._belief_map_channel[mask] = fireMap[mask]
    
    self._time_elapsed_channel[mask] = 0

    self._time_elapsed_channel[~mask & (self._time_elapsed_channel < 250)] += 1

    return reward

  @property 
  def belief_map_channel(self):
    return self._belief_map_channel

  @belief_map_channel.setter
  def belief_map_channel(self, _belief_map_channel):
    self._belief_map_channel = _belief_map_channel

  @property 
  def time_map_channel(self):
    return self._time_elapsed_channel

  @time_map_channel.setter
  def time_map_channel(self, _time_elapsed_channel):
    self._time_elapsed_channel = _time_elapsed_channel

  @property
  def drones(self):
    return self._drones

  def step(self, input, fireMap):
    self._drones[0].step(input[0])
    self._drones[1].step(input[1])
    reward1 = self._drone_scan(self._drones[0], fireMap)
    reward2 = self._drone_scan(self._drones[1], fireMap)
    return reward1, reward2

  def plot_time_elapsed(self, fig, ax):
    ax.axis(xmin=0, xmax=width)
    ax.axis(ymin=0, ymax=height)
    time_elasped_plot = ax.imshow(self._time_elapsed_channel, cmap='gray', vmin=0, vmax=250)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(time_elasped_plot, cax=cax)
    return time_elasped_plot

  def plot_belief_map(self, fig, ax):
    ax.axis(xmin=0, xmax=width)
    ax.axis(ymin=0, ymax=height)
    belief_map_plot = ax.imshow(self._belief_map_channel, cmap='gray_r', vmin=0, vmax=1)
    return belief_map_plot   

  def plot_drones(self, fig, ax):
      
    ax.axis(xmin=0, xmax=self._width)
    ax.axis(ymin=0, ymax=self._height)
    ax.set_aspect(1)
    ax.grid()

    plane_marker_1 = matplotlib.markers.MarkerStyle(marker=plane_marker)
    plane_marker_1._transform = plane_marker_1.get_transform().rotate(self.drones[0].heading_angle)

    plane_marker_2 = matplotlib.markers.MarkerStyle(marker=plane_marker)
    plane_marker_2._transform = plane_marker_2.get_transform().rotate(self.drones[1].heading_angle)

    ax.scatter(self.drones[0].x, self.drones[0].y, marker=plane_marker_1, s=30**2)

    ax.scatter(self.drones[1].x, self.drones[1].y, marker=plane_marker_2, s=30**2)

    heading_line = np.array([0, 50])

    x1 = self._drones[0].x + np.cos(np.deg2rad(-90) + self._drones[0].heading_angle) * heading_line
    y1 = self._drones[0].y + np.sin(np.deg2rad(-90) + self._drones[0].heading_angle) * heading_line

    heading_line = np.array([0, 50])

    x2 = self._drones[1].x + np.cos(np.deg2rad(-90) + self._drones[1].heading_angle) * heading_line
    y2 = self._drones[1].y + np.sin(np.deg2rad(-90) + self._drones[1].heading_angle) * heading_line
    
    ax.plot(x1, y1, '--')
    ax.plot(x2, y2, '--')

  def plot_trajectory(self, fig, ax):
      
    ax.axis(xmin=0, xmax=self._width)
    ax.axis(ymin=0, ymax=self._height)
    ax.set_aspect(1)
    ax.grid()

    plane_marker_1 = matplotlib.markers.MarkerStyle(marker=plane_marker)
    plane_marker_1._transform = plane_marker_1.get_transform().rotate(self._drones[0].heading_angle)

    plane_marker_2 = matplotlib.markers.MarkerStyle(marker=plane_marker)
    plane_marker_2._transform = plane_marker_2.get_transform().rotate(self._drones[1].heading_angle)

    ax.scatter(self.drones[0].x, self.drones[0].y, marker=plane_marker_1, s=30**2)

    ax.scatter(self.drones[1].x, self.drones[1].y, marker=plane_marker_2, s=30**2)

    x1, y1 = zip(*self.drones[0].trajectory)
    x2, y2 = zip(*self.drones[1].trajectory)

    ax.plot(x1, y1, '.')
    ax.plot(x2, y2, '.')