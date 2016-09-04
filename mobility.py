# -*- coding: utf-8 -*-
"""
Copyright (C) Fri Aug 26 19:29:28 2016  Jianshan Zhou
Contact: zhoujianshan@buaa.edu.cn	jianshanzhou@foxmail.com
Website: <https://github.com/JianshanZhou>

This program is free software: you can redistribute
 it and/or modify it under the terms of
 the GNU General Public License as published
 by the Free Software Foundation,
 either version 3 of the License,
 or (at your option) any later version.
 
This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY;
 without even the implied warranty of MERCHANTABILITY
 or FITNESS FOR A PARTICULAR PURPOSE.
 See the GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program.
 If not, see <http://www.gnu.org/licenses/>.
 
This module defines the car-following model class.
"""

import numpy as np
from node import Node

class Mobility(object):
    """
    Mobility class defines the dynamics of single vehicles in the net.
    It is worth pointing out that this model simply considers one-dimensional 
    vehicle mobility scenarios.
    """
    def __init__(self,
                 v0 = 30.0, # desired speed in m/s
                 T = 1.5, # safe time headway in s
                 a = 1.0, # maximum acceleration in m/s^2
                 b = 3.0, # desired deceleration in m/s^2
                 delta = 4.0, # acceleration exponent
                 s0 = 2, # minimum distance in m
                 l0 = 5.0): # the length of each vehicle in m
        
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.l0 = l0
        
    def car_following_IDM(self, front_vehicle, rear_vehicle):
        """
        Adopt the Intelligent Driver Model (IDM) for car following dynamics.
        """
        
        x2 = rear_vehicle.position[0]
        v2 = rear_vehicle.speed[0]
        
        if isinstance(front_vehicle, Node) and isinstance(rear_vehicle, Node):
            x1 = front_vehicle.position[0]
            v1 = front_vehicle.speed[0]
            net_distance = x1 - x2 - self.l0
            dv = v2 - v1     
        else:
            net_distance = 20.0
            dv = v2 - self.v0
            
        s = self.s0 + v2*self.T + (v2*dv)/(2*np.sqrt(self.a*self.b)) 
        component1 = (v2/self.v0)**self.delta
        component2 = (s/net_distance)**2
        rear_vehicle.acceleration[0] = self.a*(1.0 - component1 - component2)
        

