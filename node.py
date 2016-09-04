# -*- coding: utf-8 -*-
"""
Copyright (C) Fri Aug 26 17:07:53 2016  Jianshan Zhou
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
 
This module defines the class of node object as well as its sub-classes.
"""

import numpy as np
import copy

class Node(object):
    __slots__ = ["ID", "position", "speed", "acceleration", "direct_flag", 
                 "initial_energy", "p_max", "residual_energy", "current_power"]
    def __init__(self, nodeID = 0, # an int denoting node ID
                       position = np.array([0.,0.]), # an array denoting node position vector in m
                       speed = np.array([0.,0.]), # an array denoting node speed in m/s
                       acceleration = np.array([0.,0.]), # an array denoting node acc in m/s^2
                       direct_flag = 1,
                       initial_energy = 0.0, # a float in J
                       p_max = 0.0): # a float that is the usable maximum power in W
        self.ID = nodeID
        self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.residual_energy = initial_energy
        self.initial_energy = initial_energy
        self.p_max = p_max
        self.current_power = p_max
        self.direct_flag = direct_flag

    def update_acceleration(self, mobility, front_vehicle = None):
            mobility.car_following_IDM(front_vehicle, self)
    
    def update_speed(self, dt):
        self.speed = self.speed + dt*self.acceleration
        
    def update_position(self, dt):
        self.position = self.position + self.speed*dt
        

class Source(Node):
    __slots__ = ["neighborSet", "relaySet"]
    def __init__(self, nodeID = 0, # an int denoting node ID
                       position = np.array([0.,0.]), # an array denoting node position vector in m
                       speed = np.array([0.,0.]), # an array denoting node speed in m/s
                       acceleration = np.array([0.,0.]), # an array denoting node acc in m/s^2
                       direct_flag = 1,
                       initial_energy = 0.0, # a float in J
                       p_max = 0.0): # a float that is the usable maximum power in W
        Node.__init__(self, nodeID, position, speed, acceleration, direct_flag,
                      initial_energy, p_max)
        
    def initialize_neighborSet(self, neighborSet):
        self.neighborSet = neighborSet
        
    def initialize_relaySet(self):
        self.relaySet = copy.deepcopy(self.neighborSet)

class Relay(Node):
    __slots__ = ["reward", "reward_max", "reward_min", "normalized_reward", 
                 "reward_trace", "strategy_trace", "strategy", "current_action", 
                 "actionSet"]
    def __init__(self, nodeID = 0, # an int denoting node ID
                       position = np.array([0.,0.]), # an array denoting node position vector in m
                       speed = np.array([0.,0.]), # an array denoting node speed in m/s
                       acceleration = np.array([0.,0.]), # an array denoting node acc in m/s^2
                       direct_flag = 1,
                       initial_energy = 0.0, # a float in J
                       p_max = 0.0): # a float that is the usable maximum power in W
        Node.__init__(self, nodeID, position, speed, acceleration, direct_flag,
                      initial_energy, p_max)
        self.reward = 0.0
        self.reward_max = self.reward
        self.reward_min = self.reward
        self.normalized_reward = self.reward
        
        #record the rewards at each epoch
        self.reward_trace = []
        self.reward_trace.append(self.normalized_reward)
        self.strategy_trace = []
        
    def initialize_actionSet(self, actionSet):
        self.actionSet = actionSet
        if self.actionSet:
            self.strategy = (1.0/len(self.actionSet))*np.ones((len(self.actionSet),), dtype=float)
        else:
            raise
        self.strategy_trace.append(self.strategy.copy()) # record the initialized strategy
            
    def initialize_current_action(self):
        self.current_action = np.random.choice(self.actionSet) # by uniform randomness
