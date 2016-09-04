# -*- coding: utf-8 -*-
"""
Copyright (C) Sat Aug 27 12:15:12 2016  Jianshan Zhou
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
 
This module define the experiment scenario.
"""

import numpy as np
import scipy as spy
import DLbSoRS_Alg as alg

from node import Source, Relay

class Scenario(object):
    
    def __init__(self, mobility, distribution, net_info, const, 
                 lane_width = 3.0, # lane width
                 communication_delay = 10.0*10**(-3), # the delay in one-hop communication in second
                 movement_delay = 1.0,                 
                 total_sim_time = 120.0): # total simulation time in second
        self.mobility = mobility
        self.distribution = distribution
        self.net_info = net_info
        self.const = const
        
        self.lane_width = lane_width
        self.comDt = communication_delay
        self.moveDt = movement_delay
        self.total_sim_time = total_sim_time
        
    def __setup_node_pool(self):
        self.sourceNodes = []
        self.relayNodes = []
        self.node_pool = []
        
        source_id_flag = 0
        relay_id_flag = 0        
        
        node_num_per_lane = self.net_info.SOURCE_NUM_PER_LANE + self.net_info.RELAY_NUM_PER_LANE
        indices_in_lane = range(node_num_per_lane)
        
        for lane_ID in range(self.net_info.LANE_NUM):
            # select the sources in a random way
            indices_source = np.random.choice(indices_in_lane, self.net_info.SOURCE_NUM_PER_LANE, replace=False)
            #print indices_source
            # set up each lane pool    
            x0 = 0.0       
            lane_pool = []
            for index_in_lane in indices_in_lane:
                x0 += np.abs(self.distribution.headway_random.rvs()) + 5.0 + self.mobility.s0 # 5.0 is vehicle length in m
                if index_in_lane in indices_source:
                    source = Source(nodeID = source_id_flag, # an int denoting node ID
                       position = np.array([x0,self.lane_width*(0.5+lane_ID)]), # an array denoting node position vector in m
                       speed = np.array([np.abs(self.distribution.speed_random.mean()),0.]), # an array denoting node speed in m/s
                       acceleration = np.array([0.,0.]), # an array denoting node acc in m/s^2
                       direct_flag = 1,
                       initial_energy = self.net_info.initial_energy, # a float in J
                       p_max = self.net_info.p_max) # add source here
                    source_id_flag += 1
                    self.sourceNodes.append(source)
                    lane_pool.append(source)
                else:
                    relay = Relay(nodeID = relay_id_flag, # an int denoting node ID
                       position = np.array([x0,self.lane_width*(lane_ID+0.5)]), # an array denoting node position vector in m
                       speed = np.array([np.abs(self.distribution.speed_random.mean()),0.]), # an array denoting node speed in m/s
                       acceleration = np.array([0.,0.]), # an array denoting node acc in m/s^2
                       direct_flag = 1,
                       initial_energy = self.net_info.initial_energy, # a float in J
                       p_max = self.net_info.p_max) # add relay here
                    relay_id_flag += 1
                    self.relayNodes.append(relay)
                    lane_pool.append(relay)
            self.node_pool.append(lane_pool)
#        for lane_pool in self.node_pool:
#            lane_pool[-1].speed[0] = self.mobility.v0 # set the head with the desired speed
            
    def __setup_neighbor_structure(self):
        # for sources
        for source in self.sourceNodes:
            neighborSet = [] # a list that contains the indices of neighboring relays
            for relay in self.relayNodes:
                if spy.linalg.norm(source.position-relay.position) <= self.const.RADIUS:
                    neighborSet.append(relay.ID)
            source.initialize_neighborSet(neighborSet)
            print "The Source-%d has %d neighboring relays."%(source.ID,len(source.neighborSet))
            if source.neighborSet:
                source.initialize_relaySet()
            else:
                raise
        
        # for relays
        for relay in self.relayNodes:
            actionSet = []
            for source in self.sourceNodes:
                if spy.linalg.norm(relay.position-source.position) <= self.const.RADIUS:
                    actionSet.append(source.ID)
            relay.initialize_actionSet(actionSet) # relay initializes action set and strategy
            print "The Relay-%d has %d neighboring sources."%(relay.ID,len(relay.actionSet))
            if relay.actionSet:
                relay.initialize_current_action()
            else:
                raise
                
    def update_position(self):
        # this should be done with the head and following that one by one
        for lane_pool in self.node_pool:
            for index in np.arange(-1,-len(lane_pool)-1,-1):
                rear_node = lane_pool[index]
                if index == -1:
                    front_node = -1
                else:
                    front_node = lane_pool[index+1]
                rear_node.update_acceleration(self.mobility,front_node)
                rear_node.update_speed(self.comDt)
                rear_node.update_position(self.comDt)
    
    def initialization(self):
        self.__setup_node_pool()
        self.__setup_neighbor_structure()
    
    def adaptation(self):
        # 1) each player determines the action using its selection probability
        for relay in self.relayNodes:
            if len(relay.strategy) == 0 or len(relay.actionSet) == 0:
                continue
            relay.current_action = alg.Selection(relay.actionSet, relay.strategy)
            #relay.current_action = alg.RWS(relay.actionSet, relay.strategy)
        # 2) each source constructs its relay set
        for source in self.sourceNodes:
            source.relaySet = alg.ConstructRelaySet(source.ID, source.neighborSet,
                                                  self.relayNodes)                    
            # 3) each source determines the optimal power allocation
            rSet = source.relaySet # a list
            lam_ij = self.net_info.lambda_ij_array[source.ID] # iid assumption is adopted
            lam_id = self.net_info.lambda_id_array[source.ID]
            lamjd_rSet = self.net_info.lambda_jd_array[rSet,source.ID]
            pi_max = source.p_max
            w_i, wj_rSet = alg.solve_optPower(lam_ij, lam_id, lamjd_rSet, rSet, pi_max)
            # 4) set the current power and perform the cooperative communication
            alg.set_power(source.ID, self.sourceNodes, self.relayNodes, w_i, wj_rSet)
            alg.coopCommunication(self.comDt, source.ID, self.sourceNodes, self.relayNodes)
            
        # 5) each relay determines its payoff as well as normalizes it
        for relay in self.relayNodes:
            if len(relay.strategy) == 0 or len(relay.actionSet) == 0:
                continue
            relay.reward = alg.get_reward(relay.ID, self.sourceNodes, 
                                                  self.relayNodes) # update the reward
            alg.update_reward_record(relay.ID, self.relayNodes)
            alg.update_normalized_reward(relay.ID, self.relayNodes)
            # relay records its payoff
            relay.reward_trace.append(relay.normalized_reward)
            # 6) each relay updates its strategy
            alg.update_strategy(relay.ID, self.relayNodes)
            # relay records its strategy
            relay.strategy_trace.append(relay.strategy.copy())

    def update(self):
        # each source and relay update their residual energy as well as records
        for source in self.sourceNodes:
            #alg.coopCommunication(self.comDt, source.ID, self.sourceNodes, self.relayNodes)
            # source updates its neighboring relay
            alg.update_source_neighborSet(source.ID, self.sourceNodes, self.relayNodes)
            alg.update_source_relaySet(source.ID, self.sourceNodes, self.relayNodes)
        
        for relay in self.relayNodes:
            alg.update_player_actionSet_strategy(relay.ID, self.sourceNodes, self.relayNodes)
        