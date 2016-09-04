# -*- coding: utf-8 -*-
"""
Copyright (C) Sun Sep 04 11:03:54 2016  Jianshan Zhou
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
 
This module conducts the experiments in order to verify the proposed algorithm.
"""
#%% prepare basic parameters
import numpy as np

from traffic_distribution import Speed_headway_random
from mobility import Mobility
from constants import Constants
from net_statistics import Net_info


def power_W(power_in_dB):
    return 10.0**(power_in_dB/10.0)
def power_dB(power_in_W):
    return 10.0*np.log10(power_in_W)


const = Constants()

distribution = Speed_headway_random(scenario_flag = "Urban_Nonpeak")

mobility = Mobility(v0 = 120.0*1000./3600., # desired speed in m/s
                 T = 1.6, # safe time headway in s
                 a = 0.73, # maximum acceleration in m/s^2
                 b = 1.67, # desired deceleration in m/s^2
                 delta = 4.0, # acceleration exponent
                 s0 = 2.0, # minimum distance in m
                 l0 = 5.0)
                 
lane_width = 3.0
communication_delay = 10.0*10**(-3) # the delay in one-hop communication in second
movement_delay = 1.0                 
total_sim_time = 10.0
epoch_num = int(np.ceil(total_sim_time/communication_delay))
p_max = power_W(30.0) # p_max = 30 dB
initial_energy = p_max*communication_delay*epoch_num

net_info = Net_info(lane_num = 2, 
               source_num_per_lane = 2, 
               relay_num_per_lane = 5)
net_info.initialize_channel_parameters(assumption_flag = "iid")
net_info.initialize_power_level(p_max, initial_energy)

#%% initialize the simulation scenario

from scenario import Scenario

scenario = Scenario(mobility, distribution, net_info, const, 
                 lane_width, 
                 communication_delay,
                 movement_delay,                 
                 total_sim_time)
scenario.initialization()
#%% run the simulation

for epoch in range(epoch_num):
    print "adaptation at %d-th epoch..."%(epoch,)
    scenario.adaptation()
    scenario.update_position()
    scenario.update()
   
#%% record the normalized rewards received by each player in the Nash Equalibrium state
reward_results_NE = np.array([scenario.relayNodes[relayID].normalized_reward \
for relayID in range(scenario.net_info.RELAY_NUM)]):
    
#%% unilateral testing experiments

import copy
from DLbSoRS_Alg import unilateral_update

reward_results_DE = np.array([unilateral_update(copy.deepcopy(scenario), playerID) \
for playerID in range(scenario.net_info.RELAY_NUM)])