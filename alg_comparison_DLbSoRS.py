# -*- coding: utf-8 -*-
"""
Copyright (C) Wed Aug 31 07:45:27 2016  Jianshan Zhou
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
 
This module carries out a series of Monte Carlo based experiments where the 
 DLbSoRA_alg is compared with two conventional schemes, i.e., the stochastic 
 relay selection algorithm and the deterministic relay selection algorithm. 
 It should be noted that the stochastic relay selection algorithm constructs 
 the relay set for each source in a random way, while the deterministic 
 uniformly assigns relays for each source.
"""
#%% basic settings
import time
import numpy as np

from traffic_distribution import Speed_headway_random
from mobility import Mobility
from constants import Constants
from net_statistics import Net_info
from scenario import Scenario

startT = time.clock()

def power_W(power_in_dB):
    return 10.0**(power_in_dB/10.0)
def power_dB(power_in_W):
    return 10.0*np.log10(power_in_W)
def JFI_func(alist):
    return ((np.sum(alist))**2)/(1.*len(alist)*np.sum([x**2 for x in alist]))

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
p_max = power_W(40.0) # p_max 
initial_energy = p_max*communication_delay*epoch_num

sim_num = 100

lane_num = 2
source_num_per_lane_array = np.arange(1, 6)
relay_num_per_lane = 2*source_num_per_lane_array

reward_array = []
reward_JFI_array = []
reward_std_array = []
reward_JFI_std_array = []

residual_energy_array = []
residual_energy_JFI_array = []
residual_energy_std_array = []
residual_energy_JFI_std_array = []

for (source_num_per_lane, relay_num_per_lane) in zip(source_num_per_lane_array,\
relay_num_per_lane):
    reward_array_temp = []
    reward_JFI_array_temp = [] 
    residual_energy_array_temp = []
    residual_energy_JFI_array_temp = []
    for sim_flag in range(sim_num):
        print "with %d sources and %d nodes at the %d-th simulation..."%(source_num_per_lane*lane_num,\
        relay_num_per_lane*lane_num, sim_flag)
        reward_array_temp2 = []
        residual_energy_array_temp2 = []
        # do run
        net_info = Net_info(lane_num, 
                       source_num_per_lane, 
                       relay_num_per_lane)
        net_info.initialize_channel_parameters(assumption_flag = "iid")
        net_info.initialize_power_level(p_max, initial_energy)        
        scenario = Scenario(mobility, distribution, net_info, const, 
                         lane_width, 
                         communication_delay,
                         movement_delay,                 
                         total_sim_time)
        scenario.initialization()
        for epoch in range(epoch_num):
            #print "adaptation at %d-th epoch..."%(epoch,)
            scenario.adaptation()
            scenario.update_position()
            scenario.update()
        #record results
        for lane_pool in scenario.node_pool:
            for node in lane_pool:
                residual_energy_array_temp2.append(node.residual_energy)
        for player in scenario.relayNodes:
            reward_array_temp2.append(player.reward)
    
        reward_array_temp.append(np.mean(reward_array_temp2))
        reward_JFI_array_temp.append(JFI_func(reward_array_temp2))
        residual_energy_array_temp.append(np.mean(residual_energy_array_temp2))
        residual_energy_JFI_array_temp.append(JFI_func(residual_energy_array_temp2))
    
    reward_array.append(np.mean(reward_array_temp))
    reward_std_array.append(np.std(reward_array_temp))
    reward_JFI_array.append(np.mean(reward_JFI_array_temp))
    reward_JFI_std_array.append(np.std(reward_JFI_array_temp))
    
    residual_energy_array.append(np.mean(residual_energy_array_temp))
    residual_energy_std_array.append(np.std(residual_energy_array_temp))
    residual_energy_JFI_array.append(np.mean(residual_energy_JFI_array_temp))
    residual_energy_JFI_std_array.append(np.std(residual_energy_JFI_array_temp))

endT = time.clock()
print "simulation time cost: %.5f(sec)"%(endT-startT,)
np.savez("alg_comparison_results_proposed_20160902.npz",
         reward_array = np.asarray(reward_array),
         reward_std_array = np.asarray(reward_std_array),
         reward_JFI_array = np.asarray(reward_JFI_array),
         reward_JFI_std_array = np.asarray(reward_JFI_std_array),
         residual_energy_array = np.asarray(residual_energy_array), 
         residual_energy_std_array = np.asarray(residual_energy_std_array), 
         residual_energy_JFI_array = np.asarray(residual_energy_JFI_array), 
         residual_energy_JFI_std_array = np.asarray(residual_energy_JFI_std_array))

