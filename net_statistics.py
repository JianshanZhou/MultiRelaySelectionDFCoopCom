# -*- coding: utf-8 -*-
"""
Copyright (C) Sat Aug 27 18:33:12 2016  Jianshan Zhou
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
 
This module defines the net information module.
"""

class Net_info(object):
    __slots__ = ("LANE_NUM", "SOURCE_NUM", "SOURCE_NUM_PER_LANE", 
                 "RELAY_NUM", "RELAY_NUM_PER_LANE", 
                 "lambda_id_array", "lambda_ij_array", "lambda_jd_array", 
                 "p_max", "initial_energy")
    def __init__(self, lane_num = 2, 
               source_num_per_lane = 2, 
               relay_num_per_lane = 5):
                   
        self.LANE_NUM = lane_num
        self.SOURCE_NUM = source_num_per_lane*lane_num
        self.SOURCE_NUM_PER_LANE = source_num_per_lane
        self.RELAY_NUM = relay_num_per_lane*lane_num
        self.RELAY_NUM_PER_LANE = relay_num_per_lane
    
    def initialize_channel_parameters(self, assumption_flag = "iid"):
        from constants import Constants
        import numpy as np
        const = Constants()
        self.lambda_id_array = np.ones((self.SOURCE_NUM,), dtype = float)
        if assumption_flag == "iid":
            self.lambda_ij_array = const.LAMBDA_interval[0] + \
            (const.LAMBDA_interval[1]-const.LAMBDA_interval[0])*np.random.rand(self.SOURCE_NUM)
        elif assumption_flag == "niid":
            self.lambda_ij_array = const.LAMBDA_interval[0] + \
            (const.LAMBDA_interval[1]-const.LAMBDA_interval[0])*np.random.rand(self.SOURCE_NUM,self.RELAY_NUM)
        self.lambda_jd_array = const.LAMBDA_interval[0] + \
        (const.LAMBDA_interval[1]-const.LAMBDA_interval[0])*np.random.rand(self.RELAY_NUM,self.SOURCE_NUM)
        
    def initialize_power_level(self, p_max, initial_energy):
        self.p_max = p_max
        self.initial_energy = initial_energy
        

