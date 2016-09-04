# -*- coding: utf-8 -*-
"""
Copyright (C) Sat Aug 27 18:21:36 2016  Jianshan Zhou
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
 
This module defines the class of the node speed distribution and the inter-node
 space distribution.
"""
import numpy as np
from scipy.stats import norm, expon, lognorm, fisk

class Speed_headway_random(object):
    def __init__(self, scenario_flag = "Freeway_Free"):
        """
        Totally five scenarios are supported here:
        Freeway_Night, Freeway_Free, Freeway_Rush;
        Urban_Peak, Urban_Nonpeak.
        The PDFs of the vehicle speed and the inter-vehicle space are adapted 
         from existing references.
        """
        if scenario_flag == "Freeway_Night":
            self.headway_random = expon(0.0, 1.0/256.41)
            meanSpeed = 30.93 #m/s
            stdSpeed = 1.2 #m/s
        elif scenario_flag == "Freeway_Free":
            self.headway_random = lognorm(0.75, 0.0, np.exp(3.4))
            meanSpeed = 29.15 #m/s
            stdSpeed = 1.5 #m/s
        elif scenario_flag == "Freeway_Rush":
            self.headway_random = lognorm(0.5, 0.0, np.exp(2.5))
            meanSpeed = 10.73 #m/s
            stdSpeed = 2.0 #m/s
        elif scenario_flag == "Urban_Peak":
            scale = 1.096
            c = 0.314
            loc = 0.0
            self.headway_random = fisk(c, loc, scale)
            meanSpeed = 6.083 #m/s
            stdSpeed = 1.2 #m/s
        elif scenario_flag == "Urban_Nonpeak":
            self.headway_random = lognorm(0.618, 0.0, np.exp(0.685)) 
            meanSpeed = 12.86 #m/s
            stdSpeed = 1.5 #m/s
        else:
            raise
        
        self.speed_random = norm(meanSpeed, stdSpeed)
