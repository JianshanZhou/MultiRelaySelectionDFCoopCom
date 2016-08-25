# -*- coding: utf-8 -*-
"""
Copyright (C) Tue Aug 09 00:07:19 2016  Jianshan Zhou
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
 
This module defines some basic constants that may be used in the
 numerical computation involved in other programs of the Multi-Relay
 DF Cooperative Communication simulation experiments.
"""

class Constants(object):
                 
    def __init__(self):
        kw = {"RADIUS": 300.0,#the maximum communication range of nodes in (m)
              "N0": 1.0,#the variance of the noise terms in the signal model
              "ALPHA": 3.0,#the path-loss exponent in a typical environment
              "BETA": 0.01,#the outage probability threshold
              "R": 1.0,#the transmission date rate in (bps/Hz)
              "DELTA": 0.1,#the learning rate
              "a1": 1.0,
              "a2": 1.0}
        self.__dict__.update(kw)
    
    def __setattr__(self, k, v):
        raise
        
if __name__ == '__main__':
    
    conts = Constants()
    
    print conts.ALPHA 
    print conts.RADIUS
    #the follow is expected to rise an error
    conts.C = 1.0












