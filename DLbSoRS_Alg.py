# -*- coding: utf-8 -*-
"""
Copyright (C) Thu Aug 11 17:41:46 2016  Jianshan Zhou
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
 
This module realizes the decentralized learning-based self-origanized
 relay selection algorithm.
"""

#%% import some basic modules
import numpy as np
from outageProb import outage_prob_sys_iid, const
from scipy.optimize import brentq
from scipy.spatial.distance import euclidean
#%% the main components of the algorithm
def Initialization(actionSet, strategy):
    """
    This function initializes the probability strategy of a player based on its
     action set.
    Input
    actionSet: a list-type container containing the indices of a player's all 
     actions.
    strategy: an array denoting the selection probabilities of the actions.
    """
    if actionSet:
        #do initialization
        sN = len(strategy)
        if len(actionSet) == sN:
            for ind in range(sN):
                strategy[ind] = 1.0/(sN)
                
            return None
        else:
            print "The length of the strategy vector is not equal to that of\
            the action set!"
            raise
    else:
        #the action set is empty
        print "The action set is empty at the initialization phase!"
        strategy = np.array([], dtype=float)        
        return None
    
def Selection(actionSet, strategy):
    """
    This function selects an action from the action set based on the strategy.
    It returns the action, i.e., the index of the source node the player will 
     help transmit its information.
    """
    randomNum = np.random.rand()    
    cumProb = strategy.cumsum()
    for ind in range(len(cumProb)):
        if cumProb[ind]>=randomNum:
            break
    return actionSet[ind]

def neighborRelayID_with_maxReward(neighborSet, relayNodes):
    """
    This function determines the id of the neighboring relays who has the
     maximum reward currently.
     Input
     neighborSet: a list containing the IDs of a source's neighboring relays.
     relayNodes: a list containing the instances of all the relays in the netiwork.
     Output
     The ID of the selected relay, an int number.
    """
    relayID = neighborSet[0]
    for rID in neighborSet:
        if relayNodes[rID].reward>relayNodes[relayID].reward:
            relayID = rID
    return relayID
    
def ConstructRelaySet(sID, neighborSet, relayNodes):
    """
    This function constructs the relay set for a source.
    Input
    neighborSet: a list containing the indices of the neighboring 
     candidate relays of a source.
    relayNodes: a list containing the instances of the whole relay nodes in 
     the network.
    Output
    The output, i.e., the relay set, is a list containing the indices of the 
     selected relays.
    """
    rSet = []
    for neighborID in neighborSet:
        if relayNodes[neighborID].action == sID:
            rSet.append(neighborID)
    if rSet:
        return rSet
    else:
        rSet.append(neighborRelayID_with_maxReward(neighborSet,
                                                   relayNodes))
        return rSet
        
def solve_optPower(lam_ij, lam_id, lamjd_rSet, rSet, pi_max):
    """
    For the sake of demonstration, here the closed-form expression of the outage
     probability of the multi-relay DF cooperative communication under the
     assumption of i.i.d channel fading is exploited.
    Input
    pi_max and pi_min are the guessed bounds of the power level used at the
     source i, respectively.
    lamjd_rSet is an array containing the lambda_jd(i) over the set rSet
    """
    xmin = 1.0
    xmax = 10*np.log10(pi_max)
    #x is in dB
    func = lambda x: outage_prob_sys_iid(lam_ij, lam_id, 10.0**(x/10.0), rSet)
    x0 = brentq(func, xmin, xmax)
    w_i = 10.0**(x0/10.0)
    wj_rSet = lamjd_rSet*(w_i/lam_id)
    return w_i, wj_rSet#in Watt

def set_power(sID, sourceNodes, relayNodes, w_i, wj_rSet):
    """
    This function updates the current power level used by the source and its 
     relay nodes.
    """
    sourceNodes[sID].current_power = w_i
    rSet = sourceNodes[sID].relaySet
    indices = range(len(rSet))
    for (relayID, ind) in zip(rSet, indices):
        relayNodes[relayID].current_power = wj_rSet[ind]
        
    return None

def coopCommunication(dt, sID, sourceNodes, relayNodes):
    """
    This function updates the residual energy of the source and its relays.
    """
    sourceNodes[sID].residual_energy = sourceNodes[sID].residual_energy \
    - sourceNodes[sID].current_power*dt
    rSet = sourceNodes[sID].relaySet
    for relayID in rSet:
        relayNodes[relayID].residual_energy = relayNodes[relayID].residual_energy \
        - relayNodes[relayID].current_power*dt
    return None

def get_reward(relayID, sourceNodes, relayNodes):
    """
    This function udpates the reward of the players.
    """
    a1 = const.a1
    a2 = const.a2
    
    player = relayNodes[relayID]
    sourceID = player.current_action
    source = sourceNodes[sourceID]
    neighborSet = source.neighborSet
    actionSet = player.actionSet
    
    Q_ratio = (player.initial_energy \
    + source.initial_energy)/(player.residual_energy \
    + source.residual_energy)
    q_max = player.p_max + source.p_max
    q = player.current_power + source.current_power
    avg_A = 0.0
    avg_B = 0.0
    for action in actionSet:
        avg_A += sourceNodes[action].residual_energy/sourceNodes[action].initial_energy
    for neighbor in neighborSet:
        avg_B += relayNodes[neighbor].residual_energy/relayNodes[neighbor].initial_energy
    
    f = a1*(10.0*np.log10(q_max))*Q_ratio + a2*(avg_A + avg_B)/(len(neighborSet) + len(actionSet))
    g = a1*(10.0*np.log10(q))*Q_ratio
    
    return f - g

def update_reward_record(relayID, relayNodes):
    """
    This function updates the maximum and minimum reward records of a player.
    """
    player = relayNodes[relayID]
    if player.reward < player.reward_min:
        player.reward_min = player.reward
        return None
    elif player.reward > player.reward_max:
        player.reward_max = player.reward
        return None
    else:
        return None
        
def update_normalized_reward(relayID, relayNodes):
    player = relayNodes[relayID]
    reward_min = player.reward_min
    reward_max = player.reward_max
    tol = 1e-8
    if np.abs(reward_max - reward_min) <= tol:
        player.normalized_reward = 1.0
    else:
        player.normalized_reward = (player.reward - reward_min)/(reward_max - reward_min)
    return None

def update_strategy(relayID, relayNodes):
    player = relayNodes[relayID]
    current_action = player.current_action
    index = player.actionSet.index(current_action)
    tol = 1e-8
    for ind in range(len(player.strategy)):
        if np.abs(ind - index) <= tol:
            player.strategy[ind] = player.strategy[ind] \
            + const.DELTA*player.normalized_reward*(1.0 - player.strategy[ind])
        else:
            player.strategy[ind] = player.strategy[ind] \
            - const.DELTA*player.normalized_reward*player.strategy[ind]
    return None

def update_source_neighborSet(sourceID, sourceNodes, relayNodes):
    """
    This function updates the set of the neighboring candidate relays of a 
     source.
    """
    source = sourceNodes[sourceID]
    neighborSet = source.neighborSet
    moveIn = []
    indices = [index for index in range(len(relayNodes)) if index not in neighborSet]
    for ind in indices:
        if euclidean(source.position, relayNodes[ind].position) <= const.RADIUS:
            moveIn.append(ind)
    moveOut = []
    for ind in neighborSet:
        if euclidean(source.position, relayNodes[ind].position) > const.RADIUS:
            moveOut.append(ind)
    source.neighborSet = [nID for nID in neighborSet if nID not in moveOut]
    source.neighborSet.extend(moveIn)
    return None

def update_source_relaySet(sourceID, sourceNodes, relayNodes):
    """
    This function updates the relay set of the source.
    """
    source = sourceNodes[sourceID]
    rSet = source.relaySet
    moveOut = []
    for relayID in rSet:
        if euclidean(source.position, relayNodes[relayID].position) > const.RADIUS:
            moveOut.append(relayID)
    source.relaySet = [rID for rID in rSet if rID not in moveOut]
    return None

def update_player_actionSet_strategy(relayID, sourceNodes, relayNodes):
    """
    This function updates the action set of the player as well as its strategy.
    """
    player = relayNodes[relayID]
    actionSet = player.actionSet
    strategy = player.strategy.copy()
    
    moveOut = []
    for sID in actionSet:
        if euclidean(player.position, sourceNodes[sID].position) > const.RADIUS:
            moveOut.append(sID)
            
    moveIn = []
    inds = [ind for ind in range(len(relayNodes)) if ind not in actionSet]
    for sID in inds:
        if euclidean(player.position, sourceNodes[sID].position) <= const.RADIUS:
            moveIn.append(sID)
    
    out_indices = [actionSet.index(v) for v in moveOut]
    strategy[out_indices] = 0.0
    temp_sum = strategy.sum()*1.0
    for ind in range(len(strategy)):
        if ind not in out_indices:
            strategy[ind] = ((len(actionSet)-len(moveOut))*1.0/(len(actionSet)-len(moveOut)+len(moveIn)))\
            *(strategy[ind]/temp_sum)
    remain_indices = [ind for ind in range(len(strategy)) if ind not in out_indices]
    player.strategy = strategy[remain_indices]#update strategy
    
    if moveIn:
        in_strategy = np.ones((len(moveIn),),dtype=float)/(1.0*(len(actionSet)-len(moveOut)+len(moveIn)))
        player.strategy = np.append(player.strategy, in_strategy)
    
    player.actionSet = [actionSet[ind] for ind in remain_indices]
    player.actionSet.extend(moveIn)#update action set
    
    return None


#%% test functions

def test1():
    actionSet = [1,2,3,4,5]
    strategy = np.array([5,4,3,2,1],dtype=float)
    strategy = strategy/strategy.sum()
    for ind in range(10):
        print Selection(actionSet, strategy)
        
if __name__ == "__main__":
#    test1()
    
    






