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
import copy
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
    randomNum = np.random.random(1)    
    cumProb = strategy.cumsum()
    if len(cumProb) == 0:
        raise
    for index in range(len(cumProb)):
        if cumProb[index]>=randomNum:
            break
    return actionSet[index]

# The random walk selection
def RWS(actionSet, strategy):
    r = np.random.random(1)
    accumulateProList = np.add.accumulate(strategy)
    indexList = np.arange(len(strategy))
    indexList = indexList[r<=accumulateProList]
    return actionSet[indexList[0]]#return the first element in the indexList
 

def neighborRelayID_with_maxReward(neighborSet, relayNodes):
    """
    This function determines the id of the neighboring relays who has the
     maximum residual energy currently.
     Input
     neighborSet: a list containing the IDs of a source's neighboring relays.
     relayNodes: a list containing the instances of all the relays in the netiwork.
     Output
     The ID of the selected relay, an int number.
    """
    relayID = neighborSet[0]
    for rID in neighborSet:
        if relayNodes[rID].residual_energy>relayNodes[relayID].residual_energy:
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
    if neighborSet:
        for neighborID in neighborSet:
            if relayNodes[neighborID].current_action == sID:
                rSet.append(neighborID)
        return rSet        
#        if rSet:
#            return rSet
#        else:
#            rSet.append(neighborRelayID_with_maxReward(neighborSet,
#                                                       relayNodes))
#            return rSet
    else:
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
    xmin = 0.01
    xmax = 10*np.log10(pi_max)
    #x is in dB
    func = lambda x: const.BETA - outage_prob_sys_iid(lam_ij, lam_id, 10.0**(x/10.0), rSet)
    
#    lowerB = func(xmin)
#    print lowerB
#    upperB = func(xmax)
#    print upperB
#    if lowerB*upperB > 0.0:
#        print lowerB, upperB
#        raise
    
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
#    if player.reward < player.reward_min:
#        player.reward_min = player.reward
#        return None
#    elif player.reward > player.reward_max:
#        player.reward_max = player.reward
#        return None
#    else:
#        return None
    if player.reward > player.reward_max:
        player.reward_max = player.reward
        
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
        try:
            if euclidean(source.position, relayNodes[ind].position) <= const.RADIUS:
                moveIn.append(ind)
        except:
            print "source-%d position:"%(source.ID, )
            print source.position
            print "relay-%d position:"%(relayNodes[ind].ID,)
            print relayNodes[ind].position
            raise
    
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
    # determine the set of actions that move out of the neighbor of the player
    moveOut = []
    for sID in actionSet:
        if euclidean(player.position, sourceNodes[sID].position) > const.RADIUS:
            moveOut.append(sID)
    # determine the set of sources that move in the neighbor of the player        
    moveIn = []
    inds = [ind for ind in range(len(sourceNodes)) if ind not in actionSet]
    for sID in inds:
        if euclidean(player.position, sourceNodes[sID].position) <= const.RADIUS:
            moveIn.append(sID)
    
    out_indices = [actionSet.index(v) for v in moveOut] # the indices of the out-actions in the action set
    strategy[out_indices] = 0.0
    temp_sum = strategy.sum()*1.0
#    for ind in range(len(strategy)):
#        if ind not in out_indices:
#            strategy[ind] = ((len(actionSet)-len(moveOut))*1.0/(len(actionSet)-len(moveOut)+len(moveIn)))\
#            *(strategy[ind]/temp_sum)
    remain_indices = [ind for ind in range(len(strategy)) if ind not in out_indices]
    for ind in remain_indices:
        strategy[ind] = ((len(actionSet)-len(moveOut))*1.0/(len(actionSet)-len(moveOut)+len(moveIn)))\
        *(strategy[ind]/temp_sum)
    player.strategy = strategy[remain_indices] # hold the probs of actions that remain in the neighbor of the player
    
    if moveIn:
        in_strategy = np.ones((len(moveIn),),dtype=float)/(1.0*(len(actionSet)-len(moveOut)+len(moveIn)))
        # the prob corresponding to the added action is appended into the original strategy vector        
        player.strategy = np.append(player.strategy, in_strategy) 
    
    player.actionSet = [actionSet[ind] for ind in remain_indices] # hold the actions that remain in the neighbor of the player
    player.actionSet.extend(moveIn)#update action set
    
    return None

# unilateral decision making for the unilateral tests
def unilateral_change(player, scenario):
    current_action = player.current_action
    temp_actionSet = copy.deepcopy(player.actionSet)
    temp_actionSet.remove(current_action)
    #player.current_action = np.random.choice(temp_actionSet)
    rSet_size = len(scenario.sourceNodes[temp_actionSet[0]].relaySet)
    target_action = temp_actionSet[0]
    for changed_to_action in temp_actionSet:
        if len(scenario.sourceNodes[changed_to_action].relaySet)>rSet_size:
            rSet_size = len(scenario.sourceNodes[changed_to_action].relaySet)
            target_action = changed_to_action
    player.current_action = target_action
    return (current_action, player.current_action)

def unilateral_update(scenario, playerID):
    """ 
    the input scenario is the resulting network situation
    the playerID is the ID of the player that unilaterally changes its action
    """
    # the player with the input playerID unilaterally changes its action
    (old_action, new_action) = unilateral_change(scenario.relayNodes[playerID],
                                                 scenario)
    # each source re-constructs its relay set
    for source in scenario.sourceNodes:
        source.relaySet = ConstructRelaySet(source.ID, 
                                            source.neighborSet,
                                            scenario.relayNodes)
        # re-optimize the transmission power levels
        rSet = source.relaySet
        lam_ij = scenario.net_info.lambda_ij_array[source.ID]
        lam_id = scenario.net_info.lambda_id_array[source.ID]
        lamjd_rSet = scenario.net_info.lambda_jd_array[rSet, source.ID]
        pi_max = source.p_max
        w_i, wj_rSet = solve_optPower(lam_ij, lam_id, lamjd_rSet, rSet, pi_max)
        set_power(source.ID, scenario.sourceNodes, scenario.relayNodes, w_i, wj_rSet)
        coopCommunication(scenario.comDt, source.ID, 
                          scenario.sourceNodes, scenario.relayNodes)
    # evaluate the reward of unilateral change
    for relay in [scenario.relayNodes[playerID]]:
        relay.reward = get_reward(relay.ID, scenario.sourceNodes,
                                  scenario.relayNodes)
        update_reward_record(relay.ID, scenario.relayNodes)
        update_normalized_reward(relay.ID, scenario.relayNodes)

    return scenario.relayNodes[playerID].normalized_reward

#%% test functions

def test1():
    actionSet = [1,2,3,4,5]
    strategy = np.array([5,4,3,2,1],dtype=float)
    strategy = strategy/strategy.sum()
    for ind in range(10):
        print Selection(actionSet, strategy)
        
if __name__ == "__main__":
    test1()
    
    
    