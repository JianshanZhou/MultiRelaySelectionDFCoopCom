# -*- coding: utf-8 -*-
"""
Copyright (C) Thu Sep 01 16:30:08 2016  Jianshan Zhou
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
 
This module develops other two basic relay selection algorithms, one of which is 
 based on the stochastic selection and the other the best relay based.
"""
import DLbSoRS_Alg as alg
import numpy as np

# the stochastic relay selection
def stochastic_selection(player, scenario):
    return np.random.choice(player.actionSet, 1, 
                            replace = False)

# the best relay based selection
def best_relay_selection(player, scenario):
    lamjd = scenario.net_info.lambda_jd_array[player.ID, player.actionSet]
    if len(lamjd) != 0:
        ind = np.argmin(lamjd)
        return player.actionSet[ind]
    else:
        return None

# fixed relay selection
def fixed_relay_selection(player, scenario):
    if player.actionSet:
        return player.current_action
    else:
        return None

def coop(scenario, algorithm_flag = "stochastic_selection"):
    if algorithm_flag == "stochastic_selection":
        relay_selection_scheme = stochastic_selection
    elif algorithm_flag == "best_relay_selection":
        relay_selection_scheme = best_relay_selection
    elif algorithm_flag == "fixed_relay_selection":
        relay_selection_scheme = fixed_relay_selection
    else:
        raise 
    # 1) each player determines the action
    for relay in scenario.relayNodes:
        if len(relay.strategy) == 0 or len(relay.actionSet) == 0:
            continue
        relay.current_action = relay_selection_scheme(relay, scenario)

    # 2) each source constructs its relay set
    for source in scenario.sourceNodes:
        source.relaySet = alg.ConstructRelaySet(source.ID, source.neighborSet,
                                              scenario.relayNodes)                    
        # 3) each source determines the optimal power allocation
        rSet = source.relaySet # a list
        lam_ij = scenario.net_info.lambda_ij_array[source.ID] # iid assumption is adopted
        lam_id = scenario.net_info.lambda_id_array[source.ID]
        lamjd_rSet = scenario.net_info.lambda_jd_array[rSet,source.ID]
        pi_max = source.p_max
        w_i, wj_rSet = alg.solve_optPower(lam_ij, lam_id, lamjd_rSet, rSet, pi_max)
        # 4) set the current power and perform the cooperative communication
        alg.set_power(source.ID, scenario.sourceNodes, scenario.relayNodes, w_i, wj_rSet)
        alg.coopCommunication(scenario.comDt, source.ID, scenario.sourceNodes, scenario.relayNodes)
        
    # 5) each relay determines its payoff as well as normalizes it
    for relay in scenario.relayNodes:
        if len(relay.strategy) == 0 or len(relay.actionSet) == 0:
            continue
        relay.reward = alg.get_reward(relay.ID, scenario.sourceNodes, 
                                              scenario.relayNodes) # update the reward
        alg.update_reward_record(relay.ID, scenario.relayNodes)
        alg.update_normalized_reward(relay.ID, scenario.relayNodes)
        # relay records its payoff
        relay.reward_trace.append(relay.normalized_reward)




