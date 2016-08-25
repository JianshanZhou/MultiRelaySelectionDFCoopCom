# -*- coding: utf-8 -*-
"""
Copyright (C) Tue Aug 09 02:19:56 2016  Jianshan Zhou
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
 
This module defines the functions that are used to calculate the exact
 outage probability of a specified multi-relay DF cooperative communication
 system as well as to evaluate the tight lower bounds of the outage probability.
"""

#%% import some other necessary modules
import numpy as np
from itertools import combinations, chain
from constants import Constants
#%% private functions that are not expected to be imported into other programs

const = Constants()

def all_proper_subsets(alist):
    """
    Input
    alist: a list contains all the elements that should not be empty.
    
    Output: a list contains the tuples of all the combinations of the elements in 
            the input list.
    """
    if alist:
        #enumerate all the combinations of the elements in alist
        #it should be noted that when len(alist) is large, the number of all
        # the combinations arises dramatically, which may lead to memory error!
        L = [list(combinations(alist,i+1)) for i in range(len(alist))]
        return list(chain(*L))
    else:
        raise

def A_k(k,lamjd_cSet, lam_id):
    if len(lamjd_cSet) == 0:
        raise
    else:
        if k == 1:
            return 1.0
        else:
            if k == 2:
                return ((-1)**(k-1))/(lamjd_cSet[0]-lam_id)
            else:
                lam = lamjd_cSet[k-2]
                lam_jd = lamjd_cSet[0:k-2]
                dd = lam - lam_jd
                cumprod_result = (dd.cumprod()[-1])*(lam - lam_id)
                return ((-1.0)**(k-1))/cumprod_result

def B_k(k,lamjd_cSet,lam_id):
    if len(lamjd_cSet) == 0:
        raise
    else:
        if k == (1+len(lamjd_cSet)):
            return 1.0
        else:
            if k == 1:
                lam = lam_id
            else:
                lam = lamjd_cSet[k-2]
            
            lamjd_partial = lamjd_cSet[k-1:]
            dd = lamjd_partial - lam
            return 1.0/(dd.cumprod()[-1])
#%% public functions

def outage_prob(lam_ij, p_i, c):
    """
    This function calculates the outage probability of
    a single cooperative link.
    Input
    lam_ij: the exponential distribution parameter
    p_i: the normalized power level of i
    c: the parameter related to the cardinal of the relay set and the data rate
    output
    the outage probability
    """
    return 1.0-np.exp(-c*(lam_ij/p_i))

def coopSet_prob(p_i, lam_cSet, lam_aSet, c):
    """
    Input
    lam_cSet: the exponential distribution parameters between i and other relays
        in the cooperative set, array type
    lam_aSet: the exponential distribution parameters between i and other relays
        in the whole relay set, array type
    p_cSet: the power level used by the source i
    c: the parameter related to the cardinal of the relay set and the data rate    
    Output, a scalar
    the probability of constructing such a cooperative set
    """
    if lam_cSet.any():
        if len(lam_cSet) == len(lam_aSet):
            result_array = np.asarray([1.0-outage_prob(lam_ij, p_i, c) for lam_ij in lam_cSet],dtype = float)
            return result_array.cumprod()[-1]
        else:
            result_array1 = np.asarray([1.0-outage_prob(lam_ij, p_i, c) for lam_ij in lam_cSet], dtype = float)
            result_array2 = np.asarray([outage_prob(lam_ij2, p_i, c) for lam_ij2 in lam_aSet if lam_ij2 not in lam_cSet], dtype = float)
            return result_array1.cumprod()[-1]*result_array2.cumprod()[-1]
    else:
        result_array = np.asarray([outage_prob(lam_ij, p_i, c) for lam_ij in lam_aSet], dtype=float)
        return result_array.cumprod()[-1]

def outage_prob_cSet(lamjd_cSet, lam_id, c):
    if len(lamjd_cSet) == 0:
        raise
    else:
        C = lam_id*lamjd_cSet.cumprod()[-1]
        s = A_k(1,lamjd_cSet,lam_id)*B_k(1,lamjd_cSet,lam_id)*(1-np.exp(-lam_id*c))/lam_id
        for k in np.arange(2,len(lamjd_cSet)+2,dtype=int):
            lam_jd = lamjd_cSet[k-2]
            s += A_k(k,lamjd_cSet,lam_id)*B_k(k,lamjd_cSet,lam_id)*(1-np.exp(-lam_jd*c))/lam_jd
            
        return C*s

def outage_prob_sys_niid(lamij_rSet, lamjd_rSet, lam_id, p_i, pj_rSet):
    """
    This function calculates the exact outage probability of the multi-relay DF
    cooperative communication, given the fading parameters and the power levels
    used by the transmitters.
    Input
    lamij_rSet: an array containing the lamda_ij between i and each j in i's relay
                set
    lamjd_rSet: an array containing the lamda_jd(i) between each j in i's relay
                set and i's destination d(i)
    lam_id: the lamda parameter between the source i and its destination d(i)
    p_i: the normalized power level used by the source i
    pj_rSet: the set of the normalized power levels used by the i's relays, an array
    
    It should be noted that the length of pj_rSet is the same as that of lamjd_rSet
    """
    if len(lamjd_rSet) == 0:
        raise
    else:
        
        N_i = len(lamjd_rSet)
        c = 2**(const.R*(1+N_i))-1.0
        
        #when cSet is not empty
        alist = range(N_i)
        all_subset_index = all_proper_subsets(alist)
        
        outage_prob_list = []        
        
        for index_tuple in all_subset_index:
            
            lamij_cSet = []
            lamjd_cSet = []
            for relay_index in index_tuple:
                lamij_cSet.append(lamij_rSet[relay_index])
                lamjd_cSet.append(lamjd_rSet[relay_index]/pj_rSet[relay_index])
            lamij_cSet = np.asarray(lamij_cSet, dtype = float)
            lamjd_cSet = np.asarray(lamjd_cSet, dtype = float)
            
            prob_cSet = coopSet_prob(p_i, lamij_cSet, lamij_rSet, c)
            conditional_outage_prob = outage_prob_cSet(lamjd_cSet, lam_id/p_i, c)
            outage_prob_list.append(prob_cSet*conditional_outage_prob)
        
        #when cSet is an empty set
        lamij_cSet = np.array([], dtype = float)
        prob_cSet = coopSet_prob(p_i, lamij_cSet, lamij_rSet, c)
        conditional_outage_prob = outage_prob(lam_id, p_i, c)
        outage_prob_list.append(prob_cSet*conditional_outage_prob)
        
        return sum(outage_prob_list)
        
def outage_prob_sys_iid(lam_ij, lam_id, p_i, rSet):
    """
    This function evaluates the outage probability of the multi-relay DF
    cooperative communication in the situation where the channel fading are 
    assumed independent and identically distributed.
    Input
    lam_ij: a float denoting the fading parameter between i and each relay. 
    Since i.i.d. is assumed, all lam_ij are identical.
    lam_id: a float denoting the fading parameter between i and its destination
    d(i), which could be different from those of its relays, i.e., lam_jd.
    p_i: a float denoting the power level used by i.
    rSet: a list containing the indices of i's relays.
    """
    if len(rSet) == 0:
        raise
        
    N_i = len(rSet)
    c = 2**(const.R*(1+N_i)) - 1.0
    
    possible_numbers = range(N_i)
    possible_numbers.append(N_i)
    Comb = lambda n: (np.math.factorial(N_i)/(np.math.factorial(N_i-n)*np.math.factorial(n)))
    f = lambda n: (((c*(lam_id/p_i))**(n))*np.exp(-c*(lam_id/p_i)))/np.math.factorial(n)
    g = lambda n: Comb(n)*((np.exp(-c*lam_ij/p_i))**(n))*((1-np.exp(-c*lam_ij/p_i))**(N_i-n))    
    
    outageProb = 0.0
    for k in possible_numbers:
        possible_l = range(k)
        possible_l.append(k)
        outageProb += ((1.0 - sum([f(l) for l in possible_l]))*g(k))
        
    return outageProb
    
#%% the lower bounds of the outage probability under non-i.i.d fading
def lower_bound1(lamij_rSet, lamjd_rSet, lam_id, p_i, pj_rSet):
    N_i = len(lamjd_rSet)
    c = 2**(const.R*(1+N_i)) - 1.0
    h = lambda lam: 1.0 - np.exp(-c*lam/p_i)
    Prod = 1.0
    for lam in lamij_rSet:
        Prod *= h(lam)
    res1 = Prod*(1.0 - np.exp(-c*lam_id/p_i))
    res2 = outage_prob_cSet(lamjd_rSet/pj_rSet, lam_id/p_i, c)
    
    return res1 + (1-Prod)*res2
    
def lower_bound2(lamij_rSet, lamjd_rSet, lam_id, p_i, pj_rSet):
    N_i = len(lamjd_rSet)
    c = 2**(const.R*(1+N_i)) - 1.0
    h = lambda lam: 1.0 - np.exp(-c*lam/p_i)
    Prod = 1.0
    for lam in lamij_rSet:
        Prod *= h(lam)
    res1 = Prod*(1.0 - np.exp(-c*lam_id/p_i))

    h2 = lambda lam: 1.0 - np.exp(-c*lam/(N_i+1.0))
    Prod2 = 1.0
    for (lam_jd, p_j) in zip(lamjd_rSet, pj_rSet):
        Prod2 *= h2(lam_jd/p_j)
        
    Prod2 *= h2(lam_id/p_i)
    
    return res1 + (1.0 - Prod)*Prod2
    
#%% basic test functions
def test1():
    #test _all_proper_subsets
    alist = [1,2,3]
    print _all_proper_subsets(alist)
    
def test1():
    #test coopSet_prob
    c = 1.0
    p_i = 1.0
    lam_aSet = np.random.rand(5)
    lam_cSet = lam_aSet[0:3].copy()
    print "------lam_cSet is a proper subset of lam_aSet------"
    print coopSet_prob(p_i, lam_cSet, lam_aSet, c)
    print "------lam_cSet is an empty set------"
    print coopSet_prob(p_i, np.array([]), lam_aSet, c)
    print "------lam_cSet == lam_aSet------"
    print coopSet_prob(p_i, lam_aSet, lam_aSet, c)
    
def test3():
    #test A(k)
    lam = 1.0
    lamjd_cSet = np.array([2.0,3.0,4.0,5.0,6.0])
    for k in [1,2,3,4]:
        print "------k=%d------"%(k,)
        print A_k(k,lamjd_cSet,lam)

def test4():
    #test B_k(k)
    lam = 1.0
    lamjd_cSet = np.array([2.0,3.0,4.0,5.0,6.0])
    for k in [1,2,len(lamjd_cSet)]:
        print "------k=%d------"%(k,)
        print B_k(k,lamjd_cSet,lam)

def test5():
    #test A_k(k) and B_k(k)
    lamjd_cSet = np.array([2.0,3.0,4.0,5.0,6.0])
    lam = 1.0
    s = 0.0
    m = len(lamjd_cSet)+1
    for k in range(1,m+1):
        s += A_k(k,lamjd_cSet,lam)*B_k(k,lamjd_cSet,lam)
    print "the sume is %f"%(s,)
    print np.abs(s)<10**(-16)
    
if __name__ == "__main__":
#    test1()
#    test2()
#    test3()
#    test4()
    test5()
    
    