# -*- coding: utf-8 -*-
"""
Copyright (C) Tue Aug 09 22:27:34 2016  Jianshan Zhou
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
 
This module carries out a series of numerical experiments where the outage
 probability of the multi-relay DF cooperative communication as well as the 
 different lower bounds are calculated and analyzed.
"""
#%% import some basic modules
import numpy as np
import matplotlib.pyplot as plt

from outageProb import outage_prob_sys_niid,\
 outage_prob_sys_iid, lower_bound1, lower_bound2

#%% common settings on the plots
labelfont = {"family": "serif",
             "weight": "normal",
             "size": 20,
             "color": "black"}
legendfont = {"family":labelfont["family"],
              "size":labelfont["size"]}
xlabelstr = "Normalized power level ${p_{i}}/{\sigma_{0}^{2}}$ (dB)"
ylabelstr = "Outage probability $\mathit{Pr}( I_{i,d(i)} < r_{i})$"
fsize = (10,8)
linewidth = 8.0
linestyle = ["-","--","-.","-","--",":"]
linecolor = ['#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
             '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
             '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
             '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
#%% The following experiments are conducted under the i.i.d assumption.
# It is noted that the exponential distribution parameter lambda between any
#  two nodes i and j is in proportion to (distance(i,j))^(alpha) where alpha
#  alpha characterizes the path loss effect. 

relay_num = np.array([1, 3, 5, 7, 9],dtype = int)
#nind = 2

#N_i = relay_num[nind]
#rSet = range(1,N_i+1)#indices of the relays

lam_id = 1.0
lam_ij = 2.0
#lamjd_rSet = minLambda + (maxLambda-minLambda)*np.random.rand(N_i)
#lamij_rSet = lam_ij*np.ones((N_i,), dtype = float)

pi_array = np.linspace(1,40,100)# the normalized power levels can be used by i (dB)
pi_array2 = 10**(pi_array/10)

outage_prob_vector = []
flag2 = 0
for N_i in relay_num:
    rSet = range(1,N_i+1)#indices of the relays
    outage_prob_vector.append(np.zeros_like(pi_array, dtype=float))
    flag = 0
    for p_i in pi_array2:
        outage_prob_vector[flag2][flag] = outage_prob_sys_iid(lam_ij, lam_id, p_i, rSet)  
        flag += 1
    flag2 += 1
    

#%% plot the results
plt.figure(0,figsize = fsize)
plt.hold(True)

for nind in range(len(relay_num)):
    plt.semilogy(pi_array,outage_prob_vector[nind],
                 label = "$N_{i}$=%d"%(relay_num[nind],),
                 lw = linewidth,
                 c = linecolor[nind],
                 ls = linestyle[nind])
             
plt.xlabel(xlabelstr,fontdict=labelfont)
plt.ylabel(ylabelstr,fontdict=labelfont)
plt.legend(prop=legendfont,bbox_to_anchor=(0.35,0.48))
plt.xticks(fontsize=labelfont["size"],fontname=labelfont["family"])
plt.yticks(fontsize=labelfont["size"],family=labelfont["family"])

plt.grid(True)

#%% The following experiments are conducted under the assumption of non-i.i.d.
# The exact outage probability is compared with two different lower bounds.
# The exponential distribution parameters associated with the channels from
# the source i to each relay j and from i and j to the destination d(i) are 
# specified by following the uniform distribution in [1.0,2). 
np.random.seed(4)
relay_num = np.array([4, 8],dtype = int)

pi_array = np.linspace(1,40,100)# the normalized power levels can be used by i (dB)
pi_array2 = 10**(pi_array/10)

lam_id = 1.0

minLambda = 1.0
maxLambda = 2.0

outage_prob_exact = []
lower_bound_1 = []
lower_bound_2 = []

lambdaij_array = []
lambdajd_array = []

flag1 = 0
for N_i in relay_num:
    
    rSet = range(1,N_i+1)#indices of the relays
    outage_prob_exact.append(np.zeros_like(pi_array, dtype=float))
    lower_bound_1.append(np.zeros_like(pi_array, dtype=float))
    lower_bound_2.append(np.zeros_like(pi_array, dtype=float))
    
    lamjd_rSet = minLambda + (maxLambda-minLambda)*np.random.rand(N_i)    
    lamij_rSet = minLambda + (maxLambda-minLambda)*np.random.rand(N_i)
    lambdaij_array.append(lamij_rSet)
    lambdajd_array.append(lamjd_rSet)
    flag2 = 0
    for p_i in pi_array2:
        #adopting the identical powers at each relay
        pj_rSet = (np.ones((N_i,), dtype = float))*p_i#the normalized power used by each relay
        outage_prob_exact[flag1][flag2] = outage_prob_sys_niid(lamij_rSet, lamjd_rSet, lam_id, p_i, pj_rSet)
        lower_bound_1[flag1][flag2] = lower_bound1(lamij_rSet, lamjd_rSet, lam_id, p_i, pj_rSet)
        lower_bound_2[flag1][flag2] = lower_bound2(lamij_rSet, lamjd_rSet, lam_id, p_i, pj_rSet)
        flag2 += 1
    
    flag1 += 1

#%% plot the results

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

xlabelstr2 = "Normalized power level ${p_{i}}/{\sigma_{0}^{2}}$ (${p_{j}}/{\sigma_{0}^{2}}$) (dB)"
ylabelstr2 = "Outage probability"

plt.figure(1,figsize = fsize)
plt.hold(True)

for nind in range(len(relay_num)):
    plt.semilogy(pi_array,outage_prob_exact[nind],
                 label = "$\mathit{Pr}(I_{i,d(i)} < r_{i})$ at $N_{i}$=%d"%(relay_num[nind],),
                 lw = linewidth,
                 c = linecolor[nind*3],
                 ls = linestyle[0])
    plt.semilogy(pi_array,lower_bound_1[nind],
                 label = "$\mathit{Pr}_{1}^{lower}(I_{i,d(i)} < r_{i})$ at $N_{i}$=%d"%(relay_num[nind],),
                 lw = linewidth,
                 c = linecolor[nind*3+1],
                 ls = linestyle[1])
    plt.semilogy(pi_array,lower_bound_2[nind],
                 label = "$\mathit{Pr}_{2}^{lower}(I_{i,d(i)} < r_{i})$ at $N_{i}$=%d"%(relay_num[nind],),
                 lw = linewidth,
                 c = linecolor[nind*3+2],
                 ls = linestyle[2]) 
    
plt.xlabel(xlabelstr2,fontdict=labelfont)
plt.ylabel(ylabelstr2,fontdict=labelfont)
plt.legend(prop=legendfont,bbox_to_anchor=(0.65,0.58))
plt.xticks(fontsize=labelfont["size"],fontname=labelfont["family"])
plt.yticks(fontsize=labelfont["size"],family=labelfont["family"])
plt.grid(True)

ax = plt.gca()
zoomNum = 20
axins1 = zoomed_inset_axes(ax, zoomNum, 
                           loc=2)
axins1.get_xaxis().get_major_formatter().set_useOffset(False)
plt.xticks(visible = False)
plt.yticks(visible = False)
nind = 0
axins1.semilogy(pi_array,lower_bound_1[nind],
             label = "$\mathit{Pr}_{1}^{lower}(I_{i,d(i)} < r_{i})$ at $N_{i}$=%d"%(relay_num[nind],),
             lw = linewidth,
             c = linecolor[nind*3+1],
             ls = linestyle[1])
axins1.semilogy(pi_array,lower_bound_2[nind],
             label = "$\mathit{Pr}_{2}^{lower}(I_{i,d(i)} < r_{i})$ at $N_{i}$=%d"%(relay_num[nind],),
             lw = linewidth,
             c = linecolor[nind*3+2],
             ls = linestyle[2]) 
x1, x2, y1, y2 = 20.6731, 21.0998, 0.00157676, 0.00235482
axins1.set_xlim(x1,x2)
axins1.set_ylim(y1,y2)
mark_inset(ax, axins1, loc1=1, loc2=4, fc="none")
plt.draw()

zoomNum1=12000
axins2 = zoomed_inset_axes(ax, zoomNum1, 
                           loc=1)
axins2.get_xaxis().get_major_formatter().set_useOffset(False)
plt.xticks(visible=False)
plt.yticks(visible = False)
nind = 1
axins2.semilogy(pi_array,lower_bound_1[nind],
             label = "$\mathit{Pr}_{1}^{lower}(I_{i,d(i)} < r_{i})$ at $N_{i}$=%d"%(relay_num[nind],),
             lw = linewidth,
             c = linecolor[nind*3+1],
             ls = linestyle[1])
axins2.semilogy(pi_array,lower_bound_2[nind],
             label = "$\mathit{Pr}_{2}^{lower}(I_{i,d(i)} < r_{i})$ at $N_{i}$=%d"%(relay_num[nind],),
             lw = linewidth,
             c = linecolor[nind*3+2],
             ls = linestyle[2]) 
x_1, x_2, y_1, y_2 = 29.6221, 29.6229, 0.0043036, 0.00430469
axins2.set_xlim(x_1,x_2)
axins2.set_ylim(y_1,y_2)
mark_inset(ax, axins2, loc1=2, loc2=4, fc="none")
plt.draw()

plt.show()

#%% The following experiments analyzes the variation of the outage probability
# over the power levels of the source and the relays.
# Here, the total number of the relays is set to 3.

np.random.seed(0)

N_i = 2

#pNum = 100
#pi_array = np.linspace(1,40,pNum)# the normalized power levels can be used by i (dB)

pi_array = np.arange(5,5+16,dtype=float)+4
pNum=len(pi_array)
pi_array2 = 10**(pi_array/10)
pj1_array2 = pi_array2.copy()
pj2_array2 = pi_array2.copy()

pointNum = len(pi_array)

lam_id = 1.0
minLambda = 1.0
maxLambda = 2.0
lamij_rSet = minLambda + (maxLambda-minLambda)*np.random.rand(N_i)
lamjd_rSet = minLambda + (maxLambda-minLambda)*np.random.rand(N_i)

outage_prob_vector = np.zeros((pointNum, pointNum, pointNum),\
 dtype=float)

for index1 in range(pointNum):
    for index2 in range(pointNum):
        for index3 in range(pointNum):
            p_i = pi_array2[index1]
            pj_rSet = np.array([pj1_array2[index2], pj2_array2[index3]], dtype=float)
            outage_prob_vector[index1,index2,index3] = outage_prob_sys_niid(lamij_rSet,\
            lamjd_rSet, lam_id, p_i, pj_rSet)
#%% plot the results with Mayavi
from mayavi import mlab
mlab.figure(1,size=(800, 750), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
fh = mlab.pipeline.volume(mlab.pipeline.scalar_field(outage_prob_vector))
mlab.outline(fh,color=(0,0,0))
mlab.axes(fh, color=(.7, .7, .7),
            ranges=(0, 40, 0, 40, 0, 40), xlabel='p_i', ylabel='p_j1',zlabel='p_j2',
            y_axis_visibility=True,
            x_axis_visibility=True, z_axis_visibility=True)
mlab.view(40, 85)
mlab.colorbar
mlab.show()
#%% plot the results with matplotlib

def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#outage_prob_vector = np.load("outage_prob_niid_2relays.npy")

fig = plt.figure(2,figsize=(10, 8))
fig.clf()

# prepare results
xn = 4
#pi_indices = np.arange(0,0+(xn**2)*3,3)
pi_indices = range(pNum)
Zdata = [outage_prob_vector[pi_index,::,::]\
 for pi_index in pi_indices]
extent = pi_array[0], pi_array[-1], pi_array[0], pi_array[-1]

grid = ImageGrid(fig,111,
                 nrows_ncols=(xn,xn),
                 direction="row",
                 axes_pad=0.05,
                 add_all=True,
                 label_mode="1",
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.05,
                 )

#vmax = 1.0
#vmin = 0.0
vmax = 0.0
vmin = -np.ceil(np.abs(np.log10(outage_prob_vector.min())))

import matplotlib.colors
norm = matplotlib.colors.Normalize(vmax=vmax,
                                   vmin=vmin)

for ax, z in zip(grid,Zdata):
    zz = np.log10(z)
    im = ax.imshow(zz,norm=norm,
                   origin="lower",extent=extent,
                   interpolation="nearest")
                   
# With cbar_mode="single", cax attribute of all axes are identical.
ax.cax.colorbar(im)
ax.cax.toggle_label(True)
ax.cax.set_yticks(np.linspace(vmin,vmax,4))
ax.cax.set_yticklabels(["$10^{%.0f}$"%(tv,) for tv in np.linspace(vmin,vmax,4)])
axis = ax.cax.axis[ax.cax.orientation]

axis.label.set_text("Outage probability")
axis.label.set_fontname(labelfont["family"])
axis.label.set_size(labelfont["size"])
axis.major_ticklabels.set_size(labelfont["size"])
axis.major_ticklabels.set_fontname(labelfont["family"])

inner_labels = ["$p_{i}$=%.0f(dB)"%(pi_array[pi_index],)\
 for pi_index in pi_indices]
for ax, im_title in zip(grid,inner_labels):
    t = add_inner_title(ax, im_title, loc=2, size=legendfont)
    t.patch.set_ec("none")
    t.patch.set_alpha(0.5)
    
#grid[6].set_xticks([10,20,30,40])
#grid[6].set_yticks([10,20,30,40])
lind = xn*(xn-1)
grid[lind].set_xticks([pi_array[0],pi_array[7],pi_array[-1]])
grid[lind].set_yticks([pi_array[0],pi_array[7],pi_array[-1]])
grid[lind].set_xticklabels(grid[6].get_xticks(),fontdict=labelfont)
grid[lind].set_yticklabels(grid[6].get_yticks(),fontdict=labelfont)

grid[lind].set_xlabel("$p_{j^{\prime}}$ (dB)",fontdict=labelfont)
grid[lind].set_ylabel("$p_{j^{\prime\prime}}$ (dB)",fontdict=labelfont)

plt.draw()
plt.show()