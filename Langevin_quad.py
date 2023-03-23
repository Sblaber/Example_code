#Load all the packages I might need
from __future__ import division # must be first
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy import integrate
from scipy.linalg import expm
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.special import erfi
from scipy.optimize import minimize
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
#plt.style.use("dark_background") #Use this command to plot in dak mode
from datetime import datetime
startTime = datetime.now()
print(startTime)
num_sims = 10000#0 # number of Langevin trajectories
N = 3201 # number of time points
N2 = 1 # number of protocol durations
# Define important observables
deltat = np.zeros(N2) # protocol duration (will be scaled by tau_s)
delta_x_f = np.zeros(N2) # average distance from final eq. position
W_avg = np.zeros(N2) # average work
W_var = np.zeros(N2) # variance of work distribution
x_var = np.zeros(N2) # variance of position distribution
epsilon = np.zeros(N2) # fraction of trajectories that don't cross the barrier
mu_fin_vec = np.zeros(N2) # optimal final position

## System Parameters ##
# Assuming k_B = gamma = 1
u0 = -1.0 # initial trap centre
uf = 1.0 # final trap centre
k0 = 4.0 # initial stiffness
kf = 4.0 # final stiffness
k = 4.0 # stiffness
E_b = 4.0 # barrier height
tau_s = 1.0/k0 # initial relaxation time of the trap
u_STEP = (uf+u0)/2.0 # STEP value (optimal fast protocol)

t_init = 0
t_end  = 100*tau_s # protocol duration
ts = np.linspace(t_init,t_end,N) # time
dt     = ts[1]-ts[0] # time step
x_init = 0
c_sigma = np.sqrt(2)
x_tF    = np.zeros((num_sims,N))

## Compute initial distribution ##
x_distribution = np.linspace(-2,2,1601)
V_trap0 = 0.5*k0*(x_distribution-u0)**2.0 # Initial trap potential
V = E_b*(x_distribution**2.0-1.0)**2.0 # Energy landscape (double well)
V_tot0 = V + V_trap0
B_factor0 = np.exp(-V_tot0)
Z0 = np.trapz(B_factor0,x_distribution)
p0 = B_factor0/Z0
CDF0 = integrate.cumtrapz(p0,x_distribution,initial = 0)
x_avg_0 = np.trapz(x_distribution*p0,x_distribution)
sigma_0 = (np.trapz((x_distribution-x_avg_0)**2.0*p0,x_distribution))**0.5

## Compute final equilibrium distribution ##
V_trapf = 0.5*kf*(x_distribution-uf)**2.0
V_totf = V + V_trapf
B_factorf = np.exp(-V_totf)
Zf = np.trapz(B_factorf,x_distribution)
pf = B_factorf/Zf
x_avg_f = np.trapz(x_distribution*pf,x_distribution)
sigma_f = (np.trapz((x_distribution-x_avg_f)**2.0*pf,x_distribution))**0.5


## Sample from initial equilibrium distribution by inverse transform sampling ##
def f(x,*rand):
    """Invert the CDF"""
    return  np.interp(x,x_distribution,CDF0) - rand

def sample(rand):
    """map random number onto inverse CDF"""
    starting_guess = -1.0
    return fsolve(f, starting_guess, args = rand,factor=1)

## Langevin Simulation ##
# Functions for Langevin Simulation in Euler-Muryama Scheme
def mu_F(x, kt, ut):
    """Implement mu forward."""
    return -kt * (x - ut) - 4.0 * E_b * x * (x**2.0-1.0)

def dW(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(0, np.sqrt(delta_t),num_sims)

#Function for computing optimal protocol
def g(x,duration):
    return  0.5*kf*(x-uf)**2.0 + (1.0/duration)*(x-x_avg_0)**2.0 + E_b*(x**2.0-1)**2.0 + 0.5*sigma_0**2.0*E_b*(12.0*x**2.0)

# Loop over protocol durations
for l in range(0,N2):
    t_init = 0
    t_end  = 10*tau_s*2.0**(l) # increment protocol duation here
    ts = np.linspace(t_init,t_end,N)
    dt = ts[1]-ts[0]
    x_tF = np.zeros((num_sims,N))

    # Compute optimal final mean and variance
    mu_fin_guess = (kf*uf+(2.0/ts[-1])*x_avg_0)/(kf+(2.0/ts[-1]))
    mu_fin = minimize(g, mu_fin_guess,args = ts[-1]).x
    mu_fin_vec[l] = 1.0*mu_fin
    mu_opt = x_avg_0 + ((mu_fin-x_avg_0)/ts[-1])*ts
    sigma_opt = sigma_0
    # Compute optimal trap centre and stiffness
    ks = k0 + 12.0*E_b*mu_opt[0]**2.0 - 12.0*E_b*mu_opt**2.0
    u = mu_opt + (((mu_fin-x_avg_0)/ts[-1])/ks) + 4*E_b*mu_opt*(mu_opt**2.0-1)/ks

    # Calculate control parameter velocities and fix endpoins
    udot = np.gradient(u,ts)
    kdot = np.gradient(ks,ts)
    ks[0] = k0
    ks[-1] = kf
    u[0] = u0
    u[-1] = uf

    # run the Langevin Simulation
    for j in range(0,num_sims):
        x_tF[j,0] = sample(np.random.uniform(0, 1, 1)[0])
    for i in range(1, N):
        t = (i-1) * dt
        kt = ks[i]
        ut = u[i]
        x = x_tF[:,i-1]
        x_tF[:,i] = x + mu_F(x, kt, ut) * dt + c_sigma * dW(dt)

    # compute averages, variances and works
    u_sample = np.einsum('ij,j->ij',1+0*x_tF,u)
    udot_sample = np.einsum('ij,j->ij',1+0*x_tF,udot)
    kdot_sample = np.einsum('ij,j->ij',1+0*x_tF,kdot)
    W_jump_0 = 0.5*ks[1]*(x_tF[:,0]-u[1])**2.0-0.5*k0*(x_tF[:,0]-u0)**2.0 # work done in initial jump
    W_jump_f = 0.5*kf*(x_tF[:,-1]-uf)**2.0 - 0.5*ks[-2]*(x_tF[:,-1]-u[-2])**2.0 #work done in final jump
    W_u = -np.trapz(ks*udot_sample*(x_tF-u_sample), ts,axis = 1) # work done by trap centre
    W_k = 0.5*np.trapz(kdot_sample*(x_tF-u_sample)**2.0,ts,axis = 1) # work done by stiffness
    W = W_u + W_k + W_jump_0 + W_jump_f # total work
    x_avg = np.mean(x_tF,axis = 0) # average position
    W_avg[l] = np.mean(W) # average work
    W_var[l] = np.var(W) # work variances
    x_var[l] = np.var(x_tF[:,-1]) #position variance at the end of the protocol
    deltat[l] = ts[-1]/tau_s # Scaled protocol duration
    delta_x_f[l] = x_avg_f-x_avg[-1]
    epsilon[l] =  np.sum(x_tF[:,-1] < 0)/(1.0*size(x_tF[:,-1]))

W_instant = -(uf - u0)*k0*np.trapz(p0*x_distribution,x_distribution) + k0*np.trapz(udot*u,ts) # Work done by instantaneous protocol
F = 0.0 # free energy change

# Plot initial and final position distribution in red and blue respectively
plt.figure(0)
count, bins, ignored = plt.hist(x_tF[:,0], bins = 'auto', normed=True,color='red',alpha = 0.5)
count, bins, ignored = plt.hist(x_tF[:,-1], bins = 'auto', normed=True,color='blue',alpha = 0.5)
plt.plot(x_distribution,p0,'r')
plt.plot(x_distribution,pf,'b')
plt.xlim(-2,2)

## Animation ##
# Define time dependent variable for animation
x_sqrd_t = (np.mean(x_tF,axis=0)-u)**2.0

V_trap_t = 0.5*ks*x_sqrd_t
V_t = E_b*(np.mean(x_tF,axis=0)**2.0-1.0)**2.0#E_b*(np.mean(x_tF,axis = 0)**2.0-1.0)**2.0
V_tot_t = V_trap_t + V_t

std = np.std(x_tF,axis = 0)

# plot static curves
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-1, 8))
plt.plot(x_distribution,V,'grey',label = '$V_{\\rm B}(x)$',linewidth = 4.0)
plt.xticks([-2,0,2],fontsize = 16)
plt.axis('off')

# animation function.  This is called sequentially
x1,y1,y2,y3,y4 = [], [], [], [], []

line1, = ax.plot([], [], 'r',linewidth = 4.0,alpha = 0.5)
line3, = ax.plot([], [], 'g',linewidth = 4.0,alpha = 0.5)
line2, = ax.plot([], [], 'bo',markersize = 16.0)
line4, = ax.plot([], [], 'b',linewidth = 4.0)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    return line1, line3, line2, line4,

def animate(i):
    x1 = x_distribution
    y1 = 0.5*ks[8*i]*(x_distribution-u[8*i])**2.0
    x2 = np.mean(x_tF[:,8*i],axis=0)#np.mean(x_tF,axis = 0)[8*i]#x_tF[0,i]#np.mean(x_tF,axis = 0)[8*i]
    y2 = V_tot_t[8*i]
    x3 = 1.0*x_distribution
    y3 = 0.5*ks[8*i]*(x_distribution-u[8*i])**2.0+E_b*(x_distribution**2.0-1.0)**2.0
    x4 = [np.mean(x_tF[:,8*i],axis=0)-std[8*i],np.mean(x_tF[:,8*i],axis=0)+std[8*i]]
    y4 = [y2,y2]
    # update the line data
    line1.set_data([x1], [y1])
    line2.set_data([x2], [y2])
    line3.set_data([x3], [y3])
    line4.set_data([x4], [y4])

    return line1, line3, line2, line4,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(3200/8), interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation
anim.save('Quad_Opt_10tau_4k.mp4', writer=writer)

## Plot the quantiles of the distribution ##
# These quantiles are evenly spaced for a Gaussian distribution
Q9 = np.quantile(x_tF[:,:], 0.09, axis=0)
Q25 = np.quantile(x_tF[:,:], 0.25, axis=0)
Q5 = np.quantile(x_tF[:,:], 0.5, axis=0)
Q75 = np.quantile(x_tF[:,:], 0.75, axis=0)
Q91 = np.quantile(x_tF[:,:], 0.91, axis=0)


fig = plt.figure(figsize=(4,6), dpi=160)
plt.plot(Q5,ts/ts[-1],'b',linewidth = 2.0)
plt.fill_betweenx(ts/ts[-1], Q9,Q91,alpha = 0.25,color = 'b')
plt.fill_betweenx(ts/ts[-1], Q25,Q75,alpha = 0.25,color = 'b')
#plt.fill_betweenx(ts/ts[-1], Q6,Q2-(Q6-Q2),alpha = 0.25,color = 'b')
plt.xlim(-4,4)
plt.ylim(0,1)
#plt.ylabel('$t/\\Delta t$',fontsize = 18)
plt.axis('off')
plt.tight_layout()




plt.show()
print(datetime.now()-startTime)
