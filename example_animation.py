#Load all the packages I might need
from __future__ import division # must be first
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
#rc('font', **{'family':'serif','serif':['Palatino']}) # Use this to set the font for labels
rc('text', usetex=True)
#plt.style.use("dark_background") # Use this to plot in dark theme

#############################
# This code generates an animated figure of a particle moving across a double
# well potential with a moving quadratic potential. The double well potential is
# static.
#############################

# Generate data to be plotted
# Static courve
x_1 = np.linspace(-2,2,401) # x coordinate for double well potential
y_1 = (x_1**(2.0)-1)**2.0 # Static double well potential energy landscape
# Dynamic curves
t = np.linspace(0,1,101) # time
# Quadratic potential moving from left to right well
x_2 = np.linspace(-2,2,401)
x_2_t = -1+2.0*t # centre of the trap increases linearly with time from -1 to 1
y_2_t = (np.einsum('i,j->ij',x_2,1.0+0*t) - np.einsum('i,j->ij',1.0+0*x_2,x_2_t))**2.0 # 2D array of a moving parabola as a function of x_2,t
y_2 = (x_1-x_2)**2.0
# Particle moving from left to right well
x_3 = -1+2.0*t # Particles x coordinate increases linearly with time from -1 to 1
y_3 = (x_3**(2.0)-1)**2.0+(x_3-x_2_t)**2.0 # Energy of the particle including trap and landscape (y coordinate)
# Quadratic potential moving from left to right well

# Set up the figure with static background curves (x_1,y_1)
fig, ax = plt.subplots()
ax = plt.axes(xlim=(-2, 2), ylim=(-0.2, 2))
ax.plot(x_1,y_1,'k',label = '$y_{1}(x_{1})$',linewidth = 2.0)
plt.xticks([-2,0,2],fontsize = 16)
plt.yticks([0,1,2],fontsize = 16)
#ax.axis('off')

# Initialize the curves and define their linestyle
line2, = ax.plot([], [], 'r',linewidth = 2.0)
line3, = ax.plot([], [], 'bo',markersize = 32.0)

# initialization for the animation
def init():
    line2.set_data([], [])
    line3.set_data([], [])
    return line2, line3,

# animation function.  This is called sequentially
def animate(i):
    # update the line data. Variable i indexes time.
    line2.set_data([x_2[:]], [y_2_t[:,i]])
    line3.set_data([x_3[i]], [y_3[i]])

    return line2, line3,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=np.size(t), interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# https://matplotlib.org/stable/api/animation_api.html
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

#anim.save('Video.mp4', writer=writer)

plt.show()
