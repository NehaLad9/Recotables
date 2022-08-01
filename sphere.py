#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import sys
print(sys.argv)
from optparse import OptionParser

# parse arguments
usage = "usage: %prog [options]"
parser = OptionParser(usage, description=__doc__)
parser.add_option("--d", dest="distance", type="float", default=None, help="distance from surface of the sphere")
parser.add_option("--r", dest="radius", type="float", default=1, help="Radius of the sphere")
parser.add_option("--samples", dest="samples", type="int", default=100, help="explicitly declare the Δtheta while iterating for different azimuthial values")


opts, args = parser.parse_args()

print("Options Loaded")

print("Arguments: {}".format(args))
print("Options: {}".format(opts))


def create_sphere(x0,y0,z0,r,samples=100):
    phi = np.linspace(0,2*np.pi,samples)
    theta = np.linspace(0,np.pi,samples)
    phi, theta = np.meshgrid(phi,theta)
    x = x0 + r*np.cos(phi)*np.sin(theta)
    y = y0 + r*np.sin(phi)*np.sin(theta)
    z = z0 + r * np.cos(theta)
    return x,y,z



x,y,z = create_sphere(0,0,0,1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z,rstride=1,cstride=1)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show()

"""
d:distance from the surface of a sphere
r:radius of the sphere in question
theta: at which angle is the source
simulating the angular acceptance of a half active sphere
"""
def calculate_effective_area(d,r,theta):
    cone_angle = np.arccos(float(1/(1+r/d)))
    return 2*np.pi*r**2*(1-np.sin(cone_angle + theta))


theta = np.linspace(0,np.pi,100)
d=0.1
r=0.2

values=np.array(theta)
for i,th in enumerate(theta):
    values[i] = calculate_effective_area(d,r,th)
    print(values[i])

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(values,theta)
plt.savefig("thetadependency.png")

