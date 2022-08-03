from icecube.icetray import I3Units, I3Module, traysegment
from icecube.clsim import GetIceCubeDOMAcceptance, GetIceCubeDOMAngularSensitivity
import numpy as np
import matplotlib.pyplot as plt
import argparse
from optparse import OptionParser

# parse arguments
usage = "usage: %prog [options]"
parser = OptionParser(usage, description="Get the effective area of a sphere depending on polar angle")

parser.add_option("--dist", dest="dist", type="float", default=None, help="distance from surface of the sphere")
parser.add_option("--r", dest="r", type="float", default=1, help="Radius of the sphere")
parser.add_option("--samples", dest="samples", type="int", default=100, help="explicitly declare the Δtheta while iterating for different azimuthial values")


opts, _  = parser.parse_args()

print("Options Loaded")

print("Options: {}".format(opts))

dist = opts.dist
r = opts.r
samples = opts.samples
theta = np.linspace(0,np.pi,samples)
costheta=np.cos(theta)


def create_sphere(x0,y0,z0,r,samples=100):
    phi = np.linspace(0,2*np.pi,samples)
    theta = np.linspace(0,np.pi,samples)
    phi, theta = np.meshgrid(phi,theta)
    x = x0 + r*np.cos(phi)*np.sin(theta)
    y = y0 + r*np.sin(phi)*np.sin(theta)
    z = z0 + r * np.cos(theta)
    return x,y,z

x,y,z = create_sphere(0,0,0,1,samples=samples)
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z,rstride=1,cstride=1)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.show()
"""

####an infinite plane at angle theta

def effective(cth,r):
    return np.pi*r**2*(1+cth)/2

def find_projection(r,theta):
    total_area = np.pi*r**2
    values=np.array(theta)
    fraction=np.array(theta)
    for i, th in enumerate(theta):
        values[i] = effective(np.cos(th),r)
        #print(values[i])
        fraction[i] = values[i]/total_area
        #print(f"fraction is: {fraction[i]}" )
    return fraction,values

def ang_sens(costheta):
#angular_sensitivity is a function
    angular_sensitivity = GetIceCubeDOMAngularSensitivity()
    sensitivity = angular_sensitivity.GetValue(1)
    print(sensitivity,angular_sensitivity.GetCoefficients())

    #pass the cos(theta) you want
    getValues = np.vectorize(angular_sensitivity.GetValue)
    sensitivity = np.array(costheta)
    for i, costh in enumerate(costheta):
        sensitivity[i] = getValues(float(costh))
    return sensitivity

fraction,values = find_projection(r,theta)
sensitivity = ang_sens(costheta)
maxsensitivity=max(sensitivity)
fraction *= maxsensitivity
error = np.mean(sensitivity-fraction)
print(error)
fraction+=error


fig,axes = plt.subplots(3,figsize=(8,8))
#fig.canvas.set_window_title("Projection of a half-sphere at a certain zenith")
axes[0].xlim=([-1,1])
axes[0].ylim=([min(fraction[0],min(sensitivity)),max(fraction[samples-1],max(sensitivity))])
axes[0].xlim=([-1,1])
axes[0].ylim=([min(fraction[0],min(sensitivity)),max(fraction[samples-1],max(sensitivity))])
axes[0].set_xlabel("$\\cos$$\\theta$")
axes[0].set_ylabel("Fraction of Area Visible")
axes[1].set_xlabel("$\\theta$ [radius]")
axes[1].set_ylabel("Fraction of Area Visible")
#axes[0].set_title("Projection of a half-sphere at a certain zenith")
axes[1].plot(costheta,fraction, '-r',label='geometric method')
axes[0].plot(costheta, sensitivity,'-g',label='dom default sensitivity')
axes[0].legend()
axes[1].legend()
axes[2].xlim=([-1,1])
axes[2].ylim=([min(fraction[0],min(sensitivity)),max(fraction[samples-1],max(sensitivity))])
axes[2].plot(costheta,sensitivity-fraction,'-c')
fig.savefig("projection.png")

