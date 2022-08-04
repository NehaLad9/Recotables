from icecube.icetray import I3Units
from icecube.dataclasses import I3Constants
from icecube.clsim import GetIceCubeDOMAngularSensitivity
from icecube.simclasses import I3CLSimFunctionPolynomial
from icecube.icetray import I3Units, I3Module, traysegment
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
parser.add_option("--direction", choices = ('down','up'), dest="direction", default='down', help="direction of the IceCube DOM -- only legible options are 'down' and 'up' ")


opts, _  = parser.parse_args()

print("Options Loaded")

print("Options: {}".format(opts))


dist = opts.dist
r = opts.r
samples = opts.samples
theta = np.linspace(0,np.pi,samples)
costheta=np.cos(theta)
direction = opts.direction


DOMRadius = 16.510*I3Units.cm # 13" diameter
referenceArea = I3Constants.pi*DOMRadius**2

####find the effective area

def effective(cth,r,direction='down'):
    if direction == 'down':
        mult=-1
    else:
        mult=1
    return np.pi*r**2*(1+cth*mult)/2



def find_projection(r,theta,direction):
    total_area = np.pi*r**2
    values=np.array(theta)
    fraction=np.array(theta)
    for i, th in enumerate(theta):
        values[i] = effective(np.cos(th),r,direction)
        #print(values[i])
        fraction[i] = values[i]/total_area
        #print(f"fraction is: {fraction[i]}" )
    return fraction,values

def ang_sens(costheta):
    #angular_sensitivity is a function
    angular_sensitivity = GetIceCubeDOMAngularSensitivity()
    #sensitivity = angular_sensitivity.GetValue(1)
    #print(sensitivity,angular_sensitivity.GetCoefficients())

    #pass the cos(theta) you want
    getValues = np.vectorize(angular_sensitivity.GetValue)
    sensitivity = np.array(costheta)
    #saving values in reverse order
    for i, costh in enumerate(costheta):
        sensitivity[len(costheta)-1-i] = getValues(float(costh))
    return sensitivity

fraction,values = find_projection(r,theta,direction)
sensitivity = ang_sens(costheta)
maxsensitivity=max(sensitivity)
fraction *= maxsensitivity


def plot_methods(fraction,sensitivity,costheta):
    fig,axes = plt.subplots(3,figsize=(10,10))
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

plot_methods(fraction,sensitivity,costheta)


def GetIceCubeDOMAngularSensitivityLinear(direction):
    if(direction == 'down'):
        mult=-1
    else:
        mult=+1
    coefficients = np.array([1./2,1./2*mult])
    return I3CLSimFunctionPolynomial(coefficients)

GetIceCubeDOMAngularSensitivityLinear(direction)


print(GetIceCubeDOMAngularSensitivityLinear(direction))
from icecube.clsim.Gen2Sensors import GetDEggAcceptance,GetDEggAngularSensitivity


referenceArea = I3Constants.pi*(300.*I3Units.mm/2)**2
angularAcceptance = I3CLSimFunctionPolynomial(np.array([1.]))
angularAcceptance = GetDEggAngularSensitivity(pmt='both')
getValues = np.vectorize(angularAcceptance.GetValue)
sensitivity = np.array(costheta)
#saving values in reverse order
for i, costh in enumerate(costheta):
    sensitivity[len(costheta)-1-i] = getValues(float(costh))
print(GetDEggAcceptance(active_fraction=1.).GetValue(400*I3Units.nanometer))
domAcceptance = GetDEggAcceptance(active_fraction=1.).GetValue(400*I3Units.nanometer)*referenceArea
#domAcceptance = GetWOMAcceptance().GetValue(4e-7)*referenceArea
print(domAcceptance*sensitivity[0])
fig2 = plt.figure()
plt.plot(costheta,domAcceptance*sensitivity)
fig2.savefig("wavelength400")