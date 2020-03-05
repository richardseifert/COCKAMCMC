from pyfam.mcmc import MCMC

import numpy as np
import matplotlib.pyplot as plt

#Define a model function to use when fitting data.
#  For this example, that's a gaussian function plus a linear offset.
def f(x, p):
    a, b, c, d, e = p
    return a*np.exp(-(x-b)**2/(2*c**2))  +  d*x + e

#Make some fake data with added noise.
npoints = 30
a, b, c, d, e = 5., 1., 2., 0.2, 2.
p = [a, b, c, d, e]
x = np.linspace(-10, 10, npoints)
y = f(x, p)+np.random.normal(0, 0.5, npoints)
x += np.random.normal(0, .05, npoints)

#Create MCMC instance.
mcmc = MCMC(x, y, model=f)

#Add one walker.
mcmc.add_walkers(1, p0=[1., 1., 1., 1., 1.]) #Starting guess is [1.0, 1.0, 1.0, 1.0, 1.0]

#Burn in for 200 steps.
print("Burn in")
mcmc.walk(200, run_id="burn")

#Add four additional walkers.
mcmc.add_walkers(4)

#Walk each walker for 1000 steps.
print("Walking")
mcmc.walk(5000, run_id="walk")

#Plot 50 randomly drawn accepted fits.
mcmc.plot_sample(run_id="walk", n=50)

#Plot a corner plot, showing the posterior distribution of parameters.
mcmc.corner(run_id="walk", bins=35, threshold=2, fill=True, p_crosshair="best")

plt.show()
print("WAAAAAA")