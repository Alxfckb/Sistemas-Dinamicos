#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:13:47 2021

@author: juan
"""

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

N=20000000

def f(y, t, paras):
    """
    Your system of differential equations
    """

    x1 = y[0]
    x2 = y[1]
    x3 = y[2]

    try:
        k0 = paras['k0'].value
        k1 = paras['k1'].value

    except KeyError:
        k0, k1 = paras
    # the model equations
    f0 = -k0 * x1 * x2/N
    f1 = k0 * x1 *x2/N - k1 * x2
    f2 = k1 * x2
    return [f0, f1, f2]


def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(f, x0, t, args=(paras,))
    return x


def residual(paras, t, data):

    """
    compute the residual between actual data and fitted data
    """

    x0 = paras['x10'].value, paras['x20'].value, paras['x30'].value
    model = g(t, x0, paras)

    # you only have data for one of your variables
    x2_model = model[:, 1]
    return (x2_model - data).ravel()


# initial conditions
x10 = N-5
x20 = 5
x30 = 0
y0 = [x10, x20, x30]
_nSteps = 300
# measured data
t_measured = np.linspace(0,_nSteps-1,_nSteps)

x2_measured = np.array(genfromtxt('20millones.txt',delimiter=','))

#np.array([0.000, 0.416, 0.489, 0.595, 0.506, 0.493, 0.458, 0.394, 0.335, 0.309])

plt.figure()
plt.scatter(t_measured, x2_measured, marker='+', color='b', label='Datos de sim. estocástica', s=75)

# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()
params.add('x10', value=x10, vary=False)
params.add('x20', value=x20, vary=False)
params.add('x30', value=x30, vary=False)
params.add('k0', value=0.02, min=0.0000001, max=1.)
params.add('k1', value=0.0714, min=0.0000001, max=1.)

# fit model
result = minimize(residual, params, args=(t_measured, x2_measured), method='leastsq')  # leastsq nelder

# check results of the fit
data_fitted = g(t_measured, y0, result.params)

# plot fitted data
plt.plot(t_measured, data_fitted[:, 1], '-', linewidth=2, color='red', label='Ajuste de datos')
plt.ylabel('Infectados')
plt.xlabel('Tiempo (en días)')
plt.legend()
plt.xlim([0, max(t_measured)])
plt.ylim([0, 1.1 * max(data_fitted[:, 1])])
plt.savefig('20M.pdf')
# display fitted statistics
report_fit(result)

plt.show()
