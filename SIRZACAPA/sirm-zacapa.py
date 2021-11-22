

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
#Parametros tomados de Okosun&Makinde 2011

um = 0.1429*7
vm = 1000*7
N  = 72502

def f(y, t, paras):
    """
    Your system of differential equations
    """
    x1 = y[0]                   #Sh
    x2 = y[1]                   #Ih
    x3 = y[2]                   #Rh

    z1 = y[3]                   #Sm
    z2 = y[4]                   #Im

    try:

        b = paras['b'].value
        th = paras['th'].value
        tm = paras['tm'].value
        gm = paras['gm'].value
    except KeyError:
        th, tm, b ,gm = paras

    f1 = - th*b*z2*x1/N
    f2 = th*b*z2*x1/N - gm*x2
    f3 = gm*x2
    f4 = vm - tm*b*x2*z1/N - um*z1
    f5 = tm*b*x2*z1/N - um*z2

    return [f1, f2, f3, f4, f5]


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

    x0 = paras['x10'].value, paras['x20'].value, paras['x30'].value, paras['z10'].value, paras['z20'].value
    model = g(t, x0, paras)

    # you only have data for one of your variables
    x2_model = model[:, 1]
    return (x2_model - data).ravel()


# initial conditions
x10 = N-1
x20 = 1/1426
x30 = 0
z10 = 1.5E9
z20 = 1
y0 = [x10, x20, x30, z10, z20]

# measured data

x2_measured=np.array([1,83,200,357,625,761,1284,483,1426,603,521,63,241,144,74,6,67,0,1,14,
12,13,4,1,0,0,0,0,0,0,4,4,6,13,10,0,1,1,0,1,
4,2,4,3,6,6,7,11,5,8,2,4,2,12,2,5,3,7,8,14,
4,7,2,4,2,1])/1426

t_measured = np.linspace(1, 66,len(x2_measured))
plt.figure()
plt.scatter(t_measured, x2_measured, marker='+', color='b', label='Casos observados', s=75)

# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()
params.add('x10', value=x10, vary=False)
params.add('x20', value=x20, vary=False)
params.add('x30', value=x30, vary=False)
params.add('z10', value=z10, vary=False)
params.add('z20', value=z20, vary=False)

params.add('th', value=0.3, min=1E-3, max=1)
params.add('tm', value=0.015, min=1E-3, max=1)
params.add('b', value=0.025, min=1E-2, max=1)
params.add('gm', value=0.03, min=1E-2, max=1)


# fit model
result = minimize(residual, params, args=(t_measured, x2_measured), method='leastsq')  # leastsq nelder
# check results of the fit
data_fitted = g(t_measured, y0, result.params)

# plot fitted data
plt.plot(t_measured, data_fitted[:, 1], '-', linewidth=2, color='red', label='Ajuste de datos')
plt.legend()
plt.xlabel('Tiempo (en semanas)')
plt.ylabel('Incidencia de casos de chikungunya')
plt.savefig('zacapa_m.pdf')
plt.xlim([0, max(t_measured)])

# display fitted statistics
report_fit(result)

plt.show()
