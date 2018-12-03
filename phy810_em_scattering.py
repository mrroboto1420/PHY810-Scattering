# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:06:19 2018

@author: bstev
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as mpl

from scipy.constants import pi
from scipy.special import jv as Jv
from scipy.special import hankel2 as Hv2
from scipy.special import jvp as Jv_prime
from scipy.special import h2vp as Hv2_prime


def close():
    return mpl.close('all')


close()

ep_r = 5

sum_size = 50
delta = 200

ratio_x_a_range = np.arange(-2, 2 + 4/delta, 4/delta)
ratio_y_a_range = ratio_x_a_range

rho_range = np.ndarray(shape=(len(ratio_x_a_range), len(ratio_x_a_range)))
phi_range = np.ndarray(shape=(len(ratio_x_a_range), len(ratio_x_a_range)))

for i in range(len(ratio_x_a_range)):
    for j in range(len(ratio_x_a_range)):
        rho_range[i, j] = np.sqrt(ratio_x_a_range[i]**2 +
                                  ratio_y_a_range[j]**2)

for i in range(len(ratio_x_a_range)):
    for j in range(len(ratio_x_a_range)):
        phi_range[i, j] = sp.arctan2(ratio_y_a_range[j], ratio_x_a_range[i])


def E_totField_dielec_in(rho, phi):
    alp0_Ztm = np.sqrt(ep_r)
    k_0a = 4*pi
    ka = 4*pi*np.sqrt(ep_r)
    temp = np.ndarray(shape=(len(ratio_x_a_range), len(ratio_x_a_range)),
                      dtype='complex128')
    for i in range(1, len(ratio_x_a_range)):
        for j in range(1, len(ratio_x_a_range)):
            if rho[i, j] <= 1:
                for n in range(0, sum_size):
                    if sum_size == 0:
                        neumann = 1
                        temp[i, j] += \
                            1j**(-n)*neumann*np.divide(
                                    Hv2(n, k_0a)*Jv_prime(n, k_0a) -
                                    Jv(n, k_0a)*Hv2_prime(n, k_0a),
                                    alp0_Ztm*Jv_prime(n, ka)*Hv2(n, k_0a) -
                                    Hv2_prime(n, k_0a)*Jv(n, ka))*Jv(n, ka*rho[i, j])*np.cos(n*phi[i, j])
                    else:
                        neumann = 2
                        temp[i, j] += \
                            1j**(-n)*neumann*np.divide(
                                    Hv2(n, k_0a)*Jv_prime(n, k_0a) -
                                    Jv(n, k_0a)*Hv2_prime(n, k_0a),
                                    alp0_Ztm*Jv_prime(n, ka)*Hv2(n, k_0a) -
                                    Hv2_prime(n, k_0a)*Jv(n, ka))*Jv(n, ka*rho[i, j])*np.cos(n*phi[i, j])     
            else:
                 temp[i, j] = 0             
    return temp


def E_totField_dielec_out(rho, phi):
    alp0_Ztm = np.sqrt(ep_r)
    k_0a = 4*pi
    ka = 4*pi*np.sqrt(ep_r)
    temp = np.ndarray(shape=(len(ratio_x_a_range), len(ratio_x_a_range)),
                      dtype='complex128')
    for i in range(1, len(ratio_x_a_range)):
        for j in range(1, len(ratio_x_a_range)):
            if rho[i, j] <= 1:
                 temp[i, j] = 0
            else:
                for n in range(0, sum_size):
                    if sum_size == 0:
                        neumann = 1
                        temp[i, j] += \
                            1j**(-n)*neumann*np.divide(
                                    alp0_Ztm*Jv_prime(n, ka)*Jv(n, k_0a) -
                                    Jv_prime(n, k_0a)*Jv(n, ka),
                                    alp0_Ztm*Jv_prime(n, ka)*Hv2(n, k_0a) -
                                    Hv2_prime(n, k_0a)*Jv(n, ka))*Hv2(n, k_0a*rho[i, j])*np.cos(n*phi[i, j])
                    else:
                        neumann = 2
                        temp[i, j] += \
                            1j**(-n)*neumann*np.divide(
                                    alp0_Ztm*Jv_prime(n, ka)*Jv(n, k_0a) -
                                    Jv_prime(n, k_0a)*Jv(n, ka),
                                    alp0_Ztm*Jv_prime(n, ka)*Hv2(n, k_0a) -
                                    Hv2_prime(n, k_0a)*Jv(n, ka))*Hv2(n, k_0a*rho[i, j])*np.cos(n*phi[i, j])

    return np.exp(-1j*4*pi*rho*np.cos(phi)) - temp


def circle(x, y):
    return (x*x+y*y)


X, Y = np.meshgrid(ratio_x_a_range, ratio_y_a_range)
Z_in = E_totField_dielec_in(rho_range, phi_range)
Z_out = E_totField_dielec_out(rho_range, phi_range)

Z_cir = circle(X, Y)

Z_final = np.zeros([len(X), len(X)], dtype='complex128')

for i in range(0, len(X)):
    for j in range(0, len(X)):
        if rho_range[i, j] <= 1:
            Z_final[i, j] = Z_in[i, j]
        else:
            Z_final[i, j] = Z_out[i, j]

fig = mpl.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect(1)
cs = mpl.contourf(Y, X, np.abs(Z_final), 100)
mpl.contour(Y, X, Z_cir, [1])
cb = mpl.colorbar(cs, orientation='vertical')
mpl.clim()
mpl.xlim(-2, 2)
mpl.ylim(-2, 2)
mpl.title(r'Dielectric Cylinder, $\epsilon_r = %d$' % ep_r)
mpl.xlabel('x/a')
mpl.ylabel('y/a')
