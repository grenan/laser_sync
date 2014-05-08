# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:07:01 2014

@author: renan
"""

# TODO:
# I don't know why, but the coupled ones, at some eta level, blast off to 
# infinity - it's unstable. I should fix this. Perhaps use an adaptive timestep somehow?

import scipy
import scipy.integrate
import matplotlib.pyplot as plt
from math import e, pi, cos, sin, sqrt

def integrate_complex_system(t0, t1, dt, f, initial_conditions):
    """
    Solves the complex ordinary differential equation in the form
    y'(x) = f(x,y)
    where both x and y are vectors.
    This is basically just a wrapper for the scipy complex ode,
    since it requires a bit of code duplication and also doesn't
    return a list but rather values one after the other.

    f is a function in the form f(t, y), where y is a vector. 
    initial_conditions is a vector y values at t = t0.

    Returns a list of times and list of values (ts, ys).
    """
    ts = []
    ys = []
    r = scipy.integrate.complex_ode(f).set_integrator('vode', method='bdf')
    r.set_initial_value(initial_conditions, t0)
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        ys.append(r.y)
        ts.append(r.t)
    if not r.successful():
        raise ValueError("Integration not successful for some reason.")
    return (ts, ys)    

####################################################
# We'll start with unnormalized single laser.
####################################################
#I_th = 20.0
I_th = 5
N_th = 1.5E18
t_p = 4.5E-12
t_s = 700E-12
alpha = 5.0
G_n = 2.6E-6
T = t_s/t_p
P_0 = (t_p * G_n * N_th) / 2
el = 1.6E-19
V = 0.00000001
    
def I(t):
    # Its units are mA, I think.
    omega = 1E11
#    return abs(10 * cos(omega*t))
#    return 5*(1 - e**(-omega*t/10))
    return 7.0

def dE(t, E, N):
    return 0.5 * (1 + 1j*alpha)*G_n*(N-N_th) * E

def dN(t, E, N):
    return I(t)/(el*V)  - N/t_s - abs(E)*abs(E)*(1/t_p + G_n*(N-N_th))

def unnormalized_system(s, EN):
    return scipy.array([dE(s, EN[0], EN[1]), dN(s, EN[0], EN[1])])    

should_show_unnormalized_laer = False
if should_show_unnormalized_laer:
    dt = 1E-13
    max_time = dt * 10000
    initial_conditions = [100, N_th] # E, N
    times, values = integrate_complex_system(0, max_time, dt, unnormalized_system, initial_conditions)
    Es, Ns = zip(*values)
    intensities = [abs(E)**2 for E in Es]    
    
    plt.figure()
    plt.plot(times, Ns)
    plt.title("N")
    plt.figure()
    plt.plot(times, intensities)
    plt.title("Intensities")
    plt.show()

####################################################
# Ok, normal laser works as predicted for CW.
# Let's normalize.
####################################################

def P(s):
    P0 = 1E1
    P0 = 2.8875
    return P0 * (I(s * t_p)/I_th - 1) 

# Y = def= sqrt(t_s * G_N / 2) * E
# Z = def= t_p * G_N / 2 * (N - N_th)
def dY(s, Y, Z):
    return (1 + 1j*alpha)* Y * Z 

def dZ(s, Y, Z):
    return (P(s) - Z - (1 + 2*Z)*abs(Y)*abs(Y)) / T

def normalized_system(s, YZ):
    return scipy.array([dY(s, YZ[0], YZ[1]), dZ(s, YZ[0], YZ[1])])

should_show_normalized_laser = False
if should_show_normalized_laser:    
    dt = 2E-13 / t_p
    max_time = dt * 10000
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0] # Y, Z
    times, values = integrate_complex_system(0, max_time, dt, normalized_system, initial_conditions)
    Ys, Zs = zip(*values)
    intensities = [abs(Y)**2 for Y in Ys]    
    plt.figure()
    plt.plot(times, Zs)
    plt.title("Z")
    plt.figure()
    plt.plot(times, intensities)
    plt.title("Intensities")
    plt.show()
    
####################################################
# Ok, normalization works ok.
# Let's create two lasers and couple them.
####################################################
# Theta: Ratio between photon flight time from one laser to the other and
# the photon lifetime within the laser.
theta = 20

# omega: Produdct of the angular frequency of a solitary laser and the photon 
# lifetime.
f_r = 5.27E9 # Hrtz
angular_f_r = f_r * 2 * pi
omega = angular_f_r * t_p

# Coupling parameter:
eta = 4.32E-2

def dY1(s, Y1, Z1, Y2, Z2):
    return (1 + 1j*alpha)* Y1 * Z1 + eta*Z2*(s-theta)*(e**(-1j*omega*theta))

def dZ1(s, Y1, Z1, Y2, Z2):
    return (P(s) - Z1 - (1 + 2*Z1)*abs(Y1)*abs(Y1)) / T
    
def dY2(s, Y1, Z1, Y2, Z2):
    return (1 + 1j*alpha)* Y2 * Z2 + eta*Z1*(s-theta)*(e**(-1j*omega*theta))

def dZ2(s, Y1, Z1, Y2, Z2):
    return (P(s) - Z2 - (1 + 2*Z2)*abs(Y2)*abs(Y2)) / T

def coupled_system(s, Y1Z1Y2Z2):
    Y1, Z1, Y2, Z2 = Y1Z1Y2Z2
    return scipy.array([dY1(s, Y1, Z1, Y2, Z2), dZ1(s, Y1, Z1, Y2, Z2),
                        dY2(s, Y1, Z1, Y2, Z2), dZ2(s, Y1, Z1, Y2, Z2)])
    
should_show_coupled_lasers = True
if should_show_coupled_lasers:
    dt = 5E-13 / t_p
    max_time = dt * 10000
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0,    # Y1, Z1
                          (1+1E-9)*100 * sqrt(t_s*G_n/2), 0,    # Y2, Z2
                         ]
    times, values = integrate_complex_system(0, max_time, dt, coupled_system, initial_conditions)
    Y1s, Z1s, Y2s, Z2s = zip(*values)
    intensities1 = [abs(Y)**2 for Y in Y1s]
    intensities2 = [abs(Y)**2 for Y in Y2s]
    
    plt.figure()
    plt.plot(times, Z1s, '.')
    plt.plot(times, Z2s, 'x')
    plt.title("Z")    
    plt.figure()
    plt.plot(times, intensities1, '.')
    plt.plot(times, intensities2, 'x')
    plt.title("Intensities")
    plt.show()