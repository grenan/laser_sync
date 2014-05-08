# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:07:01 2014

@author: renan
"""

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

def integrate_complex_system_with_history(t0, t1, dt, f, initial_conditions):
    """
    See documentation for integrate_complex_system.
    Here instead of a function which calculates all the differentials,
    we pass a class f. It is callable, calling it replicates the effect
    of the function f in integrate_complex_system.
    However, the functions in f also have access to its history,
    which is updated here. This allows for more complicated connections
    between parameters, as they can access their past (ala difference equation)
    """
    ts = []
    ys = []
    r = scipy.integrate.complex_ode(f).set_integrator('vode', method='bdf')
    r.set_initial_value(initial_conditions, t0)
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        ys.append(r.y)
        ts.append(r.t)
        f.update_history(r.t, r.y)

    if not r.successful():
        raise ValueError("Integration not successful for some reason.")
    return (ts, ys)

####################################################
# We'll start with unnormalized single laser.
####################################################
I_th = 5
N_th = 1.5E18
t_p = 1.11E-12
t_s = 1925E-12
alpha = 4.0
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
    dt = 2E-12 / t_p
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
eta = 1.5E-1

# Note: coupling requires access to the laser state at a previous (far away)
# time. This makes it a bit harder to integrate, since you have to save
# those values and use them online. Not to worry!

class CoupledLasers(object):
    #TODO: generalize to n lasers so that it can build it automatically.
    # Also allow calibration with parameters etc. Slowly build it up, though,
    # no need to rush into too general things before it's time to do so.
    def __init__(self):
        # History: a list in the format (time, [values]), (time, values)...
        self.history = []

    def get_history_at_time(self, t):
        if self.history == []:
            return [0,0,0,0]
        # TODO: make it so that this is adjustable, if need be. 
        #!!! Assumes that the earliest time possible is t = 0.
        # Any requests to get history earlier than that will return nothing.
        if t <= 0:
            return [0,0,0,0]

        i = 1
        try:        
            while self.history[-i][0] > t:
                i+=1
            return self.history[-i][1]
        except IndexError: # Not found yet, just pesky negative indexing.
            return [0,0,0,0]

    def update_history(self, t, values):
        self.history.append((t, values))

    def dY1(self, s, Y1, Z1, Y2, Z2):
        old_y2 = self.get_history_at_time(s-theta)[2]
        return (1 + 1j*alpha)* Y1 * Z1 + eta*old_z2*(e**(-1j*omega*theta))

    def dZ1(self, s, Y1, Z1, Y2, Z2):
        return (P(s) - Z1 - (1 + 2*Z1)*abs(Y1)*abs(Y1)) / T
    
    def dY2(self, s, Y1, Z1, Y2, Z2):
        old_y1 = self.get_history_at_time(s-theta)[0]
        return (1 + 1j*alpha)* Y2 * Z2 + eta*old_z1*(e**(-1j*omega*theta))

    def dZ2(self, s, Y1, Z1, Y2, Z2):
        return (P(s) - Z2 - (1 + 2*Z2)*abs(Y2)*abs(Y2)) / T

    def coupled_system(self, s, Y1Z1Y2Z2):
        Y1, Z1, Y2, Z2 = Y1Z1Y2Z2
        return scipy.array([self.dY1(s, Y1, Z1, Y2, Z2),
                            self.dZ1(s, Y1, Z1, Y2, Z2),
                            self.dY2(s, Y1, Z1, Y2, Z2),
                            self.dZ2(s, Y1, Z1, Y2, Z2)])

    def __call__(self, s, Y1Z1Y2Z2):
        return self.coupled_system(s, Y1Z1Y2Z2)


should_show_coupled_lasers = True
if should_show_coupled_lasers:
    # Note: this must be smaller than the "theta" parameter if you want
    # any effect of actual feedback (well, otherwise you are practically
    # just playing with a different theta than you think you are)
    dt = 1.0E-12 / t_p
    max_time = dt * 5000
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0,    # Y1, Z1
                          (1+1E-5) * 1 * sqrt(t_s*G_n/2), 0,    # Y2, Z2
                         ]
    coupled  = CoupledLasers()
    times, values = integrate_complex_system_with_history(0, max_time, dt, coupled, initial_conditions)
    Y1s, Z1s, Y2s, Z2s = zip(*values)
    intensities1 = [abs(Y)**2 for Y in Y1s]
    intensities2 = [abs(Y)**2 for Y in Y2s]
    real1 = [Y.real for Y in Y1s]
    real2 = [Y.real for Y in Y2s]
    
    plt.figure()
    plt.plot(times, Z1s)
    plt.plot(times, Z2s)
    plt.title("Z")    
    plt.figure()
    plt.plot(times, intensities1)
    plt.plot(times, intensities2)
    # plt.plot(times, real1)
    # plt.plot(times, real2)
    plt.title("Intensities")
    plt.show()