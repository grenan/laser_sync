# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:07:01 2014

@author: renan
"""

import scipy
import scipy.integrate
import matplotlib.pyplot as plt
from math import e, pi, cos, sin, sqrt

flatten = lambda l : [item for sublist in l for item in sublist]

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
eta = 5.0E-4

# Note: coupling requires access to the laser state at a previous (far away)
# time. This makes it a bit harder to integrate, since you have to save
# those values and use them online. Not to worry!

class CoupledRingLasers(object):
    """
    Used solving a ring of coupled lasers, where the i-th laser feeds its beam
    into the i+1-th.

    Note for the programmer: counting starts from 0! So if there are three
    lasers, they will be labeled 0,1,2.
    """
    def __init__(self, n):
        """
        n - the number of lasers feeding to eacher other. Assumes n>=2.
        """
        self.n = n
        # History: a list in the format (time, [values]), (time, values)...
        self.history = []
        self.dYs = [self._generate_dYk(k) for k in xrange(n)]
        self.dZs = [self._generate_dZk(k) for k in xrange(n)]

    def get_history_at_time(self, t):
        default_to_return = [0] * (2 * self.n)
        if self.history == []:
            return default_to_return
        # TODO: make it so that this is adjustable, if need be. 
        #!!! Assumes that the earliest time possible is t = 0.
        # Any requests to get history earlier than that will return nothing.
        if t <= 0:
            return default_to_return

        i = 1
        try:        
            while self.history[-i][0] > t:
                i+=1
            return self.history[-i][1]
        except IndexError: # Not found yet, just pesky negative indexing.
            return default_to_return
            
    def update_history(self, t, values):
        self.history.append((t, values))

    def _generate_dYk(self, k):
        """
        Generates the function which calculates dYk/ds.
        This depends both on the history (because laser k-1 shoots into laser
        k), and on the laser k itself. The function will be of the format:
        f(s,*args), where *args is a list y0,z0,y1,z1,... of current values
        of ALL the lasers.
        """
        previous_k = (k-1) % self.n
        def dYk(s, *args):
            old_y_prev_k = self.get_history_at_time(s-theta)[2*previous_k]
            Yk = args[2*k]
            Zk = args[2*k+1]
            return (1+1j*alpha) * Yk * Zk + eta * old_y_prev_k*(e**(-1j*omega*theta))
        return dYk

    def _generate_dZk(self, k):
        """
        See description for _generate_dYk; this is the same, but for the Z
        variable.
        """
        def dZk(s, *args):
            Yk = args[2*k]
            Zk = args[2*k+1]
            return (P(s) - Zk - (1 + 2*Zk)*abs(Yk)*abs(Yk)) / T
        return dZk

    def _coupled_system(self, s, values):
        # A list in the format: [dY0, dZ0, dY1, dZ1, ...]
        all_interlaced_functions = flatten(zip(self.dYs, self.dZs))
        # Compute all the current contributions to each value, and return.
        return scipy.array([f(s, *values) for f in all_interlaced_functions])

    def __call__(self, s, values):
        """
        Values is a list of the current values of the system:
        [y0, z0, y1, z1, y2, z2, ... yn-1, zn-1]
        """
        return self._coupled_system(s, values)
        
should_show_coupled_lasers = True
if should_show_coupled_lasers:
    # Note: this must be smaller than the "theta" parameter if you want
    # any effect of actual feedback (well, otherwise you are practically
    # just playing with a different theta than you think you are)
    dt = 1.0E-12 / t_p
    max_time = dt * 20000
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0,    # Y1, Z1
                          (1+1E-5) * 1 * sqrt(t_s*G_n/2), 0,    # Y2, Z2
                          #7 * sqrt(t_s*G_n/2), 0,
                         ]
    coupled = CoupledRingLasers(2)
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