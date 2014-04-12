# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:07:01 2014

@author: renan
"""

# TODO:
# I don't know why, but the coupled ones, at some eta level, blast of to 
# infinity - it's unstable. I should fix this.

import numerics
import scipy
import matplotlib.pyplot as plt
from math import e, pi, cos, sin, sqrt

####################################################
# We'll start with unnormalized single laser.
####################################################
#I_th = 20.0
parameter_set_1 = True
if parameter_set_1:
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
    # I's units are mA, I think.
    omega = 1E11
#    return abs(10 * cos(omega*t))
#    return 5*(1 - e**(-omega*t/10))
    return 7.0

def dE(t, E, N):
    return 0.5 * (1 + 1j*alpha)*G_n*(N-N_th) * E

def dN(t, E, N):
    return I(t)/(el*V)  - N/t_s - abs(E)*abs(E)*(1/t_p + G_n*(N-N_th))

should_show_unnormalized_laer = False
if should_show_unnormalized_laer:
    dt = 1E-13
    max_time = dt * 10000
    initial_conditions = [100,N_th] # E, N
    solution = numerics.solve_coupled_system(dt, max_time, initial_conditions,
                                  [dE, dN])
    print "Complete"                              
    times = [t for (t, E, N) in solution]
    N = [N for (t, E, N) in solution]
    intensities = [abs(E)*abs(E) for (t, E, _) in solution]
    plt.figure()
    plt.plot(times, N)
    plt.show()
    plt.figure()
    plt.plot(times, intensities)

####################################################
# Ok, normal laser works as predicted for CW.
# Let's normalize.
####################################################

def P(s):
    P0 = 1E1
    return P0 * (I(s * t_p)/I_th - 1) 

# Y = def= sqrt(t_s * G_N / 2) * E
# Z = def= t_p * G_N / 2 * (N - N_th)
def dY(s, Y, Z):
    return (1 + 1j*alpha)* Y * Z 

def dZ(s, Y, Z):
    return (P(s) - Z - (1 + 2*Z)*abs(Y)*abs(Y)) / T

# Uncomment this to run single normalized laser

should_show_normalized_laser = False
if should_show_normalized_laser:    
    dt = 2E-13 / t_p
    max_time = dt * 10000
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0] # Y, Z
    solution = numerics.solve_coupled_system(dt, max_time, initial_conditions,
                                  [dY, dZ])
    print "Complete"                              
    times = [t for (t, Y, Z) in solution]
    Z = [Z for (t, Y, Z) in solution]
    intensities = [abs(Y)*abs(Y) for (t, Y, _) in solution]
    plt.figure()
    plt.plot(times, Z)
    plt.title("Z, homemade")
    plt.show()
    plt.figure()
    plt.plot(times, intensities)
    plt.title("Intensities, homemade")


def normalized_system(s, YZ):
    return scipy.array([dY(s,YZ[0],YZ[1]), dZ(s,YZ[0],YZ[1])])
    
should_show_numpy_ode_normalized_laser = False    
if should_show_numpy_ode_normalized_laser:        
    dt = 2E-13 / t_p
    numpoints = 10000
    max_time = dt * numpoints
    time_spacing = scipy.linspace(0, max_time, numpoints)
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0] # Y, Z
#    scipy.integrate.odeint(dFoo, initial_conditions, time_spacing, args=())                           
#    r = scipy.integrate.complex_ode(dFoo).set_integrator('zvode')
    r = scipy.integrate.complex_ode(normalized_system).set_integrator('vode', method='bdf')
    r.set_initial_value(initial_conditions, 0)
    ys = []
    zs = []
    ts = []
    while r.successful() and r.t < max_time:
        r.integrate(r.t+dt)
#        print("%g dt%g" % (r.t, r.y[0]))
        ys.append(abs(r.y[0]**2))
        zs.append(r.y[1])
        ts.append(r.t)
    plt.figure()
    plt.plot(ts, zs)
    plt.title("Z, numpy")
    plt.figure()
    plt.plot(ts, ys)
    plt.title("Intensities, numpy")

    
####################################################
# Ok, normalization works ok.
# Let's create two lasers and couple them.
####################################################
# Theta: Ratio between photon flight time from one laser to the other,
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
    
should_show_coupled_lasers = True
if should_show_coupled_lasers:
    dt = 5E-13 / t_p
    max_time = dt * 10000
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0,    # Y1, Z1
                          (1+1E-9)*1 * sqrt(t_s*G_n/2), 0,    # Y2, Z2
                          ]
    solution = numerics.solve_coupled_system(dt, max_time, initial_conditions,
                                  [dY1, dZ1, dY2, dZ2])
    print "Complete"                              
    times = [t for (t, Y1, Z1, Y2, Z2) in solution]
    Z1 = [Z1 for (t, Y1, Z1, Y2, Z2) in solution]
    intensities1 = [abs(Y1)*abs(Y1) for (t, Y1, _, _, _) in solution]
    intensities2 = [abs(Y2)*abs(Y2) for (t, _, _, Y2, _) in solution]
    plt.figure()
    plt.plot(times, Z1)
    plt.title("Z, homemade")
    plt.show()
    plt.figure()
    plt.plot(times, intensities1, '.')
    plt.plot(times, intensities2, '.')
    plt.title("Intensities, homemade")

def coupled_system(s, Y1Z1Y2Z2):
    Y1, Z1, Y2, Z2 = Y1Z1Y2Z2
    return scipy.array([dY1(s, Y1, Z2, Y2, Z2),
                        dZ1(s, Y1, Z2, Y2, Z2),
                        dY2(s, Y1, Z2, Y2, Z2),
                        dZ2(s, Y1, Z2, Y2, Z2)])

should_show_numpy_ode_coupled_laser = True    
if should_show_numpy_ode_coupled_laser:        
    dt = 2E-13 / t_p
    numpoints = 10000
    max_time = dt * numpoints
    time_spacing = scipy.linspace(0, max_time, numpoints)
    initial_conditions = [100 * sqrt(t_s*G_n/2), 0,    # Y1, Z1
                          (1+1E-9)*1 * sqrt(t_s*G_n/2), 0,    # Y2, Z2
                          ]
    r = scipy.integrate.complex_ode(coupled_system).set_integrator('vode', method='bdf')
    r.set_initial_value(initial_conditions, 0)
    y1s = []
    y2s = []
    z1s = []
    ts = []
    while r.successful() and r.t < max_time:
        r.integrate(r.t+dt)
#        print("%g dt%g" % (r.t, r.y[0]))
        y1s.append(abs(r.y[0]**2))
        y2s.append(abs(r.y[2]**2))
        z1s.append(r.y[1])
        ts.append(r.t)                        
    plt.figure()
    plt.plot(ts, z1s)
    plt.title("Z, numpy")    
    plt.figure()
    plt.plot(ts, y1s, '.')
    plt.plot(ts, y2s, '.')
    plt.title("Intensities, numpy")
    
## Theta: Ratio between photon flight time from one laser to the other,
## tau_f, and the photon lifetime inside the laser, tau_p.
#tau_p = 1.11E-12
#tau_f = 22.2E-12
#theta = 20
#
## omega: Produdct of the angular frequency of a solitary laser and the photon 
## lifetime.
#f_r = 5.27E9 # Hrtz
#angular_f_r = f_r * 2 * pi
#omega = angular_f_r * tau_p
#
## Linewidth enhancement factor. (???)
##XXX: Find out what this is.
#alpha = 4
#
## T: ratio between carrier lifetime and the photon lifetime.
#T = 1710
#
## P: Dimensionless pumping current above solitary laser threshold.
#P = 1.155
#
## Eta: Coupling parameter between the two lasers.
#eta = 5E-4
#eta = 0 
#
#
#def dE1(t, E1, E2, N1, N2):
#    s = t / tau_p
#    return (1+alpha*1j)*N1*E1 + eta*E2*(s - theta)*(e**(-1j*theta*omega))
#    
#def dE2(t, E1, E2, N1, N2):
#    s = t / tau_p
#    return (1+alpha*1j)*N2*E2 + eta*E1*(s - theta)*(e**(-1j*theta*omega))    
#    
#def dN1(t, E1, E2, N1, N2):
#    return (P - N1 - (1+2*N1)*abs(E1)*abs(E1)) / T
#    
#def dN2(t, E1, E2, N1, N2):
#    return (P - N2 - (1+2*N2)*abs(E2)*abs(E2)) / T
#    
#dt = 1E-12
#max_time = 10000 * dt
#initial_conditions = [1.,1.,1.,1.] # E1, E2, N1, N2
#solution = numerics.solve_coupled_system(dt, max_time, initial_conditions,
#                              [dE1, dE2, dN1, dN2])
#print "Complete"                              
#
#times = [t for (t, E1, E2, N1, N2) in solution]
#N = [N1 for (t, E1, E2, N1, N2) in solution]
#intensities = [abs(E1)*abs(E1) for (t, E1, E2, N1, N2) in solution]
#
#plt.plot(N)
#plt.show()