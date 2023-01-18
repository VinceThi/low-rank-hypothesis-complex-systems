# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


from scipy.integrate import ode
import numpy as np
import time as timer

""" We recommand solve_ivp with BDF in general. """

def integrate_rk4(t0, t1, dt, dynamics, adjacency_matrix,
                  init_cond, *args):
    args = (adjacency_matrix, *args)
    f = dynamics
    tvec = np.arange(t0, t1, dt)
    sol = [init_cond]
    for i, t in enumerate(tvec[0:-1]):
        k1 = f(t, sol[i], *args)
        k2 = f(t+dt/2, sol[i] + k1/2, *args)
        k3 = f(t+dt/2, sol[i] + k2/2, *args)
        k4 = f(t+dt, sol[i] + k3, *args)
        sol.append(sol[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
    return sol


def integrate_dopri45(t0, t1, dt, dynamics, adjacency_matrix,
                      init_cond, *args):
    args = (adjacency_matrix, *args)
    f = dynamics
    tvec = np.arange(t0, t1, dt)
    sol = [init_cond]
    for i, t in enumerate(tvec[0:-1]):
        k1 = f(t, sol[i], *args)
        k2 = f(t + 1./5*dt, sol[i] + dt*(1./5*k1), *args)
        k3 = f(t + 3./10*dt, sol[i] + dt*(3./40*k1 + 9./40*k2), *args)
        k4 = f(t + 4./5*dt, sol[i] + dt*(44./45*k1 - 56./15*k2 + 32./9*k3),
               *args)
        k5 = f(t + 8./9*dt, sol[i] + dt*(19372./6561*k1 - 25360./2187*k2
                                         + 64448./6561*k3 - 212./729*k4),
               *args)
        k6 = f(t + dt, sol[i] + dt*(9017./3168*k1 - 355./33*k2 + 46732./5247*k3
                                    + 49./176*k4 - 5103./18656*k5), *args)
        v5 = 35./384*k1 + 500./1113*k3 \
            + 125./192*k4 - 2187./6784*k5 + 11./84*k6
        # k7 = f(t + dt, sol[i] + dt*v5, *args)
        # v4 = 5179./57600*k1 + 7571./16695*k3 + 393./640*k4 \
        #     - 92097./339200*k5 + 187./2100*k6 + 1./40*k7

        sol.append(sol[i] + dt*v5)

    return sol


def integrate_dynamics(t0, t1, dt, dynamics, adjacency_matrix,
                       integrator, init_cond, *args, print_process_time=False):
    """

    :param t0:
    :param t1:
    :param dt:
    :param dynamics:
    :param adjacency_matrix:
    :param integrator:
    :param init_cond:
    :param args:
    :param print_process_time:
    :return:
    """
    # print(args, len(args))
    r = ode(dynamics).set_integrator(integrator, max_step=dt)
    r.set_initial_value(init_cond, t0).set_f_params(adjacency_matrix, *args)
    t = [t0]
    sol = [init_cond]
    time_0 = timer.clock()
    i = 0
    for r.t in range(int(t1/dt)-1):
        if r.successful():
            # print(r.t+dt, r.integrate(r.t+dt))
            t.append(r.t + dt)
            sol.append(r.integrate(r.t + dt))
            i += 1
            print(i)
        # else:
        #     print("Integration was not successful for this step.")

    if print_process_time:
        print("Integration done. Time to process:",
              np.round((timer.clock()-time_0)/60, 5),
              "minutes", "(", np.round(timer.clock()-time_0, 5), " seconds)")

    return np.array(sol)

