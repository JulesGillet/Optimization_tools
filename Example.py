from optim_wrapper import optim_wrapper
from constraints_wrapper import *
import numpy as np

def obj(X):
    """
    Objective function: Rosenbrock function with n variables
    Minimum for obj(1.0,...,1.0) = 0.0
    """
    res = 0.0
    for i_param in range(len(X)-1):
        res += 100.0*(X[i_param+1]-X[i_param]**2)**2 + (X[i_param]-1.0)**2
    return res

def pen(X):
    """
    Penalties to constrain the Rosenbrock function with a cubic and a line
    """
    pen0 = Ineg_wrapper(0.0,(X[0]-1.0)**3 - X[1] + 1.0)
    pen1 = Ineg_wrapper(0.0,X[0] + X[1] - 2.0)
    return [pen0, pen1]

def Example_1(nb_param):
    """
    Testing the optimization wrapper with Rosenbrock function with n variables
    Prints the results of the optimization
    """
    print "\n** Example_1: Finding the minimum of the Rosenbrock function with {0} variables **".format(nb_param)

    Ex = optim_wrapper()
    X0 = np.zeros(nb_param)
    lim = [(-2.0,2.0)]*nb_param
    Ex.set_X0(X0)
    Ex.set_lim(lim)
    Ex.set_norm_count(50*nb_param*2)
    Ex.set_nb_best(50*nb_param)
    Ex.set_obj_func(obj)
    Ex.set_wrapper()
    Ex.launch_multi_opti()
    print Ex

    X_solution = [1.0]*nb_param
    res_string = "Results of the optimisation: {:03.4f}, expected results: {:03.4f}".format(obj(Ex.get_res()),obj(X_solution))
    print res_string
    print "*"*len(res_string)

def Example_2():
    """
    Testing the optimization wrapper with Rosenbrock function with 2 variables and penalties
    Prints the results of the optimization
    """
    print "\n** Example_2: Finding the minimum of the Rosenbrock function with 2 variables under constraints **"

    Ex = optim_wrapper()
    X0 = np.zeros(2)
    lim = [(-2.0, 2.0)]*2
    Ex.set_X0(X0)
    Ex.set_lim(lim)
    Ex.set_penalties_func(pen)
    Ex.set_norm_count(200)
    Ex.set_nb_best(100)
    Ex.set_obj_func(obj)
    Ex.set_wrapper()
    Ex.launch_multi_opti()
    print Ex

    X_solution = [1.0, 1.0]
    res_string = "Results of the optimisation: {:03.4f}, expected results: {:03.4f}".format(obj(Ex.get_res()), obj(X_solution))
    print res_string
    print "*" * len(res_string)

if __name__ == "__main__":
    Example_1(5)
    Example_2()