from optim_wrapper import optim_wrapper
from constraints_wrapper import *

def obj(X):
    return X[0]**2 + X[1]

def pen(X):
    return [Ineg_wrapper(X[0],X[1]), Eg_wrapper(X[0],2.0)]

test = optim_wrapper()
test.set_X0([10.0,10.0])
test.set_lim([(1.0,10.0),(-1.0,20.0)])
test.set_obj_func(obj)
test.set_penalties_func(pen)
test.set_wrapper()
test.launch_opti()