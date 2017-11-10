import numpy as np
from threading import Thread
from random import uniform
from time import time, sleep

def obj(X):
    """
    Objective function: Rosenbrock function with n variables
    Minimum for obj(1.0,...,1.0) = 0.0
    """
    sleep(0.1)
    res = 0.0
    for i_param in range(len(X)-1):
        res += 100.0*(X[i_param+1]-X[i_param]**2)**2 + (X[i_param]-1.0)**2
    return res

class thread_container(Thread):

    def __init__(self,func):
        """
        Constructor of the container used to launch function a multiple time simultaneously
        func is the function that will be launched
        """
        Thread.__init__(self)
        self.__func     = func
        self.__param    = None
        self.__res      = None

    def set_param(self,param):
        self.__param = param

    def get_res(self):
        return self.__res

    def run(self):
        self.__res = self.__func(self.__param)


def thread_test(nb_max_proc):
    nb_test = 100
    i_proc = 0
    res_list = []
    proc_list = []
    while i_proc < nb_test+nb_max_proc:
        if len(proc_list) < nb_max_proc:
            X = np.zeros(2)
            for i_param in range(len(X)):
                X[i_param] = uniform(0, 2)
            proc_list.append(thread_container(obj))
            proc_list[-1].set_param(X)
            proc_list[-1].start()
            i_proc += 1
        i_proc_verif = 0
        while i_proc_verif< len(proc_list):
            if proc_list[i_proc_verif].is_alive():
                i_proc_verif += 1
            else:
                res_list.append(proc_list[i_proc_verif].get_res())
                proc_list.remove(proc_list[i_proc_verif])
    return res_list

if __name__ == "__main__":
    tic = time()
    print len(thread_test(1))
    print time()-tic

    tic = time()
    print len(thread_test(2))
    print time() - tic

    tic = time()
    print len(thread_test(4))
    print time() - tic

    tic = time()
    print len(thread_test(10))
    print time() - tic