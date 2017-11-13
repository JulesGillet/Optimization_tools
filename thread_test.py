import numpy as np
from threading import Thread
from random import uniform
from time import time, sleep
from psutil import cpu_count

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

def pen(X):
    """
    Constraint test
    """

    res = np.sum(X)

    return res

class thread_container(Thread):

    def __init__(self,func):
        """
        Constructor of the container used to launch function a multiple time simultaneously
        func is the function that will be launched
        """
        Thread.__init__(self)
        self.__func  = func
        self.__param = None
        self.__res   = None

    def set_param(self,param):
        self.__param = param

    def get_res(self):
        return self.__res

    def run(self):
        if isinstance(self.__func,list):
            self.__res = []
            for i_func in range(len(self.__func)):
                res_tmp = self.__func[i_func](self.__param)
                self.__res.append(res_tmp)
        else:
            self.__res = self.__func(self.__param)



def thread_test(nb_max_proc):
    nb_test = 10
    i_proc = 0
    res_list = []
    proc_list = []
    while i_proc < nb_test+nb_max_proc:
        if len(proc_list) < nb_max_proc:
            X = np.zeros(2)
            for i_param in range(len(X)):
                X[i_param] = uniform(0, 2)
            proc_list.append(thread_container([obj,pen]))
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
    # print thread_test(cpu_count()-1)
    print thread_test(1)
    print time()-tic