from threading import Thread
from psutil import cpu_count

class thread_container(Thread):

    def __init__(self,func):
        """
        Constructor of the container used to launch function a multiple time simultaneously
        func is the function that will be launched
        """
        Thread.__init__(self)
        self.__func  = func
        self.__param = None
        self.__args  = None
        self.__res   = None

    def set_param(self,param):
        """
        Used to pass a list of parameters to the function
        """
        self.__param = param

    def get_res(self):
        """
        used to retrieve the result of the function
        """
        return self.__res

    def get_param(self):
        """
        used to retrieve the parameters used
        """
        return self.__param


    def run(self):
        """
        Used to start a thread
        Launch the function with the set of parameter self.__param
        Store the result in self.__res
        """
        self.__res = self.__func(self.__param,Full_output=True)

class thread_launcher:

    """
    Generic parallel processing launcher
    """

    def __init__(self,func,X_list):
        self.__func             = func
        self.__nb_multi_thread  = cpu_count() - 1
        self.__X_list           = X_list
        self.__nb_param         = len(X_list[0])
        self.__tot_count        = len(X_list)

    def set_multi_proc(self,param):
        """
        Function used to set the number of parallel process
        Default is the number of thread of the computer-1
        """
        self.__nb_multi_thread = param

    def launch(self):
        X_list_res  = []
        res_list    = []
        proc_list   = []
        count       = 0
        while count < self.__tot_count:
            # Creating and launching processes
            if len(proc_list) < self.__nb_multi_thread:
                proc_list.append(thread_container(self.__func))
                proc_list[-1].set_param(self.__X_list[count])
                proc_list[-1].start()
                count += 1

            # Checking and getting back the results of the launched processes
            i_proc_verif = 0
            while i_proc_verif < len(proc_list):
                if proc_list[i_proc_verif].is_alive():
                    i_proc_verif += 1
                else:
                    X_list_res.append(proc_list[i_proc_verif].get_param())
                    res_list.append(proc_list[i_proc_verif].get_res())
                    proc_list.remove(proc_list[i_proc_verif])

        return [X_list_res, res_list]