from scipy.optimize import fmin_l_bfgs_b
from constraints_wrapper import *
from time import time
from random import uniform
import warnings
from threading import Thread, RLock

lock = RLock()

class optim_wrapper:

    def __init__(self):
        """
        Optim wrapper constructor
        Create all variables used in the wrapper
        See bellow for variables descriptions
        """

        # Starting point related variable
        self.__X0       = None
        self.__nb_param = 0

        # Boundaries related variable
        self.__real_lim     = None
        self.__real_lim_sup = None
        self.__real_lim_inf = None
        self.__lim          = []

        # Objective function related variable
        self.__real_obj_func = None

        # Penalty function related variable
        self.__pen_func = None
        self.__is_pen   = 0
        self.__nb_pen   = None

        # Normalization related variable
        self.__norm_count = 50.0
        self.__norm_obj   = 1.0
        self.__norm_pen   = [1.0]

        # enhanced multi start variables
        self.__nb_best     = 1
        self.__X_best_list = []

        # Results related variable
        self.__res_opt    = [0.0]
        self.__final_grad = [0.0]
        self.__final_pen  = [0.0]

        # Statistics related variable
        self.__obj_calc_time    = []
        self.__total_calc_time  = [0.0,0.0,0.0]
        self.__total_init_time  = [0.0,0.0]
        self.__nb_of_calls      = 0

    def set_X0(self,X0):
        """
        Function used to pass the initial guess for the optimization
        Also used to get the number of variables
        X0 can be a float, a list or an array
        """
        self.__X0       = np.array(X0)
        self.__nb_param = len(X0)

    def set_lim(self,real_lim):
        """
        Function used to pass the boundaries of the variables
        Also used to check the compliance between the number of parameters and the number of boundaries
        real_lim must be a list of tuple
        """
        if len(real_lim) == self.__nb_param:
            self.__real_lim     = real_lim
            self.__real_lim_sup = np.zeros(self.__nb_param)
            self.__real_lim_inf = np.zeros(self.__nb_param)
            for i_lim in range(self.__nb_param):
                self.__real_lim_sup[i_lim] = self.__real_lim[i_lim][1]
                self.__real_lim_inf[i_lim] = self.__real_lim[i_lim][0]
        else:
            raise ValueError("size of bounds ({0}) is not compliant with the size of X0 ({1})".format(len(real_lim),self.__nb_param))

    def set_obj_func(self,func):
        """
        Function used to pass the objective function
        func must be a function
        """
        self.__real_obj_func = func

    def set_penalties_func(self,func):
        """
        Function used to pass the penalties function (optional)
        func must be a function
        Also used to check if func return a list (mandatory)
        """
        self.__is_pen   = 1
        self.__pen_func = func
        pen_tmp         = self.__pen_func(self.__X0)
        if isinstance(pen_tmp,list):
            self.__nb_pen = len(pen_tmp)
        else:
            raise ValueError("The penalty function must return a list")

    def set_norm_count(self,norm_count):
        """
        Function used to set the normalisation time
        norm_time can be a float or an array
        """
        self.__norm_count = norm_count

    def set_nb_best(self,nb_best):
        """
        Function used to set the number of best normalisation results (optional)
        nb_best must be a int
        """
        self.__nb_best = nb_best

    def get_res(self):
        """
        Function used to get the results of optimization
        Return a list of array
        """
        return self.__res_opt

    def get_grad(self):
        """
        Function used to get the final gradient
        Return a list of array
        """
        return self.__final_grad

    def get_pen(self):
        """
        Function used to get the final values of the penalties
        Return a list of array
        """
        return self.__final_pen

    def __rescale(self,X):
        """
        Function used to rescale the X vector from (0.0,1.0) to the real boundaries
        Return the rescaled parameter list of array
        """
        real_X = np.zeros(self.__nb_param)
        for i_param in range(self.__nb_param):
            real_X[i_param] = X[i_param]*(self.__real_lim_sup[i_param]-self.__real_lim_inf[i_param]) + self.__real_lim_inf[i_param]
        return real_X

    def __norm(self):
        """
        Function used to normalize the outputs on the objective and penalties functions
        Normalization happens during self.__norm_count seconds
        A security exist in order to evaluate all the functions more than 10 but less than 1e4 times
        """
        res_list = []
        X_list   = []
        out_res  = []
        pen_res  = []
        count    = 0.0
        i_proc = 0
        proc_list = []
        while count<10 or (count<self.__norm_count and count<10000.0):
            X = np.zeros(self.__nb_param)
            for i_param in range(self.__nb_param):
                X[i_param] = uniform(0.0,1.0)
            X_list.append(X)
            res_list.append(self.__obj_func(X))
            out_res.append(self.__obj)
            pen_res.append(self.__pen)
            count += 1.0

            #TODO multi proc
            for i in range(3):
                proc_list.append(thread_container(self.__obj_func))
                proc_list[i_proc].set_param(X)
                proc_list[i_proc].start()
                proc_list[i_proc].join()
                i_proc += 1
                with lock:
                    X_list.append(X)
                    res_list.append(self.__obj_func(X))
                    out_res.append(self.__obj)
                    pen_res.append(self.__pen)
                    count += 1.0
                    
        # Keeping the bests results of the normalization to use them for a multi start
        if count>self.__nb_best:
            Ind_list_best = sorted(range(len(res_list)), key=lambda i: res_list[i])[:self.__nb_best]
            for i in range(self.__nb_best):
                self.__X_best_list.append(X_list[Ind_list_best[i]])
        else:
            warnings.warn("Warning: The number of starting point is higher than the number of evaluation of the objective function. Set a higher normalization count (set_norm_count) and/or set a lower number of starting point (set_nb_best)")
            self.__X_best_list = X_list
            self.__nb_best = len(self.__X_best_list)

        self.__norm_obj = np.mean(out_res)

        if self.__is_pen == 1:
            self.__norm_pen = []
            for i_pen in range(self.__nb_pen):
                pen_tmp = []
                for i in range(len(pen_res)):
                    pen_tmp.append(pen_res[i][i_pen])
                self.__norm_pen.append(np.mean(pen_tmp))

    def __obj_func(self,X):
        """
        Function used to calculate and apply the normalization of the objective and penaltiy functions
        This function will be minimized by the fmin_l_bfgs_b function
        """
        t0 = time()
        self.__nb_of_calls += 1
        real_X = self.__rescale(X)
        self.__obj = self.__real_obj_func(real_X)/self.__norm_obj
        if self.__is_pen == 1:
            self.__pen = np.array(self.__pen_func(real_X))/np.array(self.__norm_pen)
        else:
            self.__pen = [0.0,0.0]

        self.__obj_calc_time.append(time()-t0)

        return self.__obj + np.sum(self.__pen)

    def set_wrapper(self):
        """
        Function used to initialize the wrapper
        This function normalize the problem and create generic boundaries
        """
        t0 = time()
        self.__norm()
        if np.isnan(self.__norm_obj) or np.isinf(self.__norm_obj):
            raise ValueError("Normalization failed, result is NaN or inf")
        self.__lim = []
        for i_bounds in range(self.__nb_param):
            self.__lim.append((0.0, 1.0))

        # Time calculation (statistics related)
        calc_time_tmp = time()-t0
        self.__total_init_time[0] = int(np.floor(calc_time_tmp / 60.0))
        calc_time_tmp -= self.__total_calc_time[1] * 60.0
        self.__total_init_time[1] = calc_time_tmp

    def launch_opti(self):
        """
        Function used to launch the optimization function
        Also calculate computation times for statistics purpose
        """
        t0 = time()
        maxls_calc = max(self.__nb_param, 20)
        res = fmin_l_bfgs_b(self.__obj_func, x0=self.__X0, approx_grad=True, bounds=self.__lim, m=self.__nb_param, factr=1e12, pgtol=1e-05, epsilon=1e-04, maxls=maxls_calc)
        self.__obj_func(res[0])
        self.__res_opt    = self.__rescale(res[0])
        self.__final_grad = res[2]['grad']
        self.__final_pen  = self.__pen*np.array(self.__norm_pen)

        # Time calculation (statistics related)
        calc_time_tmp             = time() - t0
        self.__total_calc_time[0] = int(np.floor(calc_time_tmp/60.0/60.0))
        calc_time_tmp -= self.__total_calc_time[0]*60.0*60.0
        self.__total_calc_time[1] = int(np.floor(calc_time_tmp/60.0))
        calc_time_tmp -= self.__total_calc_time[1]*60.0
        self.__total_calc_time[2] = calc_time_tmp

    def launch_multi_opti(self):
        """
        Function used to launch the optimization function multiple time
        The multi start is based on the nb_best results of the normalization function
        Also calculate computation times for statistics purpose
        """

        t0 = time()
        maxls_calc = max(self.__nb_param, 20)
        f_min = 1e20
        for i_best in range(self.__nb_best):
            X0 = self.__X_best_list[i_best]
            res = fmin_l_bfgs_b(self.__obj_func, x0=X0, approx_grad=True, bounds=self.__lim, m=self.__nb_param,factr=1e12, pgtol=1e-05, epsilon=1e-04, maxls=maxls_calc)
            self.__obj_func(res[0])
            if self.__obj<f_min:
                f_min = self.__obj
                self.__res_opt = self.__rescale(res[0])
                self.__final_grad = res[2]['grad']
                self.__final_pen = self.__pen*np.array(self.__norm_pen)

        # Time calculation (statistics related)
        calc_time_tmp = time() - t0
        self.__total_calc_time[0] = int(np.floor(calc_time_tmp / 60.0 / 60.0))
        calc_time_tmp -= self.__total_calc_time[0] * 60.0 * 60.0
        self.__total_calc_time[1] = int(np.floor(calc_time_tmp / 60.0))
        calc_time_tmp -= self.__total_calc_time[1] * 60.0
        self.__total_calc_time[2] = calc_time_tmp

    def __repr__(self):
        """
        Function used to print results and statistics of the optimization wrapper
        """
        info_string = ""
        info_string += "\n** Results of the optimization **"
        if len(self.__res_opt)<=10:
            info_string += "\n  - Optimization results: {0}".format(self.__res_opt)
            info_string += "\n  - Final gradient:       {0}".format(self.__final_grad)
        else:
            info_string += "\n  - Solution and gradient are too long to be displayed, please refer to get_res and get_grad method"
        if self.__is_pen == 1:
            if self.__nb_pen<10:
                info_string += "\n  - Final penalties:      {0}".format(self.__final_pen)
            else:
                info_string += "\n  - Penalties are too long to be displayed, please refer to get_pen method"
        info_string += "\n"
        info_string += "\n** Statistics of the optimization **"
        info_string += "\n  - Number of objective function call =               {0} times".format(self.__nb_of_calls)
        info_string += "\n  - Total initialization time =                       {0}mn {1:2.3f}s".format(self.__total_init_time[0], self.__total_init_time[1])
        info_string += "\n  - Total optimization time =                         {0}h {1}mn {2:2.3f}s".format(self.__total_calc_time[0],self.__total_calc_time[1],self.__total_calc_time[2])

        if np.mean(self.__obj_calc_time)<1:
            info_string += "\n  - Mean calculation time of the objective function = {0:4.2f}ms".format(np.mean(self.__obj_calc_time)*1000.0)
        else:
            info_string += "\n  - Mean calculation time of the objective function = {0:4.2f}s".format(np.mean(self.__obj_calc_time))

        info_string += "\n\n"

        return info_string

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