import pickle 
import numpy as np
from qm_shors_algorithm import *
import Gates as gts
import shor_helpers as sh
import copy
from tim import Tim
import matplotlib.pyplot as plt
import numpy_helpers as nph

class analysing_xi_obj(object):
    def __init__(self,
                  N,
                  x,
                  xi_mpo,
                  xi_mps,
                  p_success_mpo,
                  samples_mpo,
                  tim_create_mpos,
                  tim_before_fourier,
                  tim_after_fourier,
                  xi_max_in_mpos,
                  xi_max_in_mpos_zip_up,
                  xi_max_in_mps,
                  xi_mpo_regular=-1,
                  xi_max_regular=-1,
                  p_success_regular=-1):
        self.N = N
        self.x = x
        self.xi_mpo = xi_mpo
        self.xi_mps = xi_mps
        self.p_success_regular = p_success_regular
        self.samples_mpo = samples_mpo
        self.tim_create_mpos = tim_create_mpos
        self.tim_before_fourier = tim_before_fourier
        self.tim_after_fourier = tim_after_fourier
        self.xi_max_in_mpos = xi_max_in_mpos
        self.xi_max_in_mpos_zip_up = xi_max_in_mpos_zip_up
        self.xi_max_in_mps = xi_max_in_mps
        self.p_success_mpo = p_success_mpo
        self.xi_mpo_regular = xi_mpo_regular
        self.xi_max_regular = xi_max_regular

def main_start_end(path_to_pickles,N,x):
    xi_mpo = 2**13
    xi_mps = 2**13
    pickle_objs = []
    errors = []
    pickle_objs.append(get_analysing_xi_obj(N,x,xi_mpo,xi_mps))
    xi_mpo = np.max(pickle_objs[0].xi_max_in_mpos_zip_up)
    xi_mps = xi_mpo
    endmps = 3
    endmpo = 2
    stepsize = 1
    i = xi_mpo
    j = xi_mps
    while i > endmps:
        print(i, "xi_mps outside")
        j = xi_mpo
        while j > endmpo:
            print(j, "xi_mpo outside")
            try:
                pickle_objs.append(get_analysing_xi_obj(N,x,j,i))
            except Exception as e:
                print(e)
                errors.append(error_obj(N,x,j,i,-1,str(e)))
                endmpo = j
            j -= stepsize
            pickle.dump(pickle_objs,open(path_to_pickles+"xi_analysis_N_"+str(N)+"_x_"+str(x)+".pkl","wb"))
            pickle.dump(errors,open(path_to_pickles+"xi_analysis_N_"+str(N)+"_x_"+str(x)+"_errors.pkl","wb"))
        i -= stepsize
    pickle.dump(pickle_objs,open(path_to_pickles+"xi_analysis_N_"+str(N)+"_x_"+str(x)+".pkl","wb"))
    pickle.dump(errors,open(path_to_pickles+"xi_analysis_N_"+str(N)+"_x_"+str(x)+"_errors.pkl","wb"))

class error_obj(object):
    def __init__(self,N,x,xi_mpo,xi_mps,error,errmsg):
        self.N = N
        self.x = x
        self.xi_mpo = xi_mpo
        self.xi_mps = xi_mps
        self.error = error
        self.errmsg = errmsg


def get_analysing_xi_obj(N,x,xi_mpo,xi_mps):
    print("N: ",N)
    print("x: ",x)
    print("xi_mpo within object: ",xi_mpo)
    print("xi_mps within object: ",xi_mps)
    n_samples = 10**4
    len_a = 2*int(np.ceil(np.log2(N)))+3

    
    cx_mpos, tim_create_mpos  = gts.cx_pow_2k_mod_N_mpo_mpo(N,x,len_a,xi_mpo)
    

    xi_max_in_mpos = [mpo.maximum_bond_dim() for mpo in cx_mpos]
    xi_max_in_mpos_zip_up = [mpo.largest_xi_during_zip_up for mpo in cx_mpos]
    

    mps , len_a, tim_before_fourier = get_shor_mpo(N,x,xi_mps,cx_mpos)
    xi_max_in_mps = mps.maximum_bond_dim()

    tim_after_fourier = tim.Tim()
    new = mps.collapse_subspace(len_a,mps.L)
    tim_after_fourier.print_since_last("Collapse subspace")

    new = circ.fourier_transform_MPS(new,len_a,inv=True)
    
    tim_after_fourier.print_since_last("Fourier transform")

    samples =  new.sample_range(0,len_a,n_samples)

    tim_after_fourier.print_since_last("Sampling")

    p_success_mpo = sh.success_prob_measurement_samples(samples,x,N,invert=False)
    print("p_success_mpo: ",p_success_mpo)
    
    return analysing_xi_obj(N,
                            x,
                            xi_mpo,
                            xi_mps,
                            p_success_mpo,
                            samples,
                            tim_create_mpos,
                            tim_before_fourier,
                            tim_after_fourier,
                            xi_max_in_mpos,
                            xi_max_in_mpos_zip_up,
                            xi_max_in_mps)

if __name__ == "__main__":
    main_start_end("/space/ge65kox/BA/pickles/analysing_xi_max/",45,23)





