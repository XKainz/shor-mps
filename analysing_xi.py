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
        self.xi_max_in_mps = xi_max_in_mps
        self.p_success_mpo = p_success_mpo
        self.xi_mpo_regular = xi_mpo_regular
        self.xi_max_regular = xi_max_regular

def main_start_end(path_to_pickles):
    xi_mps = 4
    xi_mpo = 16
    pickle_objs =[]
    start = 20
    end = 10000
    path_to_pickles = path_to_pickles+"ana_ent_objs_"
    for i in range(start,end):
        if sh.N_valid(i):
            success = 0
            while success == 0:
                x = sh.get_x_for_N(i)
                ana_ent_obj = get_analysing_xi_obj(i,x,xi_mpo,xi_mps)
                try:
                    
                    if ana_ent_obj.p_success_mpo > 0:
                        pickle_objs.append(ana_ent_obj)
                        success = 1
                except Exception as e:
                    print(e)
                    success = 0
            pickle.dump(pickle_objs,open(path_to_pickles+str(start)+"_"+str(end)+"_xi_mpo_"+str(xi_mpo)+"_xi_mps_"+str(xi_mps)+"_2.pkl","wb"))

def main_N_x_given(path_to_pickles,N,x):
    pass

def get_analysing_xi_obj(N,x,xi_mpo,xi_mps):
    print("N: ",N)
    print("x: ",x)
    print("xi_mpo: ",xi_mpo)
    print("xi_mps: ",xi_mps)
    n_samples = 10**4
    len_a = 2*int(np.ceil(np.log2(N)))+3

    
    cx_mpos, tim_create_mpos  = gts.cx_pow_2k_mod_N_mpo_mpo(N,x,len_a,xi_mpo)
    

    xi_max_in_mpos = [mpo.maximum_bond_dim() for mpo in cx_mpos]

    mps , len_a, tim_before_fourier = get_shor_mpo(N,x,xi_mps,cx_mpos)

    tim_after_fourier = tim.Tim()
    
    mps = circ.fourier_transform_MPO(mps,len_a,inv=True)
    xi_max_in_mps = mps.maximum_bond_dim()
    
    tim_after_fourier.print_since_last("Fourier transform")

    samples =  mps.sample_range(0,len_a,n_samples)

    tim_after_fourier.print_since_last("Sampling")

    p_success_mpo = sh.success_prob_measurement_samples(samples,x,N,invert=True)
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
                            xi_max_in_mps)

if __name__ == "__main__":
    main_start_end("/space/ge65kox/BA/pickles/")


#get_analysing_xi_obj(15,4,2**8,2**8)     
