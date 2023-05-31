import pickle 
import numpy as np
from qm_shors_algorithm import *
import Gates as gts
import shor_helpers as sh
import copy
from tim import Tim
import matplotlib.pyplot as plt
import numpy_helpers as nph

def plot_inverse_samples(samples,inv=False):
    psi = np.zeros(2**len(samples[0,:]))
    for i in range(len(samples[:,0])):
        if inv:
            psi[nph.binary_array_to_number(np.flip(samples[i,:]))] += 1
        else:
            psi[nph.binary_array_to_number(samples[i,:])] += 1
    return psi/np.sum(psi)


class analysing_entanglement_obj(object):
    def __init__(self,
                 N,
                 x,
                 schmidt_values_before_fourier,
                 schmidt_values_fourier_mpo,
                 schmidt_values_fourier_circuit,
                 entanglement_entropy_before_fourier,
                 entanglement_entropy_fourier_mpo,
                 entanglement_entropy_fourier_circuit,
                 p_success_fourier_mpo,
                 p_success_fourier_circuit,
                 samples_mpo,
                 samples_circuit,
                 schmidt_values_mpos,
                 tim_before_fourier,
                 tim_fourier):
        self.N = N
        self.x = x
        self.schmidt_values_before_fourier = schmidt_values_before_fourier
        self.schmidt_values_fourier_mpo = schmidt_values_fourier_mpo
        self.schmidt_values_fourier_circuit = schmidt_values_fourier_circuit
        self.entanglement_entropy_before_fourier = entanglement_entropy_before_fourier
        self.entanglement_entropy_fourier_mpo = entanglement_entropy_fourier_mpo
        self.entanglement_entropy_fourier_circuit = entanglement_entropy_fourier_circuit
        self.p_success_fourier_mpo = p_success_fourier_mpo
        self.p_success_fourier_circuit = p_success_fourier_circuit
        self.samples_mpo = samples_mpo
        self.samples_circuit = samples_circuit
        self.schmidt_values_mpo = schmidt_values_mpos
        self.tim_before_fourier = tim_before_fourier

def main(path_to_pickles):
    xi_start = 2**13
    pickle_objs =[]
    start = 30
    end = 300
    path_to_pickles = path_to_pickles+"ana_ent_objs_"
    for i in range(start,end):
        if sh.N_valid(i):
            success = 0
            while success == 0:
                x = sh.get_x_for_N(i)
                #try:
                ana_ent_obj = get_pickle_element(x,i,xi_start)
                try:
                    if ana_ent_obj.p_success_fourier_circuit > 1e-3:
                        pickle_objs.append(ana_ent_obj)
                        success = 1
                except Exception as e:
                    print(e)
                    success = 0
        if i%10 == 0:
            pickle.dump(pickle_objs,open(path_to_pickles+str(start)+"_"+str(end)+".pkl","wb"))
    pickle.dump(pickle_objs,open(path_to_pickles+str(start)+"_"+str(end)+".pkl","wb"))
    

def get_pickle_element(x,N,xi):
    print("N",N)
    print("x",x)
    n_samples = 1000
    len_a = 2*int(np.ceil(np.log2(N)))+3
    cx_mpos = gts.cx_pow_2k_mod_N_mpo_from_fatU(N,x,len_a,xi)
    mps,len_a,tim_before_fourier = get_shor_mpo(N,x,xi,cx_mpos)

    schmidt_values_mpo = cx_mpos[0].get_schmidt_values_all_sites()

    schmidt_values_before_fourier = mps.get_schmidt_values_all_sites()
    print(mps.maximum_bond_dim(),"Maximum bond dimension before fourier")
    entanglement_entropy_before_fourier = mps.get_entanglement_all_sites()

    circmps = mps
    mpomps = copy.deepcopy(circmps)

    tim_after_fourier = tim.Tim()

    circmps = circ.fourier_transform_MPS(circmps,len_a,inv=True,rev=False)
    circ_samples = circmps.sample_range(0,len_a,n_samples)
    schmidt_values_fourier_circuit = circmps.get_schmidt_values_all_sites()
    print(circmps.maximum_bond_dim(),"Maximum bond dimension Circuit")
    entanglement_entropy_fourier_circuit = circmps.get_entanglement_all_sites()
    p_success_fourier_circuit = sh.success_prob_measurement_samples(circ_samples,x,N,invert=False)
    print("p_success_fourier_circuit",p_success_fourier_circuit)
    

    tim_after_fourier.print_since_last("fourier transform circuit")

    mpomps = circ.fourier_transform_MPO(mpomps,len_a,inv=True)
    mpo_samples = mpomps.sample_range(0,len_a,n_samples)
    schmidt_values_fourier_mpo = mpomps.get_schmidt_values_all_sites()
    print(mpomps.maximum_bond_dim(),"Maximum bond dimension mpo mps")
    entanglement_entropy_fourier_mpo = mpomps.get_entanglement_all_sites()
    p_success_fourier_mpo = sh.success_prob_measurement_samples(mpo_samples,x,N,invert=True)
    print("p_success_fourier_mpo",p_success_fourier_mpo)

    tim_after_fourier.print_since_last("fourier transform mpo")

    return analysing_entanglement_obj(N,
                                      x,
                                      schmidt_values_before_fourier,
                                      schmidt_values_fourier_mpo,
                                      schmidt_values_fourier_circuit,
                                      entanglement_entropy_before_fourier,
                                      entanglement_entropy_fourier_mpo,
                                      entanglement_entropy_fourier_circuit,
                                      p_success_fourier_mpo,
                                      p_success_fourier_circuit,
                                      mpo_samples,
                                      circ_samples,
                                      schmidt_values_mpo,
                                      tim_before_fourier,
                                      tim_after_fourier)



if __name__ == "__main__":
    main("/space/ge65kox/BA/pickles/")



#get_pickle_element(150,391,2**13)

