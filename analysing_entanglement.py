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
                 schmidt_values_measurement_before_fourier,
                 schmidt_values_measurement_after_fourier,
                 entanglement_entropy_before_fourier,
                 entanglement_entropy_fourier_mpo,
                 entanglement_entropy_fourier_circuit,
                 entanglement_entropy_measurement_before_fourier,
                 entanglement_entropy_measurement_after_fourier,
                 p_success_fourier_mpo,
                 p_success_fourier_circuit,
                 p_success_measurement,
                 samples_mpo,
                 samples_circuit,
                 samples_measurement,
                 schmidt_values_mpos,
                 tim_before_fourier,
                 tim_after_fourier,
                 tim_measurement):
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
        self.tim_after_fourier = tim_after_fourier
        self.tim_measurement = tim_measurement
        self.p_success_measurement = p_success_measurement
        self.samples_measurement = samples_measurement
        self.schmidt_values_measurement_before_fourier = schmidt_values_measurement_before_fourier
        self.schmidt_values_measurement_after_fourier = schmidt_values_measurement_after_fourier
        self.entanglement_entropy_measurement_before_fourier = entanglement_entropy_measurement_before_fourier
        self.entanglement_entropy_measurement_after_fourier = entanglement_entropy_measurement_after_fourier


def main(path_to_pickles):
    xi_start = 2**13
    pickle_objs =[]
    start = 911
    end = 4050
    path_to_pickles = path_to_pickles+"ana_ent_objs_"
    errors = []
    for i in range(start,end):
        if sh.N_valid(i):
            success = 0
            while success == 0:
                x = sh.get_x_for_N(i)
                try:
                    ana_ent_obj = get_pickle_element(x,i,xi_start)
                    if ana_ent_obj.p_success_fourier_circuit > 1e-3:
                        pickle_objs.append(ana_ent_obj)
                        success = 1
                    else: 
                        errors.append(error_obj(i,x,-1,"p_success_fourier_circuit too small"))
                        success = 0 
                except Exception as e:
                    errors.append(error_obj(i,x,-1,str(e)))
                    success = 0
            pickle.dump(pickle_objs,open(path_to_pickles+str(start)+"_"+str(end)+".pkl","wb"))
            pickle.dump(errors,open(path_to_pickles+str(start)+"_"+str(end)+"_errors.pkl","wb"))
    pickle.dump(pickle_objs,open(path_to_pickles+str(start)+"_"+str(end)+".pkl","wb"))
    pickle.dump(errors,open(path_to_pickles+str(start)+"_"+str(end)+"_errors.pkl","wb"))

def main_2(path_to_pickles):
    xi_start = 2**14
    #get x and N to analyse from pickle
    N_x = pickle.load(open("/space/ge65kox/BA/pickles/"+"N_x_values.pkl","rb"))
    x = N_x["x"]
    N = N_x["N"]
    if len(x) != len(N):
        raise Exception("x and N have different length")
    pickle_objs =[]
    errors = []
    for i in range(len(x)):
            try:
                ana_ent_obj = get_pickle_element(x[i],N[i],xi_start)
                if ana_ent_obj.p_success_fourier_circuit > 1e-3:
                    pickle_objs.append(ana_ent_obj)
                    success = 1
                else: 
                    errors.append(error_obj(i,x,-1,"p_success_fourier_circuit too small"))
                    success = 0 
            except Exception as e:
                errors.append(error_obj(N[i],x[i],-1,str(e)))
                print("Error",e)
                success = 0
            pickle.dump(pickle_objs,open(path_to_pickles+"entanglement_all"+".pkl","wb"))
            pickle.dump(errors,open(path_to_pickles+"entanglement_all"+"_errors.pkl","wb"))
    pickle.dump(pickle_objs,open(path_to_pickles+"entanglement_all"+".pkl","wb"))
    pickle.dump(errors,open(path_to_pickles+"entanglement_all"+"_errors.pkl","wb"))
    
class error_obj(object):
    def __init__(self,N,x,error,errmsg):
        self.N = N
        self.x = x
        self.error = error
        self.errmsg = errmsg

def get_pickle_element(x,N,xi):
    print("N",N)
    print("x",x)
    n_samples = 1000
    len_a = 2*int(np.ceil(np.log2(N)))+3
    cx_mpos = gts.cx_pow_2k_mod_N_mpo_from_fatU(N,x,len_a,xi)
    
    mps,len_a,tim_before_fourier = get_shor_mpo(N,x,xi,cx_mpos)
    #mps , len_a, tim_before_fourier = get_shor_fat(N,x,xi)

    schmidt_values_mpo = cx_mpos[0].get_schmidt_values_all_sites()

    schmidt_values_before_fourier = mps.get_schmidt_values_all_sites()
    print(mps.maximum_bond_dim(),"Maximum bond dimension before fourier and before measurement")
    entanglement_entropy_before_fourier = mps.get_entanglement_all_sites()

    circmps = mps
    mpomps = copy.deepcopy(circmps)

    tim_measurement = tim.Tim()
    measurement_mps = circmps.collapse_subspace(len_a,circmps.L)
    
    print(measurement_mps.maximum_bond_dim(),"Maximum bond dimension after collapse U")
    schmidt_values_measurement_before_fourier = measurement_mps.get_schmidt_values_all_sites()
    entanglement_entropy_measurement_before_fourier = measurement_mps.get_entanglement_all_sites()
    tim_measurement.print_since_last("collapse U")
    
    measurement_mps = circ.fourier_transform_MPS(measurement_mps,len_a,inv=True,rev = False)
    tim_measurement.print_since_last("fourier transform measurment")
    print(measurement_mps.maximum_bond_dim(),"Maximum bond dimension after fourier")
    schmidt_values_measurement_after_fourier = measurement_mps.get_schmidt_values_all_sites()
    entanglement_entropy_measurement_after_fourier = measurement_mps.get_entanglement_all_sites()
    
    measurement_samples = measurement_mps.sample_range(0,len_a,n_samples)
    tim_measurement.print_since_last("sampling measurement")
    p_success_measurement = sh.success_prob_measurement_samples(measurement_samples,x,N,invert=False)
    print("p_success_measurement",p_success_measurement)



    tim_after_fourier = tim.Tim()

    circmps = circ.fourier_transform_MPS(circmps,len_a,inv=True,rev=False)

    tim_after_fourier.print_since_last("fourier transform circuit")

    circ_samples = circmps.sample_range(0,len_a,n_samples)
    schmidt_values_fourier_circuit = circmps.get_schmidt_values_all_sites()
    print(circmps.maximum_bond_dim(),"Maximum bond dimension Circuit")
    entanglement_entropy_fourier_circuit = circmps.get_entanglement_all_sites()
    p_success_fourier_circuit = sh.success_prob_measurement_samples(circ_samples,x,N,invert=False)
    print("p_success_fourier_circuit",p_success_fourier_circuit)
    

    tim_after_fourier.print_since_last("sampling fourier transform circuit")

    mpomps = circ.fourier_transform_MPO(mpomps,len_a,inv=True)

    tim_after_fourier.print_since_last("fourier transform mpo")

    mpo_samples = mpomps.sample_range(0,len_a,n_samples)
    schmidt_values_fourier_mpo = mpomps.get_schmidt_values_all_sites()
    print(mpomps.maximum_bond_dim(),"Maximum bond dimension mpo mps")
    entanglement_entropy_fourier_mpo = mpomps.get_entanglement_all_sites()
    p_success_fourier_mpo = sh.success_prob_measurement_samples(mpo_samples,x,N,invert=True)
    print("p_success_fourier_mpo",p_success_fourier_mpo)

    tim_after_fourier.print_since_last("sampling fourier transform mpo")


    return analysing_entanglement_obj(N,
                                      x,
                                      schmidt_values_before_fourier,
                                      schmidt_values_fourier_mpo,
                                      schmidt_values_fourier_circuit,
                                      schmidt_values_measurement_before_fourier,#
                                      schmidt_values_measurement_after_fourier,#
                                      entanglement_entropy_before_fourier,
                                      entanglement_entropy_fourier_mpo,
                                      entanglement_entropy_fourier_circuit,
                                      entanglement_entropy_measurement_before_fourier,#
                                      entanglement_entropy_measurement_after_fourier,#
                                      p_success_fourier_mpo,
                                      p_success_fourier_circuit,
                                      p_success_measurement,#
                                      mpo_samples,
                                      circ_samples,
                                      measurement_samples,#
                                      schmidt_values_mpo,
                                      tim_before_fourier,
                                      tim_after_fourier,
                                      tim_measurement)#



if __name__ == "__main__":
    main_2("/space/ge65kox/BA/pickles/")




