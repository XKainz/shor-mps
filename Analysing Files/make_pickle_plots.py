'''make_pickle_plots.py'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from to_pickle import *
import os


folder_path = "./pickles/"

def unpickle_file(filename):
    with open(filename,"rb") as f:
        to_pickle = pickle.load(f)
    to_pickle.MPO_mpo.plot_schmidt_values()
    N = str(to_pickle.N)
    x = str(to_pickle.x)
    p_suc_fat = to_pickle.p_success_fat
    #print(p_suc_fat)
    plt.figure(figsize=(17,5))
    plt.plot(p_suc_fat[:,0],p_suc_fat[:,1])
    plt.savefig("./pickles/"+N+"_"+x+"p_suc_fat.png")

    p_suc_mpo = to_pickle.p_success_mpo
    #print(p_suc_mpo)
    plt.figure(figsize=(17,5))
    plt.plot(p_suc_mpo[:,0],p_suc_mpo[:,1])
    plt.savefig("./pickles/"+N+"_"+x+"p_suc_mpo.png")

    len_a = int(2*np.ceil(np.log2(to_pickle.N)))
    psi = to_pickle.MPS_mpo.measure_subspace(0,len_a)
    plt.figure(figsize=(17,5))
    plt.plot(psi)
    plt.savefig("./pickles/"+N+"_"+x+"psi_mpo.png")
    
    psi = to_pickle.MPS_fat.measure_subspace(0,len_a)
    plt.figure(figsize=(17,5))
    plt.plot(psi)
    plt.savefig("./pickles/"+N+"_"+x+"psi_fat.png")

# Loop through every file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is actually a file (not a subdirectory)
    if os.path.isfile(os.path.join(folder_path, filename)):
        # Print the filename
        unpickle_file(os.path.join(folder_path, filename))

