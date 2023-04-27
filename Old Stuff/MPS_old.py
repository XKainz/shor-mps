'''MPS.py'''
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Define important constants
bitDimension = 2
#smaller singular value
ssv = 1e-8

#Define helper function to analyse the psi state
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def measure_subspace_psi(psi,idx):
    ilong = [*range(len(psi.shape))]
    isum = np.setdiff1d(ilong,idx)
    abs_psi = np.multiply(psi,np.conj(psi))
    return np.sum(abs_psi,axis=totuple(isum))

def total_prabability(psi):
    return measure_subspace_psi(psi,[])

def plot_psi_as_barchart(psi):
    L = len(psi.shape)
    mpsi = np.absolute(psi.reshape(2**L))
    y_pos = np.arange(len(mpsi))
    plt.figure(figsize=(15,7))
    plt.bar(y_pos, mpsi,align='center') 
    plt.xticks(y_pos,range(len(mpsi)))
    plt.show()

def plot_psi(psi,subspace_idx):
    mpsi = measure_subspace_psi(psi,subspace_idx).flatten()
    plt.figure(figsize=(15,7))
    plt.plot(range(len(mpsi)),mpsi)
    plt.show()

def print_amplitudes(psi):
  L = len(psi.shape)
  for i in range(2**L):
    X =  np.array([0]*(L - len(list(bin(i)[2:]))) + list(bin(i)[2:]),dtype=int)
    print(X,":",psi.reshape(2**L)[i])

#Plot bond dimensions of MPS
def plot_bond_dimensions(MPS):
    plt.figure(figsize=(15,7))
    bond_dim = [MPS[i].shape[0] for i in range(0,len(MPS),2)]
    plt.plot(range(len(bond_dim)),bond_dim)
    plt.show()

def print_all_bond_norms(MPS):
    print("MPS bond norms")
    L = len(MPS)//2
    for i in range(1,L-1):
        idx = 2*i
        print(la.norm(np.diag(MPS[idx])))
    print(la.norm(np.diag(MPS[-1])))

def print_total_probability_MPS(MPS):
    print("Total probability of MPS")
    psi = MPS_to_PSI(MPS)
    print(total_prabability(psi))

def plot_schmidt_values(MPS):
    plt.figure(figsize=(15,7))
    for i in range(2,len(MPS)-1,2):
        plt.plot(range(len(np.diagonal(MPS[i]))),np.absolute(np.diagonal(MPS[i])))
    plt.show()
#Helper functions 


def print_all_shapes(MPS):
    print("MPS shapes")
    for i in range(len(MPS)):
        print(MPS[i].shape)

#flatten numpy tensor
def flatten(tensor):
    return np.reshape(tensor,(np.prod(tensor.shape),))

#renormalize complex numpy vector
def renormalize_vector(vector):
    #check if tensor is vector
    if len(vector.shape) == 1:
        return vector/la.norm(vector)
    else:
        print("Error: Tensor is not a vector")
        return -1

#MPS to PSI create one big tensor from tensor list
def MPS_to_PSI(MPS):
    PSI = MPS[0]
    for i in range(1,len(MPS)):
        PSI = np.tensordot(PSI,MPS[i],axes=([-1],[0]))
    return PSI

#Create MPS initialized at |0>
def create_MPS(L):
    MPS = [np.ones((1,1),dtype="complex")]
    for i in range(L):
        MPS.append(np.array([[[1],[0]]],dtype="complex"))
        MPS.append(np.ones((1,1),dtype="complex"))
    return MPS

#Apply 1-site gate to MPS
def apply_gate_1(MPS,gate,idx):
    i = 1+2*idx
    MPS[i] = np.tensordot(MPS[i],gate,[1,1])
    MPS[i] = np.transpose(MPS[i],[0,2,1])

def apply_U_gate(MPS,R,U,xi):
    partMPS = MPS[-3:]+[R,]
    Q = MPS_to_PSI(partMPS)
    L = len(Q.shape)-1
    if len(U.shape) != 2*L:
        raise Exception("Error: U has wrong shape")
    Q = np.tensordot(U,Q,([i for i in range(L,2*L)],[i for i in range(1,L+1)]))
    Q = np.transpose(Q,[L]+[i for i in range(L)])
    
    shape = Q.shape
    s1 = shape[0:2]
    s2 = shape[2:]
    Q = np.reshape(Q,(np.prod(s1),np.prod(s2)))
    u,s,v = la.svd(Q,full_matrices=False)
    
    #cut off singular values after xi
    u = u[:,:xi]
    s = s[:xi]
    v = v[:xi,:]
    #cut off singular values below 1e-5
    u = u[:,s>ssv]
    v = v[s>ssv,:]
    s = s[s>ssv]
    xi = min(xi,len(s))
    #renormalize singular values
    s = renormalize_vector(s)
    
    u = np.reshape(u,s1+(xi,))
    v = np.reshape(v,(xi,)+s2)
    d = np.diag(s)

    lam1 = np.diagonal(partMPS[0])
    invlam1 = 1. / lam1
    u = np.tensordot(np.diag(invlam1),u,axes=([1],[0]))

    MPS[-2:] = [u,d]
    return v






#Apply 2-site gate neighbouring gate to MPS
def apply_gate_2(MPS,gate,idx,xi):
    i = 1+2*idx
    partMPS = MPS[i-1:i+4]
    THETA = MPS_to_PSI(partMPS)
    THETA = np.tensordot(gate,THETA,([2,3],[1,2]))
    THETA = np.transpose(THETA,[2,0,1,3])
    xi1 = THETA.shape[0]
    xi2 = THETA.shape[3]
    THETA = np.reshape(THETA,(bitDimension*xi1,bitDimension*xi2))
    u,s,v = la.svd(THETA,full_matrices=False)
    #cut off singular values after xi
    u = u[:,:xi]
    s = s[:xi]
    v = v[:xi,:]
    #cut off singular values below 1e-5
    u = u[:,s>ssv]
    v = v[s>ssv,:]
    s = s[s>ssv]
    xi = min(xi,len(s))
    #renormalize singular values
    s = renormalize_vector(s)
    d = np.diag(s)
    u = np.reshape(u,(xi1,bitDimension,xi))
    v = np.reshape(v,(xi,bitDimension,xi2))
    lam1 = np.diagonal(partMPS[0])
    lam3 = np.diagonal(partMPS[4])
    invlam1 = 1. / lam1
    invlam3 = 1. / lam3
    u = np.tensordot(np.diag(invlam1),u,axes=([1],[0]))
    v = np.tensordot(v,np.diag(invlam3),axes=([2],[1]))
    MPS[i:i+3] = [u,d,v]

#Reverse bit order of MPS
def reverse_bit_order_MPS(MPS):
    L = len(MPS)//2
    for i in range(L):
        for j in range(L-i-1):
            apply_gate_2(MPS,SWAP,j,xi)

def lam_to_B_MPS(MPS):
    L = len(MPS)
    BMPS = []
    for i in range(1,L,2):
        M = np.tensordot(MPS[i],MPS[i+1],axes=([-1],[0]))
        BMPS.append(M)
    return BMPS

def to_MPS(psi,xi,cutoff=1e-8):
    L = len(psi.shape)
    print(psi.shape,"psi shape")
    #psi = np.reshape(psi,psi.shape)
    MPS = [np.ones((1,1),dtype="complex")]
    for i in range(L-1):
        s1 = psi.shape[:2]
        s2 = psi.shape[2:]
        psi = np.reshape(psi,(int(np.prod(s1)),int(np.prod(s2))))
        u,s,v = la.svd(psi,full_matrices=False)
        #cut off singular values after xi
        u = u[:,:xi]
        s = s[:xi]
        v = v[:xi,:]
        #cut off singular values below 1e-5
        u = u[:,s>cutoff]
        v = v[s>cutoff,:]
        s = s[s>cutoff]
        ximin = min(xi,len(s))
        #renormalize singular values
        s = renormalize_vector(s)
        d = np.diag(s)
        u = np.reshape(u,s1+(ximin,))
        v = np.reshape(v,(ximin,)+s2)
        MPS.append(u)
        MPS.append(d)
        psi = v
    return MPS

def to_MPO(O,xi,cutoff=1e-8):
    L = len(O.shape)
    if L%2 != 0:
        raise ValueError("O must have even number of indices")
    transpose = []
    for i in range(L//2):
        transpose = transpose + [i,L//2+i]
    O = np.transpose(O,transpose)
    O = np.reshape(O,(1,)+(4,)*(L//2))
    MPO = to_MPS(O,xi)
    print_all_shapes(MPO)
    for i in range(1,len(MPO),2):
        t = MPO[i]
        xi0 = t.shape[0]
        xi1 = t.shape[2]
        t = np.reshape(t,(xi0,2,2,xi1))
        MPO[i] = t
    return MPO

#function that gives bach array of multiplation of two 

def sample_MPS(MPS):
    BMPS = lam_to_B_MPS(MPS)
    r = np.zeros(len(BMPS)+1,dtype="int")
    contracted_tensor = np.ones((1,1))
    p_total = 1
    for i in range(len(BMPS)):
        contracted_tensor = np.tensordot(contracted_tensor[r[i],:],BMPS[i],axes=([0],[0]))
        density_matrix = np.tensordot(contracted_tensor,np.conj(contracted_tensor),axes=([-1],[-1]))/p_total
        p = np.diagonal(density_matrix)
        p = np.real_if_close(p,tol=10**4)
        r[i+1] = np.random.choice([0,1],p=p)
        p_total *= p[r[i+1]]
    return r[1:]