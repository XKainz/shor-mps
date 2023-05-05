'''numpy_helpers.py'''
import numpy as np
import numpy.linalg as la

#renormalize complex numpy vector
def renormalize_vector(vector):
    #check if tensor is vector
    if len(vector.shape) == 1:
        return vector/la.norm(vector)
    else:
        raise ValueError("Tensor is not a vector")
    
def trunc_svd(tensor,xi,cutoff=1e-8):
    u,s,v = la.svd(tensor,full_matrices=False)
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
    return u,s,v,ximin

def trunc_svd_before_index(tensor,index,xi,cutoff=1e-8):
    if index <= 0 or index >= len(tensor.shape):
        raise ValueError("Index out of range")
    #reshape tensor to 2d
    s1 = tensor.shape[:index]
    s2 = tensor.shape[index:]
    tensor = np.reshape(tensor,(int(np.prod(s1)),int(np.prod(s2))))
    #perform SVD
    u,s,v,ximin = trunc_svd(tensor,xi,cutoff)
    #reshape u and v
    u = np.reshape(u,s1+(ximin,))
    v = np.reshape(v,(ximin,)+s2)
    return u,s,v

def number_to_binary_array(N):
    if N < 0:
        raise ValueError("N must be positive")
    if N == 0:
        return np.array([0])
    else:
        return [int(i) for i in bin(N)[2:]]
    
