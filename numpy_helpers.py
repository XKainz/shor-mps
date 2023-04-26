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

