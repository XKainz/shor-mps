'''numpy_helpers.py'''
import numpy as np
import numpy.linalg as la

def von_neumann_entropy(s):
    return -np.sum(s**2 *np.log(s**2))

#renormalize complex numpy vector
def renormalize_vector(vector):
    #check if tensor is vector
    if len(vector.shape) == 1:
        return vector/la.norm(vector)
    else:
        raise ValueError("Tensor is not a vector")
    
def trunc_svd(tensor,xi,cutoff=1e-8,norm=1):
    u,s,v = la.svd(tensor,full_matrices=False)
    #cut off singular values after xi
    scopy = np.copy(s)
    u = u[:,:xi]
    s = s[:xi]
    v = v[:xi,:]
    #cut off singular values below 1e-5
    u = u[:,s>cutoff]
    v = v[s>cutoff,:]
    s = s[s>cutoff]
    ximin = min(xi,len(s))
    #renormalize singular values
    xi_away = scopy[ximin:]
    if norm>0:
        s = norm*renormalize_vector(s)
    return u,s,v,ximin,xi_away

def trunc_svd_before_index(tensor,index,xi,cutoff=1e-8,norm=1):
    u,s,v, ximin = trunc_svd_before_index_ximin(tensor,index,xi,cutoff,norm)
    return u,s,v


def trunc_svd_before_index_xi_min_away(tensor,index,xi,cutoff=1e-8,norm=1):
    if index <= 0 or index >= len(tensor.shape):
        raise ValueError("Index out of range")
    #reshape tensor to 2d
    s1 = tensor.shape[:index]
    s2 = tensor.shape[index:]
    tensor = np.reshape(tensor,(int(np.prod(s1)),int(np.prod(s2))))
    #perform SVD
    u,s,v,ximin,xi_away = trunc_svd(tensor,xi,cutoff,norm=norm)
    #reshape u and v
    u = np.reshape(u,s1+(ximin,))
    v = np.reshape(v,(ximin,)+s2)
    return u,s,v, ximin, xi_away

def trunc_svd_before_index_ximin(tensor,index,xi,cutoff=1e-8,norm=1):
    u,s,v,ximin,xi_away = trunc_svd_before_index_xi_min_away(tensor,index,xi,cutoff,norm=norm)
    return u,s,v, ximin

def qr_before_index(tensor,index):
    if index <= 0 or index >= len(tensor.shape):
        raise ValueError("Index out of range")
    #reshape tensor to 2d
    s1 = tensor.shape[:index]
    s2 = tensor.shape[index:]
    tensor = np.reshape(tensor,(int(np.prod(s1)),int(np.prod(s2))))
    #perform QR
    q,r = la.qr(tensor)
    #reshape q and r
    q = np.reshape(q,s1+(q.shape[-1],))
    r = np.reshape(r,(r.shape[0],)+s2)
    return q,r

def number_to_binary_array(N):
    if N < 0:
        raise ValueError("N must be positive")
    else:
        return [int(i) for i in bin(N)[2:]]
    
def binary_array_to_number(v):
    n = 0
    for i in range(len(v)):
        n += 2**i*v[len(v)-1-i]
    return n
    
def get_random_unitary(d):
    #get random complex matrix
    A = np.random.rand(d,d) + 1j*np.random.rand(d,d)
    #make it unitary
    A = la.qr(A)[0]
    return A
