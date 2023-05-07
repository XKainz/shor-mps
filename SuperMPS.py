'''SuperMPS.py'''
import numpy as np
import numpy.linalg as la
import numpy_helpers as nph
import matplotlib.pyplot as plt

class SuperMPS:
    def __init__(self,*args,xi,cutoff=1e-8):
        self.xi = xi
        self.cutoff = cutoff
        if isinstance(args[0],list):
            self.MPS = args[0]
            if self.MPS[0].shape != (1,):
                self.MPS = [np.ones(1,dtype="complex")] + self.MPS
            if self.MPS[-1].shape != (1,):
                self.MPS = self.MPS + [np.ones(1,dtype="complex")]
            self.indices_per_node = len(self.MPS[1].shape)-2
            self.L = len(self.MPS)//2
            for i in range(len(self.MPS)):
                if max(self.MPS[i].shape) > self.xi:
                    raise ValueError("xi too small for MPS")

        elif isinstance(args[0],np.ndarray) and isinstance(args[1],int):
            self.indices_per_node = args[1]
            tensor = args[0]
            if len(tensor.shape)%self.indices_per_node != 0:
                raise ValueError("Tensor shape is not compatible with indices_per_node")
            self.L = len(tensor.shape)//self.indices_per_node
            if self.indices_per_node != 1:
                tensor = transpose_gate_ind_format(tensor,self.indices_per_node)
            MPS = [np.ones(1,dtype="complex")]
            tensor = np.reshape(tensor,(1,)+tensor.shape+(1,))
            for i in range(self.L):
                u,s,v = nph.trunc_svd_before_index(tensor,1+self.indices_per_node,self.xi,self.cutoff)
                u = np.einsum('k,k...->k...',1/MPS[-1],u)
                MPS.append(u)
                MPS.append(s)
                tensor = np.einsum('k,k...->k...',s,v)
            self.MPS = MPS

    def __len__(self):
        return self.L
    
    def __getitem__(self,i):
        return self.MPS[2*i+1]
    
    def __setitem__(self,i,value):
        self.MPS[2*i+1] = value
    
    def get_contracted_tensor(self,i_start,i_end):
        if i_start <0 or i_start >= self.L:
            raise ValueError("i_start out of range")
        if i_end < i_start or i_end > self.L:
            raise ValueError("i_end out of range")
        contracted_tensor = np.diag(self.get_schmidt_values(i_start,'l'))
        for i in range(i_start,i_end):
            contracted_tensor = np.tensordot(contracted_tensor,self[i],axes=([-1],[0]))
            contracted_tensor = np.einsum('...k,k->...k',contracted_tensor,self.get_schmidt_values(i,'r'))
        return contracted_tensor

    def to_MPS_index(self,i):
        return 2*i+1
    
    def to_Tensor(self):
        tensor = self.get_contracted_tensor(0,self.L)
        tensor = np.reshape(tensor,tensor.shape[1:-1])
        return tensor
    
    def get_schmidt_values(self,i,side):
        index = self.to_MPS_index(i)
        if side == 'r':
            return self.MPS[index+1]
        elif side == 'l':
            return self.MPS[index-1]
        else:
            raise ValueError("side must be 'l' or 'r'")
    
    def set_schmidt_values(self,i,side,value):
        index = self.to_MPS_index(i)
        if side == 'r':
            self.MPS[index+1] = value
        elif side == 'l':
            self.MPS[index-1] = value
        else:
            raise ValueError("side must be 'l' or 'r'")
    
    def get_A_B_config(self,j):
        ABMPS = []
        for i in range(j):
            A = np.einsum('k,k...->k...',self.get_schmidt_values(i,'l'),self[i])
            ABMPS.append(A)
        ABMPS.append(self.get_schmidt_values(j,'l'))
        for i in range(j,self.L):
            B = np.einsum('...k,k->...k',self[i],self.get_schmidt_values(i,'r'))
            ABMPS.append(B)
        return ABMPS
    
    def get_all_A(self,i,j):
        if i < 0 or i >= self.L:
            raise ValueError("i out of range")
        if j < i or j > self.L:
            raise ValueError("j out of range")
        As = []
        for k in range(i,j):
            As.append(self.get_A_of_site(k))
        return As
    
    def get_all_B(self,i,j):
        if i < 0 or i >= self.L:
            raise ValueError("i out of range")
        if j < i or j > self.L:
            raise ValueError("j out of range")
        Bs = []
        for k in range(i,j):
            Bs.append(self.get_B_of_site(k))
        return Bs
    
    def get_B_config(self):
        return self.get_A_B_config(0)
    
    def get_A_config(self):
        return self.get_A_B_config(self.L)
    
    def get_A_of_site(self,i):
        A = np.einsum('k,k...->k...',self.get_schmidt_values(i,'l'),self[i])
        return A
    
    def get_B_of_site(self,i):
        B = np.einsum('...k,k->...k',self[i],self.get_schmidt_values(i,'r'))
        return B
    
    def print_all_shapes(self):
        for i in range(self.to_MPS_index(len(self))):
            print("[",i,"]",self.MPS[i].shape)
    
    def plot_schmidt_values(self):
        plt.figure(figsize=(15,7))
        for i in range(self.L):
            d = self.get_schmidt_values(i,'l')
            plt.plot(range(len(d)),np.absolute(d),label="i="+str(i))
        plt.show()

    def plot_bond_dims(self):
        plt.figure(figsize=(15,7))
        d = []
        for i in range(self.L-1):
            d.append(self[i].shape[-1])
        plt.plot(range(len(d)),np.absolute(d),label="Bond Dimensions from i to i+1")
        plt.show()
    
    def maximum_bond_dim(self):
        d = []
        for i in range(self.L-1):
            d.append(self[i].shape[-1])
        return max(d)

    def set_xi(self,xi):
        s = self.get_schmidt_values(0,'l')
        xin = min(xi,len(s))
        if xin < len(s):
            s = s[:xin]
            self.set_schmidt_values(0,'l',s)
            self[0] = self[0][:xin,...]
        for i in range(self.L-1):
            s = self.get_schmidt_values(i,'r')
            xin = min(xi,len(s))
            if xin < len(s):
                s = s[:xin]
                s = nph.renormalize_vector(s)
                self.set_schmidt_values(i,'r',s)
                self[i] = self[i][...,:xin]
                self[i+1] = self[i+1][:xin,...]
        s = self.get_schmidt_values(self.L-1,'r')
        xin = min(xi,len(s))
        if xin < len(s):
            s = s[:xin]
            self.set_schmidt_values(0,'r',s)
            self[self.L-1] = self[self.L-1][:xin,...]
        self.xi = xi

def transpose_gate_ind_format(gate,ind_per_node):
    if len(gate.shape)%ind_per_node != 0:
        raise ValueError("Tensor shape is not compatible with indices_per_node")
    L = len(gate.shape)//ind_per_node
    tranpose_vec = [0,]*len(gate.shape)
    for i in range(L):
        for j in range(ind_per_node):
            tranpose_vec[i*ind_per_node+j] = i+j*L
    return np.transpose(gate,tranpose_vec)