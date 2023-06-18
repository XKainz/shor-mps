'''MPS.py'''
from SuperMPS import SuperMPS
import numpy as np
import numpy_helpers as nph
import numpy.linalg as la
import Gates as gt
import tim
import time
import copy


class MPS(SuperMPS):
    def __init__(self,mps_array,L,indices_per_node,xi,cutoff):
        super().__init__(mps_array,L,indices_per_node,xi,cutoff)
        if self.indices_per_node != 1:
            raise ValueError("MPS must have 1 index per node")
    
    @staticmethod
    def create_MPS_init_to_r(r,xi,cutoff=1e-8):
        mps = [np.ones(1)]
        for i in r:
            if i == 0:
                mps.append(np.array([[[1],[0]]]))
            elif i == 1:
                mps.append(np.array([[[0],[1]]]))
            else:
                raise ValueError("r not binary")
            mps.append(np.ones(1))
        mps_object = SuperMPS.create_SuperMPS_from_tensor_array(mps,xi,cutoff=cutoff)
        mps_object.__class__ = MPS
        return mps_object

    @staticmethod
    def create_MPS_init_to_N(N,L,xi,cutoff=1e-8):
        if N < 0:
            raise ValueError("N must be positive")
        if N == 0:
            return MPS.create_MPS_init_to_r([0,]*L,xi,cutoff)
        elif np.ceil(np.log2(N)) > L:
            raise ValueError("N must be less than 2**L")
        r = nph.number_to_binary_array(N)
        zeros = [0,]*(L-len(r))
        r = zeros + r
        return MPS.create_MPS_init_to_r(r,xi,cutoff)

    @staticmethod
    def create_MPS_init_to_1(length,xi,cutoff=1e-8):
        return MPS.create_MPS_init_to_N(1,length,xi,cutoff)
    
    def apply_1_site_gate(self, gate, i):
        if i < 0 or i >= self.L:
            raise ValueError("i must be in range [0,self.L)")
        if gate.shape != (2,2):
            raise ValueError("gate must be a 1x1 matrix")
        self[i] = np.tensordot(gate,self[i],axes=(1,1)).transpose((1,0,2))

    def apply_2_site_gate(self,gate,i):
        if i < 0 or i >= self.L-1:
            raise ValueError("i must be in range [0,self.L-1)")
        if gate.shape != (2,)*4:
            raise ValueError("gate must be a 2x2x2x2 matrix")
        theta = self.get_contracted_tensor(i,i+2)
        theta = np.einsum('komn,imnj->ikoj',gate,theta)
        u,s,v = nph.trunc_svd_before_index(theta,2,xi=self.xi,cutoff=self.cutoff)
        u = np.einsum('k,k...->k...',1/self.get_schmidt_values(i,'l'),u)
        v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(i+1,'r'))
        self[i] = u
        self.set_schmidt_values(i,'r',s)
        self[i+1] = v

    def sample_range(self,i,j,n_samples):
        if i < 0 or i > self.L:
            raise ValueError("i must be in range [0,self.L)")
        if j < i or j > self.L:
            raise ValueError("j must be in range [i,self.L)")
        BMPS = self.get_A_B_config(i)[i:j+1]
        con_start = np.diag(BMPS[0])
        BMPS = BMPS[1:]
        samples = np.zeros((n_samples,len(BMPS)),dtype="int")
        for m in range(n_samples):
            r = np.zeros(len(BMPS),dtype="int")
            contracted_tensor = con_start
            p_total = 1
            for n in range(0,j-i):
                contracted_tensor = np.einsum('ik,k...->i...',contracted_tensor,BMPS[n])
                density_matrix = np.einsum('ijk,iqk->jq',contracted_tensor,np.conj(contracted_tensor))/p_total
                p = np.diagonal(density_matrix)
                p = np.real_if_close(p,tol=10**4)
                r[n] = np.random.choice([0,1],p=p)
                p_total *= p[r[n]]
                contracted_tensor = contracted_tensor[:,r[n],:]
            samples[m,:] = r
        return samples
    
    def sample(self,n_samples):
        return self.sample_range(0,self.L,n_samples)
    
    def apply_mpo_zip_up(self,mpo,i):
        tim1 = tim.Tim()
        if i < 0 or i > self.L-mpo.L:
            raise ValueError("i must be in range [0,self.L-mpo.L)")
        k1 = len(self.get_schmidt_values(i,'l'))
        C = np.identity(k1).reshape((k1,1,k1))
        Bmps = self.get_all_A(i,i+mpo.L)
        Bmpo = mpo.get_all_A(0,mpo.L)
        tim1.print_since_last("all get A")
        for j in range(mpo.L):
            C = np.einsum('ijk,klm->ijlm',C,Bmps[j])
            C = np.einsum('jqlp,ijlm->iqpm',Bmpo[j],C)
            u,s,v = nph.trunc_svd_before_index(C,2,self.xi,self.cutoff)
            self[i+j]=np.einsum('k,k...->k...',1/self.get_schmidt_values(i+j,'l'),u)
            self.set_schmidt_values(i+j,'r',s)
            C = np.einsum('k,k...->k...',s,v)
        if i < self.L-1:
            C = np.einsum('k,k...->k...',1/self.get_schmidt_values(i,'r'),C)
            C = C.reshape((C.shape[0],C.shape[1]))
            self[i+1] = np.einsum('ik,klm->ilm',C,self[i+1])
        else:
            s = self.get_schmidt_values(i,'l')
            self.set_schmidt_values(i,'l',np.einsum('k,k->k',s,C.reshape(-1)))
        
        tim1.print_since_last("time for first sweep")
        #self.plot_bond_dims("bond_dims_after_first_sweep"+str(time.time()))
        print("Maximum bond dimension after first sweep: "+str(self.maximum_bond_dim()))
        self.into_canonical_form()
        #self.plot_bond_dims("bond_dims_after_second_sweep"+str(time.time()))
        print("Maximum bond dimension after second sweep: "+str(self.maximum_bond_dim()))
        tim1.print_since_last("time to compress into canonicla form")

    def apply_mpo_zip_up_2(self,mpo,i):
        if i < 0 or i > self.L-mpo.L:
            raise ValueError("i must be in range [0,self.L-len(alist))")
        end_index = i+mpo.L-1
        send = self.get_schmidt_values(end_index,'r')
        k1 = len(send)
        Amps1 = self.get_all_A(i,i+mpo.L)
        Amps1[mpo.L-1] = np.einsum('...k,k->...k',Amps1[mpo.L-1],send)
        Ampo2 = mpo.get_all_A(0,mpo.L)
        C = np.identity(k1).reshape((1,k1,k1))
        for j in range(mpo.L-1,-1,-1):
            C = np.einsum('knm,pmi->knpi',Amps1[j],C)
            C = np.einsum('jqnp,knpi->jkqi',Ampo2[j],C)
            u,s,v = nph.trunc_svd_before_index(C,2,self.xi,cutoff=self.cutoff,norm=2**((self.L)/2))
            v  = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(i+j,'r'))
            self[i+j]=v
            self.set_schmidt_values(i+j,'l',s)
            C = np.einsum('...k,k->...k',u,s)
        if i > 0:
            C = np.einsum('...k,k->...k',C,1/self.get_schmidt_values(i,'l'))
            C = C.reshape(C.shape[1:])
            self[i-1] = np.einsum('knm,mi->kni',self[i-1],C)
        else:
            s = self.get_schmidt_values(i,'l')
            self.set_schmidt_values(i,'l',np.einsum('k,k->k',s,C.reshape(-1)))
        self.into_canonical_form(up_down='down')

    def apply_mpo_regularily(self,mpo,i):
        if i < 0 or i > self.L-mpo.L:
            raise ValueError("i must be in range [0,self.L-1-mpo.L)")
        self.set_schmidt_values(i,'l',np.tensordot(mpo.get_schmidt_values(0,'l'),self.get_schmidt_values(i,'l'),axes=0).reshape(-1))
        for j in range(mpo.L):
            nten = np.einsum('mijn,kjl->mkinl',mpo[j],self[i+j])
            shape = nten.shape
            nten = np.reshape(nten,(shape[0]*shape[1],shape[2],shape[3]*shape[4]))
            self[i+j] = nten
            self.set_schmidt_values(i+j,'r',np.tensordot(mpo.get_schmidt_values(j,'r'),self.get_schmidt_values(i+j,'r'),axes=0).reshape(-1))
        Amps = self.get_all_A(i,i+mpo.L)
        for j in range(mpo.L-1):
            theta = np.einsum('kil,ljm->kijm',Amps[j],Amps[j+1])
            q,r = nph.qr_before_index(theta,2)
            Amps[j]=q
            Amps[j+1]=r
        for j in range(mpo.L,1,-1):
            contracted_tensor = np.einsum('ijk,klm->ijlm',Amps[j-2],Amps[j-1])
            contracted_tensor = np.einsum('ijlm,m->ijlm',contracted_tensor,self.get_schmidt_values(j+i-1,'r'))
            u,s,v = nph.trunc_svd_before_index(contracted_tensor,2,xi=self.xi,cutoff=self.cutoff)
            v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(j+i-1,'r'))
            self.set_schmidt_values(j+i-1,'l',s)
            self[j+i-1] = v
            Amps[j-2] = u
        self[i] = np.einsum('k,k...->k...',1/self.get_schmidt_values(i,'l'),Amps[0])

    def collapse_subspace(self,i,j):
        if i < 0 or i > self.L-1:
            raise ValueError("i must be in range [0,self.L)")
        if j <= i or j > self.L:
            raise ValueError("j must be in range [i+1,self.L]")
        if i==0 and j==self.L:
            raise ValueError("sample the whole MPS instead of collapsing")
        samples = self.sample_range(i,j,1)[0,:]
        As = self.get_all_A(i,j)
        As[-1] = np.einsum('...k,k->...k',As[-1],self.get_schmidt_values(j,'l'))
        contracted_ten = As[0][:,samples[0],:]
        for l in range(1,j-i):
            print(l,"l")
            print(contracted_ten.shape,As[l][:,samples[l]:].shape)
            contracted_ten = np.tensordot(contracted_ten,As[l][:,samples[l],:],axes=([-1],[0]))
            
        
        tensors = [copy.deepcopy(self.get_schmidt_values(0,'l'))]
        if 0<i and j<self.L:
            for m in range(0,i-1):
                tensors.append(copy.deepcopy(self[m]))
                tensors.append(copy.deepcopy(self.get_schmidt_values(m,'r')))
            
            middle = np.tensordot(np.einsum('k,k...->k...',self.get_schmidt_values(i-1,'l'),self[i-1]),contracted_ten,axes=([-1],[0]))
            middle = np.tensordot(middle,self[j],axes=([-1],[0]))
            u,s,v = nph.trunc_svd_before_index(middle,2,xi=self.xi,cutoff=self.cutoff)
            u = np.einsum('k...,k->k...',1/self.get_schmidt_values(i-1,'l'),u)
            tensors.append(u)
            tensors.append(s)
            v = np.einsum('...k,k->...k',v,1/self.get_schmidt_values(j,'r'))
            tensors.append(v)
            for m in range(j,self.L):
                tensors.append(copy.deepcopy(self[m]))
                tensors.append(copy.deepcopy(self.get_schmidt_values(m,'r')))
        elif i==0:
            middle = np.tensordot(contracted_ten,self[j],axes=([-1],[0]))
            tensors.append(middle)
            tensors.append(copy.deepcopy(self.get_schmidt_values(j,'r')))
            for m in range(j+1,self.L):
                tensors.append(copy.deepcopy(self[m]))
                tensors.append(copy.deepcopy(self.get_schmidt_values(m,'r')))
        elif j==self.L:
            for m in range(0,i-1):
                tensors.append(copy.deepcopy(self[m]))
                tensors.append(copy.deepcopy(self.get_schmidt_values(m,'r')))
            middle = np.tensordot(self[i-1],contracted_ten,axes=([-1],[0]))
            tensors.append(middle)
            tensors.append(copy.deepcopy(self.get_schmidt_values(self.L-1,'r')))
        
        new =  SuperMPS.create_SuperMPS_from_tensor_array(tensors,self.xi,self.cutoff) 
        new.__class__ = MPS 
        new.into_canonical_form('down')
        return new


    def measure_subspace(self,i,j):
        if i < 0 or i > self.L-1:
            raise ValueError("i must be in range [0,self.L)")
        if j <= i or j > self.L:
            raise ValueError("j must be in range [i+1,self.L+1]")
        contracted_tensor = self.get_contracted_tensor(i,j)
        s = contracted_tensor.shape
        contracted_tensor = np.reshape(contracted_tensor,(s[0],int(np.prod(s[1:-1])),s[-1]))
        r = np.einsum('ijk,ijk->j',contracted_tensor,np.conj(contracted_tensor))
        r = np.real_if_close(r,tol=10**4)
        return r