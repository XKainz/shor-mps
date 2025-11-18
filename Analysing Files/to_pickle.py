'''to_pickle.py'''

class ToPickle:
    def __init__(self,
                  N,
                  x,
                  len_a,
                  MPS_mpo,
                  MPS_fat,
                  p_success_mpo,
                  p_success_fat,
                  MPO_mpo,
                  max_bond_dim_mpo_mpo,
                  max_bond_dim_mps_mpo,
                  max_bond_dim_mps_fat):
        self.N = N
        self.x = x
        self.len_a = len_a
        self.MPS_mpo = MPS_mpo
        self.MPS_fat = MPS_fat
        self.p_success_mpo = p_success_mpo
        self.p_success_fat = p_success_fat
        self.MPO_mpo = MPO_mpo
        self.max_bond_dim_mpo_mpo = max_bond_dim_mpo_mpo
        self.max_bond_dim_mps_mpo = max_bond_dim_mps_mpo
        self.max_bond_dim_mps_fat = max_bond_dim_mps_fat
