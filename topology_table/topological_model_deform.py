import numpy as np
import numpy.linalg as LA
from abc import ABC, abstractmethod

Supported_Gap_Types = ['point', 'real line', 'imaginary line']

sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
sigma_0 = np.array([[1,0],[0,1]])

import numba as nb
from numba.typed import List

@nb.jit()
def get_P(P, allvectors, n):
    for index in range(len(allvectors)):  # for the each k point
        for i in range(n):
            for j in range(n):
                for vector in allvectors[index]:  # for the each band
                        P[index, i, j] += np.conjugate(vector[j])*vector[i]

class DeformTopologicalModel:
    def __init__(self, hamiltonian, gap_type="point", verbose=False):
        self.hamiltonian = hamiltonian

        if gap_type not in Supported_Gap_Types:
            raise Exception("Unsupported gap type")
        self.gap_type = gap_type

        self.n_dim = hamiltonian.get_n_dim()
        self.n = hamiltonian.get_n()
        self.verbose = verbose

    def get_n_dim(self):
        return self.n_dim

    def get_n(self):
        return self.n
    
    def get_Hamiltonian(self, k):
        return self.hamiltonian.get_Hamiltonian(k)
   
    def get_gap_type(self):
        return self.gap_type
    
    def get_kpoints(self):
        return self.hamiltonian.get_kpoints()

    def calculate_Q(self, kpoints, perturbation=None):
        P = self.calculate_projection_operator(kpoints=kpoints, perturbation=perturbation)
        n_P = len(P)
        Q = np.zeros((n_P, self.n, self.n), dtype=complex)
        for n in range(n_P):
            Q[n] = np.identity(self.n) - 2*P[n]

        return Q

    def calculate_projection_operator(self, kpoints, perturbation=None):
        # calculate the projection operator
        if self.n_dim == 0:
            values, vectors = self.get_eigensystem(0, perturbation=perturbation)
            index = np.where(values<0.0)
            vectors = vectors[index]

            P = np.zeros((self.n, self.n), dtype=complex)
            for i in range(self.n):
                for j in range(self.n):
                    for vector in vectors:
                        P[i,j] += np.conjugate(vector[j])*vector[i]

            return [P]
        
        allvectors = list()

        #kpoints = self.get_kpoints()
        for k in kpoints:
            values, vectors = self.get_eigensystem(k, perturbation=perturbation)
            index = np.where(values<0.0)
            vectors = vectors[index]
            allvectors.append(vectors[index])

        P = np.zeros((len(allvectors), self.n, self.n), dtype=complex)
        typed_allvectors = List()
        [typed_allvectors.append(x) for x in allvectors]
        get_P(P, typed_allvectors, self.n)

        return P

    def get_eigensystem(self, k, perturbation=None):
        hk = self.hamiltonian.get_Hamiltonian(k)

        #print(gap_type)
        hp = 0
        if perturbation is not None:
            hp = perturbation.get_Hamiltonian(k)
        hk = hk + hp

        if np.allclose(hk, np.transpose(np.conjugate(hk))): # Hermitian matrix
            values, vectors = LA.eigh(hk)
            values = np.real(values)
        else:
            raise Exception("Resulted Hamiltonian after Hermitian flattening is Non-Hermitian !")
        
        vectors = np.transpose(vectors)
        
        index = np.argsort(values)
        values = values[index]
        vectors = vectors[index]
        return values, vectors