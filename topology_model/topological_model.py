import numpy as np
import numpy.linalg as LA
import scipy.linalg as sLA
from abc import ABC, abstractmethod

Supported_Gap_Types = ['point', 'real line', 'imaginary line']

sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
sigma_0 = np.array([[1,0],[0,1]])

import numba as nb
from numba.typed import List

@nb.jit()
def get_P(P, allRvectors, allLvectors, n):
    for ik in range(len(allRvectors)):  # for the each k point
        for i in range(n):
            for j in range(n):
                for ivec in range(len(allRvectors[ik])):  # for the each band
                        P[ik, i, j] += np.conjugate(allLvectors[ik][ivec][j])*allRvectors[ik][ivec][i]

class NonHermitianTopologicalModel:
    def __init__(self, hamiltonian, gap_type="point", verbose=False):
        self.hamiltonian = hamiltonian

        if gap_type not in Supported_Gap_Types:
            raise Exception("Unsupported gap type")
        self.gap_type = gap_type

        self.n_dim = hamiltonian.get_n_dim()
        self.n = hamiltonian.get_n()
        if gap_type == "point":
            self.n = 2*self.n

        self.verbose = verbose

    def get_n_dim(self):
        return self.n_dim

    def get_n(self):
        return self.n
    
    def get_Hamiltonian(self, k):
        return self.hamiltonian.get_Hamiltonian(k)
    
    def initiate(self):
        self.hamiltonian.initiate()
   
    def is_real_line_gap(self):
        return self.hamiltonian.is_real_line_gap()
    
    def is_imaginary_line_gap(self):
        return self.hamiltonian.is_imaginary_line_gap()
    
    def get_parameters(self):
        return self.hamiltonian.get_parameters()
    
    def set_parameters(self, **kwargs):
        self.hamiltonian.set_parameters(**kwargs)

    def get_kpoints(self):
        return self.hamiltonian.get_kpoints()
    
    def get_topological_invariant(self):
        return self.hamiltonian.get_topological_invariant()

    def calculate_Q(self, kpoints, perturbation=None, E_ref=0.):
        P = self.calculate_projection_operator(kpoints=kpoints, perturbation=perturbation, E_ref=E_ref)
        n_P = len(P)
        Q = np.zeros((n_P, self.n, self.n), dtype=complex)
        for n in range(n_P):
            Q[n] = np.identity(self.n) - 2*P[n]

        return Q
    
    def get_flatten_Hamiltoian(self, kpoints, perturbation=None, E_ref=0.):
        if self.n_dim == 0:
            values, Rvectors, Lvectors = self.get_eigensystem(0, perturbation=perturbation)

            if self.gap_type == "point":
                flatten_Hamiltonian = np.sum([np.sign(val-E_ref)*(1/(Rvectors[iv].dot(np.conjugate(Lvectors[iv]))))*np.tensordot(Rvectors[iv], np.conjugate(Lvectors[iv]), axes=0) \
                                              for iv,val in enumerate(values)], axis=0)

            elif self.gap_type == "real line":
                flatten_Hamiltonian = np.sum([np.sign(np.real(val-E_ref))*(1/(Rvectors[iv].dot(np.conjugate(Lvectors[iv]))))*np.tensordot(Rvectors[iv], np.conjugate(Lvectors[iv]), axes=0) \
                                              for iv,val in enumerate(values)], axis=0)

            elif self.gap_type == "imaginary line":
                flatten_Hamiltonian = np.sum([np.sign(np.imag(val-E_ref))*(1/(Rvectors[iv].dot(np.conjugate(Lvectors[iv]))))*np.tensordot(Rvectors[iv], np.conjugate(Lvectors[iv]), axes=0) \
                                              for iv,val in enumerate(values)], axis=0)
                
            return np.array([flatten_Hamiltonian],dtype=complex)

        flatten_Hamiltonian_list = np.zeros((len(kpoints), self.n, self.n), dtype=complex)

        for ik, k in enumerate(kpoints):
            values, Rvectors, Lvectors = self.get_eigensystem(k, perturbation=perturbation)
            if self.gap_type == "point":
                flatten_Hamiltonian = np.sum([np.sign(val-E_ref)*(1/(Rvectors[iv].dot(np.conjugate(Lvectors[iv]))))*np.tensordot(Rvectors[iv], np.conjugate(Lvectors[iv]), axes=0) \
                                              for iv,val in enumerate(values)], axis=0)

            elif self.gap_type == "real line":
                flatten_Hamiltonian = np.sum([np.sign(np.real(val-E_ref))*(1/(Rvectors[iv].dot(np.conjugate(Lvectors[iv]))))*np.tensordot(Rvectors[iv], np.conjugate(Lvectors[iv]), axes=0) \
                                              for iv,val in enumerate(values)], axis=0)

            elif self.gap_type == "imaginary line":
                flatten_Hamiltonian = np.sum([np.sign(np.imag(val-E_ref))*(1/(Rvectors[iv].dot(np.conjugate(Lvectors[iv]))))*np.tensordot(Rvectors[iv], np.conjugate(Lvectors[iv]), axes=0) \
                                              for iv,val in enumerate(values)], axis=0)

            flatten_Hamiltonian_list[ik] = flatten_Hamiltonian

        return flatten_Hamiltonian_list


    def calculate_projection_operator(self, kpoints, perturbation=None, E_ref=0.0):
        # calculate the projection operator
        if self.n_dim == 0:
            values, Rvectors, Lvectors = self.get_eigensystem(0, perturbation=perturbation)

            if self.gap_type == "point":
                indices = np.where(values<E_ref)[0]
            elif self.gap_type == "real line":
                indices = np.where(np.real(values)<E_ref)[0]
            elif self.gap_type == "imaginary line":
                indices = np.where(np.imag(values)<E_ref)[0]
            #Rvectors = Rvectors[indices]
            #Lvectors = Lvectors[indices]

            P = np.zeros((self.n, self.n), dtype=complex)
            for i in range(self.n):
                for j in range(self.n):
                    for index in indices:
                        # P[i,j] += np.conjugate(Lvectors[index][j])*Rvectors[index][i]
                        f = Rvectors[indices][index]
                        fc = Lvectors[indices][index]
                        P += (1/(f.dot(np.conjugate(fc))))*np.tensordot(f, np.conjugate(fc), axes=0)

            return [P]
        
        allRvectors = list()
        allLvectors = list()

        #kpoints = self.get_kpoints()
        for k in kpoints:
            values, Rvectors, Lvectors = self.get_eigensystem(k, perturbation=perturbation)
            
            if self.gap_type == "point":
                indices = np.where(values<E_ref)[0]
            elif self.gap_type == "real line":
                indices = np.where(np.real(values)<E_ref)[0]
            elif self.gap_type == "imaginary line":
                indices = np.where(np.imag(values)<E_ref)[0]

            #vectors = vectors[index]
            allRvectors.append(Rvectors[indices])
            allLvectors.append(Lvectors[indices])

        P = np.zeros((len(allRvectors), self.n, self.n), dtype=complex)

        # typed_allRvectors = List()
        # [typed_allRvectors.append(x) for x in allRvectors]
        # typed_allLvectors = List()
        # [typed_allLvectors.append(x) for x in allLvectors]

        # get_P(P, typed_allRvectors, typed_allLvectors, self.n)
        for ik in range(len(allRvectors)):  # for the each k point
                for ivec in range(len(allRvectors[ik])):  # for the each band
                    Rvectors = allRvectors[ik]
                    Lvectors = allLvectors[ik]
                    P[ik] += (1/(np.conjugate(Lvectors[ivec]).dot(Rvectors[ivec])))*np.tensordot(Rvectors[ivec], np.conjugate(Lvectors[ivec]), axes=0)
                    #P[ik, i, j] += np.conjugate(allLvectors[ik][ivec][j])*allRvectors[ik][ivec][i]
        return P

    def get_eigensystem(self, k, perturbation=None):
        hk = self.hamiltonian.get_Hamiltonian(k)

        #print(gap_type)
        hp = 0
        if perturbation is not None:
            hp = perturbation.get_Hamiltonian(k)
        
        if self.gap_type == "point":
            # introduce a chiral symmetry
            hkc = np.transpose(np.conjugate(hk))
            hk = np.block([[np.zeros(hk.shape), hk], [hkc, np.zeros(hk.shape)]])
            if perturbation is not None:
                hpc = np.transpose(np.conjugate(hp))
                hp = np.block([[np.zeros(hp.shape), hp], [hpc, np.zeros(hp.shape)]])
                hk = hk + hp

            if np.allclose(hk, np.transpose(np.conjugate(hk))): # Hermitian matrix
                values, vectors = LA.eigh(hk)
            else:
                raise Exception("Resulted Hamiltonian after Hermitian flattening is Non-Hermitian !")
            
            vectors = np.transpose(vectors)
            index = np.argsort(values)
            values = values[index]
            Rvectors = vectors[index]
            Lvectors = vectors[index]

            return values, Rvectors, Lvectors

        elif (self.gap_type == "real line") or (self.gap_type == 'imaginary line'):
            if perturbation is not None:
                hk = hk + hp

            values, Lvectors, Rvectors = sLA.eig(hk, left=True, right=True)
            Rvectors = np.transpose(Rvectors)
            Lvectors = np.transpose(Lvectors)

            for i in range(len(values)):
                _v = np.conjugate(Lvectors[i]).dot(Rvectors[i])
                Lvectors[i] = np.conjugate(1/np.sqrt(_v))*Lvectors[i]
                Rvectors[i] = 1/np.sqrt(_v)*Rvectors[i]

            # values, Rvectors = LA.eig(hk)
            # _, Lvectors = LA.eig(np.conjugate(np.transpose(hk)))

            # Rvectors = np.transpose(Rvectors)
            # Lvectors = np.transpose(Lvectors)

            # Mmat = np.zeros((len(values), len(values)),dtype=complex)
            # for i in range(len(values)):
            #     for j in range(len(values)):
            #         Mmat[i,j] = np.conjugate(Lvectors[i]).dot(Rvectors[j])
            # Lvectors = np.conjugate(np.transpose(LA.inv(Mmat))).dot(Lvectors)

            # # normalize
            # Umat = np.zeros((len(values), len(values)),dtype=complex)
            # for i in range(len(values)):
            #     for j in range(len(values)):
            #         Umat[i,j] = np.conjugate(Lvectors[i]).dot(Rvectors[j])
            # Lvectors = np.conjugate(np.transpose(LA.inv(Umat))).dot(Lvectors)

            # check_biorthogonal = 0
            # for i in range(len(values)):
            #     check_biorthogonal += np.real(np.conjugate(Lvectors[i]).dot(Rvectors[i]))

            # if np.round(check_biorthogonal) != len(values):
            #     raise Exception("Error in calculating the eigenvectors, not full biorthogonal [{0}] at [k={1}]".format(
            #         check_biorthogonal/len(values), k
            #         ))

            return values, Rvectors, Lvectors
            
        else:
            raise Exception("Unsupported gap type! : {0}".format(self.gap_type))
    
class NonHermitianHamiltonian(ABC):

    def __init__(self, E_ref=0.):
        self.E_ref = E_ref

    def get_E_ref(self):
        return self.E_ref

    @abstractmethod
    def get_n(self):
        pass

    @abstractmethod
    def get_n_dim(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs):
        pass

    @abstractmethod
    def initiate(self):
        pass

    @abstractmethod
    def get_kpoints(self):
        pass

    def is_real_line_gap(self):
        return None

    def is_imaginary_line_gap(self):
        return None
    
    def get_topological_invariant(self):
        return 0

    @abstractmethod
    def get_Hamiltonian(self, k):
        pass

class NonHermitianHamiltonianOBC(NonHermitianHamiltonian):
    '''
    with OBC
    '''
    def __init__(self, E_ref=0.):
        super().__init__(E_ref=E_ref)
        #self.gs = gs # how many sites along the different lattice constant vectors

    def get_n_dim(self):
        return 0
    
    # def get_gs(self):
    #     return self.gs