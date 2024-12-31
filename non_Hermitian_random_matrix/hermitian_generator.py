# Generate Hermitian Hamiltonian base on Hermitian symmetry classes

import numpy as np
from .rmt import gaussian, circular, sym_list, bott_periodic_table, inverse_bott_periodic_table
from .rmt import c as ischiral
from scipy import linalg as LA

class GenerateHamiltonian0D:
    def __init__(self, n, sym):
        self.n = n # the size of the Hamiltonian: n*n
        self.sym = sym

        if sym not in sym_list:
            raise Exception("Unknown the symmetry class")

    def get_random_hamiltonian(self, v=1.0):
        '''
        Return the terms in H_{0}
        '''
        return gaussian(self.n, self.sym, v=v)

    def __chech_Hermitian(self, h):
        return np.allclose(h, np.transpose(np.conjugate(h)))
    
    def get_charge_range(self):
        sym_1D = bott_periodic_table[self.sym]

        if ischiral(self.sym):
            nr = self.n//2
            if sym_1D in ["D", "DIII"]:
                return [-1,1]
            elif sym_1D in ['AIII', 'BDI']:
                return list(range(nr+1))
            elif sym_1D in ['CII']:
                return list(range(nr//2+1))
        else:
            if sym_1D in ["D", "DIII"]:
                return [-1,1]
            elif sym_1D in ['AIII', 'BDI']:
                return list(range(self.n+1))
            elif sym_1D in ['CII']:
                return list(range(self.n//2+1))
            
        return None

    def get_random_hamiltonian_lower_dimension(self, charge=None):
        '''
        Return the terms in H_{0} by 1D reflection coefficient: 1D -> 0D
        ''' 
        sym_1D = bott_periodic_table[self.sym]
        
        #print(self.n)

        if ischiral(self.sym):
            nr = self.n//2
            if charge is None:
                # Topological invariant of the matrix. Should be one of 1, -1 in symmetry
                # classes D and DIII, should be from 0 to n in classes AIII and BDI,
                # and should be from 0 to n / 2 in class CII.
                if sym_1D in ["D", "DIII"]:
                    charge = np.random.choice([-1,1])
                elif sym_1D in ['AIII', 'BDI']:
                    charge = np.random.choice(range(nr+1))
                elif sym_1D in ['CII']:
                    charge = np.random.choice(range(nr//2+1))
            
            r = circular(nr, sym_1D, charge=charge) # 1D reflection coefficient
            #h0 = 1/2*(r + np.transpose(np.conjugate(r)))
            H = np.zeros((self.n, self.n), dtype=complex)
            H[nr:, :nr] = -1j*r
            H[:nr, nr:] = 1j*np.transpose(np.conjugate(r))

            if self.__chech_Hermitian(H):
                return H
            else:
                raise Exception("Error: non-Hermitian Hamiltonian")
        else:          
            if charge is None:
                # Topological invariant of the matrix. Should be one of 1, -1 in symmetry
                # classes D and DIII, should be from 0 to n in classes AIII and BDI,
                # and should be from 0 to n / 2 in class CII.
                if sym_1D in ["D", "DIII"]:
                    charge = np.random.choice([-1,1])
                elif sym_1D in ['AIII', 'BDI']:
                    charge = np.random.choice(range(self.n+1))
                elif sym_1D in ['CII']:
                    charge = np.random.choice(range(self.n//2+1))

            r = circular(self.n, sym_1D, charge=charge) # 1D reflection coefficient
            #print("det: ", np.linalg.det(1/2*(r + np.transpose(np.conjugate(r)))))
            H = 1/2*(r + np.transpose(np.conjugate(r)))
            #H = H/np.abs(np.linalg.det(H))
            
            if self.__chech_Hermitian(r):   # r is a Hermitian matrix
                return r
            # elif self.__chech_Hermitian(H):
            #     return H
            else:
                raise Exception("Error: non-Hermitian Hamiltonian")

class GenerateHermitianHamiltonian:
    def __init__(self, n, sym, n_dim, is_inverse_parity=False, verbose=False):
        self.n = n # the size of the Hamiltonian: n*n
        self.sym = sym
        self.n_dim = n_dim
        self.is_inverse_parity = is_inverse_parity

        if sym not in sym_list:
            raise Exception("Unknown the symmetry class")

        self.n_new = n
        self.verbose = verbose

        if self.verbose:
            print("Hermitian Generator: OD = {0}, ops = {1}".format(
                self.find_0D_dimension_symmetry(),
                self.update_dimension()
            ))

    def get_n(self):
        self.generate() # obtain the new dimension of Hamiltonian after run generate function once
        return self.n_new

    def get_sym_name(self):
        return self.sym
    
    def get_charge_range(self):
        sym_0D = self.find_0D_dimension_symmetry()
        generate0D = GenerateHamiltonian0D(self.n, sym_0D)
        return generate0D.get_charge_range()

    def update_dimension(self):
        sym = self.sym
        ops = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            ops[i] = ischiral(sym)
            
            if not self.is_inverse_parity:
                sym = bott_periodic_table[sym]
            else:
                sym = inverse_bott_periodic_table[sym]

        return ops 

    def find_0D_dimension_symmetry(self):
        sym = self.sym
        for _ in range(self.n_dim):
            if not self.is_inverse_parity:
                sym = inverse_bott_periodic_table[sym]
            else:
                sym = bott_periodic_table[sym]
        return sym

    def generate(self, v=1.0):
        sym_0D = self.find_0D_dimension_symmetry()
        generate0D = GenerateHamiltonian0D(self.n, sym_0D)
        H0 = generate0D.get_random_hamiltonian(v=v)
        ops = self.update_dimension()

        self.n_new = H0.shape[0]    # update the shape of Hamiltonian matrix
        self.n_new = self.n*2**(np.sum(ops==1))

        return [H0, ops] 

    def generate_by_reflection_coefficient(self, charge=None):
        sym_0D = self.find_0D_dimension_symmetry()
        generate0D = GenerateHamiltonian0D(self.n, sym_0D)
        H0 = generate0D.get_random_hamiltonian_lower_dimension(charge=charge)
        ops = self.update_dimension()

        self.n_new = H0.shape[0]    # update the shape of Hamiltonian matrix
        self.n_new = self.n*2**(np.sum(ops==1))

        return [H0, ops]

    def get_random_hamiltonian_samples(self, n_sample, v=1.0, n_cores=1):
        '''
        Generate the Hamiltonian randomly with the symmetry properties

        Reference:
            [1] Generating random correlation matrices based on vines and extended onion method https://www.sciencedirect.com/science/article/pii/S0047259X09000876
        '''
        hamiltonians = list()
    
        if n_cores == 1:
            for i in range(n_sample):
                hamiltonian = HermitianHamiltonian(
                    n=self.n,  n_dim=self.n_dim, 
                    hamiltonian=self.generate(v=v),
                    is_inverse_parity=self.is_inverse_parity
                    )
                hamiltonians.append(hamiltonian)
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return hamiltonians

    def get_random_hamiltonian_lower_dimension_samples(self, n_sample, charges=None, n_cores=1):
        '''
        Generate the Hamiltonian randomly with the symmetry properties

        Reference:
            [1] Generating random correlation matrices based on vines and extended onion method https://www.sciencedirect.com/science/article/pii/S0047259X09000876
        '''
        hamiltonians = list()

        if n_cores == 1:
            for i in range(n_sample):
                charge = None
                if charges is not None:
                    charge = charges[i]

                hamiltonian = HermitianHamiltonian(
                    n=self.n,  n_dim=self.n_dim, 
                    hamiltonian=self.generate_by_reflection_coefficient(charge=charge),
                    is_inverse_parity=self.is_inverse_parity
                    )
                hamiltonians.append(hamiltonian)
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return hamiltonians

class HermitianHamiltonian:
    def __init__(self, n, n_dim, hamiltonian, is_inverse_parity=False, n_nearest_terms=None):
        self.n = n
        self.n_dim = n_dim
        if self.n_dim < 0:
            raise Exception("Wrong dimension: [{0}]".format(self.n_dim))

        self.is_inverse_parity = is_inverse_parity
        self.n_nearest_terms = n_nearest_terms
        if self.n_nearest_terms is None:
            self.n_nearest_terms = self.n_dim*[1]
        
        self.hamiltonian = hamiltonian

    def get_n_dim(self):
        return self.n_dim
    
    def get_generation_information(self):
        '''
        [H0, ops]
        '''
        return self.hamiltonian

    def get_Hamiltonian(self, k):
        if self.n_dim == 0:
            hk = self.hamiltonian[0]
            ops = list()
        else:
            hk = self.hamiltonian[0]
            ops = self.hamiltonian[-1]

        for i in range(self.n_dim):
            k_long_range = self.n_nearest_terms[i]
            d1, d2 = np.cos(k_long_range*k[i]), np.sin(k_long_range*k[i])

            if self.is_inverse_parity:
                d1, d2 = np.cos(k_long_range*k[i]), -np.sin(k_long_range*k[i])

            if not ops[i]:
                n = hk.shape[0]
                #print("C")
                #tau_z = 1j*LA.block_diag(*((n // 2)*[[[1,0],[0,-1]]]))
                #tau_z = LA.block_diag(*((n // 2)*[[[1,0],[0,-1]]]))
                tau_z = np.diag(np.concatenate([np.ones(n//2), -np.ones(n//2)]))
                hk = hk*d1 + tau_z*d2
            else:
                a1 = np.block([[np.zeros(hk.shape), hk], [hk, np.zeros(hk.shape)]])
                a2 = np.block([[np.zeros(hk.shape), -1j*np.eye(hk.shape[0])], [1j*np.eye(hk.shape[0]), np.zeros(hk.shape)]])
                hk = a1*d1 + a2*d2
                #print("dada", a1)

        return hk     

    def __call__(self, k):
        k = np.array(k)
        return self.get_Hamiltonian(k)

if __name__=="__main__":
    from scipy.linalg import ishermitian
    n = 8
    for sym in sym_list:
        generator = GenerateHamiltonian0D(n=n,sym=sym)
        h = generator.get_random_hamiltonian_lower_dimension()
        print(ishermitian(h, atol=5e-11))