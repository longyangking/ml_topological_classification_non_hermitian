import numpy as np
import scipy.linalg as LA
import itertools
import random
from .hermitian_generator import GenerateHermitianHamiltonian
from .non_hermitian_symmetry_class import non_Hermitian_symmetry_list_Hermitian_counterparts, non_hermitian_symmetry_list
from .non_hermitian_symmetry_class import is_block_diagonalization_line_gap, is_complexification_line_gap
from .non_hermitian_symmetry_class import is_block_diagonalization_point_gap, is_complexification_point_gap

gap_types = [
    'real line', 'imaginary line', 'point'
]

class DeformHamiltonianGenerator:
    def __init__(self, n, n_dim, non_Hermitian_symmetry_class, gap_type='real line', is_inverse_parity=False, verbose=False):
        self.n = n
        self.n_dim = n_dim
        self.non_Hermitian_symmetry_class = non_Hermitian_symmetry_class
        self.gap_type = gap_type
        self.is_inverse_parity = is_inverse_parity

        if non_Hermitian_symmetry_class not in non_hermitian_symmetry_list:
            raise Exception("Error: Unknown non-Hermitian symmetry class")

        if gap_type not in gap_types:
            raise Exception("Error: Unknown gap type for non-Hermitian Hamiltonian")

        self.verbose = verbose
        self.is_block = False
        self.is_complexification = False
        self.Hermitian_symmetry_class = None
        self.generator = None

        self.initiate()

    def initiate(self):
        is_block, is_complexification = False, False
        if self.gap_type == 'real line':
            symmetry_class = non_Hermitian_symmetry_list_Hermitian_counterparts[self.non_Hermitian_symmetry_class][0]
            is_block, _ = is_block_diagonalization_line_gap(self.non_Hermitian_symmetry_class)
            is_complexification, _ = is_complexification_line_gap(self.non_Hermitian_symmetry_class)
        elif self.gap_type == 'imaginary line':
            symmetry_class = non_Hermitian_symmetry_list_Hermitian_counterparts[self.non_Hermitian_symmetry_class][1]
            _, is_block = is_block_diagonalization_line_gap(self.non_Hermitian_symmetry_class)
            _, is_complexification = is_complexification_line_gap(self.non_Hermitian_symmetry_class)
        else: # point gap
            symmetry_class = non_Hermitian_symmetry_list_Hermitian_counterparts[self.non_Hermitian_symmetry_class][2]
            is_block = is_block_diagonalization_point_gap(self.non_Hermitian_symmetry_class)
            is_complexification = is_complexification_point_gap(self.non_Hermitian_symmetry_class)

        if self.verbose:
            print("Is double block? ", is_block)
            print("Is complexification? ", is_complexification)

        self.is_block = is_block
        self.is_complexification = is_complexification
        self.Hermitian_symmetry_class = symmetry_class

        is_double = is_block or is_complexification
        n_hermitian = self.n 
        if is_double:
            n_hermitian = int(self.n/2)

        if self.verbose:
            print("Hermitian Generator: n = {0}, sym = {1}, n_dim={2}, is_parity={3}".format(
                n_hermitian, symmetry_class, self.n_dim, self.is_inverse_parity
            ))

        self.generator = GenerateHermitianHamiltonian(
            n=n_hermitian, 
            sym=symmetry_class, 
            n_dim=self.n_dim,
            is_inverse_parity=self.is_inverse_parity,
            verbose=self.verbose
            )

    def generate(self, n_sample, amplitude=1.0, n_cores=1):
        hamiltonians = list()

        if n_cores == 1:
            h1 = self.generator.get_random_hamiltonian_lower_dimension_samples(n_sample=n_sample)
            if self.is_block:
                charge_range_1 = self.generator.get_charge_range()
                charge_range_2 = self.generator.get_charge_range()

                if (charge_range_1 is not None) and (charge_range_2 is not None):
                    # this part is for optimization, otherwise too long computation time
                    charge_list = 2*list(itertools.product(charge_range_1, charge_range_2))
                    charge_list_random = random.choices(charge_list, k=n_sample-len(charge_list))
                    charge_list_np = np.array(charge_list + charge_list_random)

                    if n_sample > len(charge_list_np):
                        raise Exception("Error: too few samples!")

                    charge_list_1 = charge_list_np[:,0]
                    h1 = self.generator.get_random_hamiltonian_lower_dimension_samples(n_sample=n_sample, charges=charge_list_1)
                
                    charge_list_2 = charge_list_np[:,1]
                    h2 = self.generator.get_random_hamiltonian_lower_dimension_samples(n_sample=n_sample, charges=charge_list_2)

                else:
                    h1 = self.generator.get_random_hamiltonian_lower_dimension_samples(n_sample=n_sample)
                    h2 = self.generator.get_random_hamiltonian_lower_dimension_samples(n_sample=n_sample)   
            else:
                h2 = h1

            for hs in zip(h1, h2):
                hamiltonian = DeformHamiltonian(
                    n=self.generator.get_n(), 
                    n_dim=self.n_dim, non_Hermitian_symmetry_class=self.non_Hermitian_symmetry_class, 
                    gap_type=self.gap_type,  is_block=self.is_block, is_complexification=self.is_complexification, 
                    hamiltonians=hs, amplitude=amplitude, verbose=self.verbose)
                hamiltonians.append(hamiltonian)
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return hamiltonians
        
class DeformHamiltonian:
    def __init__(self, n, n_dim, non_Hermitian_symmetry_class, gap_type,  is_block, is_complexification, hamiltonians, amplitude=1.0, verbose=False):
        self.n = n
        self.n_dim = n_dim
        self.non_Hermitian_symmetry_class = non_Hermitian_symmetry_class
        self.gap_type = gap_type

        self.is_block = is_block
        self.is_complexification = is_complexification
        self.hamiltonians = hamiltonians

        self.amplitude = amplitude
        self.verbose = verbose

    def get_n(self):
        if self.is_block or self.is_complexification:
            return 2*self.n
        
        return self.n
    
    def get_n_dim(self):
        return self.n_dim
    
    def get_symmetry_class(self):
        return self.non_Hermitian_symmetry_class
    
    def get_gap_type(self):
        return self.gap_type

    def get_Hamiltonian(self, k):
        if self.is_block:
            h1, h2 = self.hamiltonians[0](k), self.hamiltonians[1](k)
            hk = np.block([[h1, np.zeros(h1.shape)], [np.zeros(h1.shape), h2]])
            hk = self.amplitude*hk # adjust the amplitude, so to be a perturbation
            return hk
        
        if self.is_complexification:
            k = np.array(k)
            h1, h2 = self.hamiltonians[0](k), np.conjugate(self.hamiltonians[1](-k))
            hk = np.block([[h1, np.zeros(h1.shape)], [np.zeros(h1.shape), h2]])
            hk = self.amplitude*hk # adjust the amplitude, so to be a perturbation
            return hk
        
        hk = self.hamiltonians[0](k)
        hk = self.amplitude*hk # adjust the amplitude, so to be a perturbation
        return hk
    
class PerturbationGenerator(DeformHamiltonianGenerator):
    def generate(self, n_sample, amplitude=0.05, n_cores=1):
        return super().generate(n_sample, amplitude, n_cores)
    
    def generate_random(self, n_sample, amplitude=0.05, n_cores=1):
        hamiltonians = list()

        if n_cores == 1:
            h1 = self.generator.get_random_hamiltonian_samples(n_sample=n_sample)
            if self.is_block:
                h2 = self.generator.get_random_hamiltonian_samples(n_sample=n_sample)
            else:
                h2 = h1

            for hs in zip(h1, h2):
                hamiltonian = DeformHamiltonian(
                    n=self.n, 
                    n_dim=self.n_dim, non_Hermitian_symmetry_class=self.non_Hermitian_symmetry_class, 
                    gap_type=self.gap_type,  is_block=self.is_block, is_complexification=self.is_complexification, 
                    hamiltonians=hs, amplitude=amplitude)
                hamiltonians.append(hamiltonian)
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return hamiltonians


# non_Hermitian_symmetry_list_band_Hermitian_counterparts = {
#     # Complex AZ class
#     "A":    ['A', 'A'], 
#     "AIII": ['AIII', 'A'],
#     # Real AZ class
#     "AI":   ['AI', 'D'],
#     "BDI":  ['BDI', 'D'],
#     "D":    ['D', 'D'],
#     "DIII": ['DIII', 'A'],
#     "AII":  ['AII', 'C'], 
#     "CII":  ['CII', 'C'],
#     "C":    ['C', 'C'],
#     "CI":   ['CI', 'A'],
#     "AI+":  ['AI', 'AI'],
#     "BDI+": ['BDI', 'AI'],
#     # "D+":   ['D', 'AI'],
#     "DIII+": ['DIII', 'A'],
#     "AII+": ['AII', 'AII'],
#     "CII+": ['CII', 'AII'],
#     # "C+":   ['C', 'AII'],
#     "CI+":  ['CI', 'A'],
#     # Complex AZ class with sublattice symmetry
#     "A:S":      ['AIII', 'AIII'], 
#     "AIII:S+":  ['AIII', 'AIII'],
#     "AIII:S-":  ['A', 'A'],
#     # Real AZ class with sublattice symmetry
#     "BDI:S++":  ['BDI', 'BDI'],
#     "DIII:S--":  ['DIII', 'AIII'],
#     "CII:S++":  ['CII', 'CII'], 
#     "CI:S--":   ['CI', 'AIII'],
#     "AI:S-":    ['CI', 'DIII'],
#     "BDI:S-+":  ['AI', 'D'],
#     "D:S+":     ['BDI', 'BDI'],
#     "CII:S-+":  ['AII', 'C'],
#     "C:S+":     ['CII', 'CII'],
#     "DIII:S++": ['AIII', 'AIII'],
#     "CI:S++":   ['AIII', 'AIII'],
#     "AI:S+":    ['BDI', 'BDI'],
#     "BDI:S+-":  ['D', 'D'],
#     "D:S-":     ['DIII', 'DIII'],
#     "DIII:S+-": ['AII', 'AII'],
#     "AII:S+":   ['CII', 'CII'],
#     "CII:S+-":  ['C', 'C'],
#     "C:S-":     ['CI', 'CI'],
#     "CI:S+-":   ['AI', 'AI'],
# }

# non_hermitian_symmetry_list = list(non_Hermitian_symmetry_list_band_Hermitian_counterparts.keys())


# class NonHermitianHamiltonianGenerator:
#     def __init__(self, n, n_dim, non_Hermitian_symmetry_class, verbose=False):
#         self.n = n # number of bands
#         self.non_Hermitian_symmetry_class = non_Hermitian_symmetry_class # name of symmetry classes
#         if non_Hermitian_symmetry_class not in non_hermitian_symmetry_list:
#             raise Exception("Error: Unsuppored non-Hermitian symmetry classes")

#         self.n_dim = n_dim # dimension
#         self.verbose = verbose

#     def get_n(self):
#         return self.n
    
#     def get_symmetry_class(self):
#         return self.non_Hermitian_symmetry_class
    
#     def get_n_dim(self):
#         return self.n_dim

#     def generate_Hamiltonian_parts(self, n_sample):
#         sym1, sym2 = non_Hermitian_symmetry_list_band_Hermitian_counterparts[self.non_Hermitian_symmetry_class]
#         is_complex_hr, is_complex_hi = is_complexification(self.non_Hermitian_symmetry_class)
#         is_block_hr, is_block_hi = is_block_diagonalization(self.non_Hermitian_symmetry_class)

#         n_r, n_i = 0, 0
#         if is_block_hr:
#             generator1 = GenerateHermitianHamiltonian(n=int(self.n/2), sym=sym1, n_dim=self.n_dim)
#             hr1_list = generator1.get_random_hamiltonian_lower_dimension_samples(n_sample)
#             hr2_list = generator1.get_random_hamiltonian_lower_dimension_samples(n_sample)
#         elif is_complex_hr:
#             generator1 = GenerateHermitianHamiltonian(n=int(self.n/2), sym=sym1, n_dim=self.n_dim)
#             hr1_list = generator1.get_random_hamiltonian_lower_dimension_samples(n_sample)
#             hr2_list = hr1_list
#         else:
#             generator1 = GenerateHermitianHamiltonian(n=self.n, sym=sym1, n_dim=self.n_dim)
#             hr1_list = generator1.get_random_hamiltonian_lower_dimension_samples(n_sample)
#             hr2_list = hr1_list
#         n_r = generator1.get_n()
#         hr_list = list(zip(hr1_list, hr2_list))

#         if is_block_hi:
#             n_new = int(self.n/2)
#             generator2 = GenerateHermitianHamiltonian(n=n_new, sym=sym2, n_dim=self.n_dim)
#             print(generator2.get_n())
#             while generator2.get_n() != int(n_r/2):
#                 #print(generator2.get_n(), n_r)
#                 n_new = int(n_new/2)
#                 print(n_new)
#                 generator2 = GenerateHermitianHamiltonian(n=n_new, sym=sym2, n_dim=self.n_dim)

#             hi1_list = generator2.get_random_hamiltonian_lower_dimension_samples(n_sample)
#             hi2_list = generator2.get_random_hamiltonian_lower_dimension_samples(n_sample)

#         elif is_complex_hi:
#             generator2 = GenerateHermitianHamiltonian(n=int(self.n/2), sym=sym2, n_dim=self.n_dim)
#             hi1_list = generator2.get_random_hamiltonian_lower_dimension_samples(n_sample)
#             hi2_list = hi1_list
#         else:
#             generator2 = GenerateHermitianHamiltonian(n=self.n, sym=sym2, n_dim=self.n_dim)
#             hi1_list = generator2.get_random_hamiltonian_lower_dimension_samples(n_sample)
#             hi2_list = hi1_list

#         n_i = generator2.get_n()
#         hi_list = list(zip(hi1_list, hi2_list))

#         print(n_r, n_i)
#         hamiltonian_parts_list = list(zip(hr_list, hi_list))
#         return hamiltonian_parts_list

#     def generate_Hamiltonian_samples(self, n_sample):
#         hamiltonian_parts_list = self.generate_Hamiltonian_parts(n_sample)
#         hamiltonians = [NonHermitianHamiltonian(
#                         n=self.n, 
#                         n_dim=self.n_dim, 
#                         non_Hermitian_symmetry_class=self.non_Hermitian_symmetry_class,
#                         hamiltonian_parts=hamiltonian_parts
#             ) for hamiltonian_parts in hamiltonian_parts_list]

#         return hamiltonians
      
# class NonHermitianHamiltonian:
#     def __init__(self, n, n_dim, non_Hermitian_symmetry_class, hamiltonian_parts):
#         self.n = n
#         self.n_dim = n_dim
#         self.hamiltonian_parts = hamiltonian_parts # h1, h2 are two lists
#         self.non_Hermitian_symmetry_class = non_Hermitian_symmetry_class

#     def get_n(self):
#         return self.n

#     def __call__(self, k):
#         return self.get_Hamiltonian(k)

#     def get_Hamiltonian(self, k):
#         k = np.array(k)
#         is_complex_hr, is_complex_hi = is_complexification(self.non_Hermitian_symmetry_class)
#         is_block_hr, is_block_hi = is_block_diagonalization(self.non_Hermitian_symmetry_class)
#         hr_list, hi_list = self.hamiltonian_parts

#         # constraint: h1
#         hr1, hr2 = hr_list
#         if is_block_hr:
#             hk1, hk2 = hr1(k), hr2(k)
#             h1 = np.block([[hk1, np.zeros(hk1.shape)], [np.zeros(hk2.shape), hk2]])

#         if is_complex_hr:
#             hk1 = hr1(k)
#             hk1c = np.conjugate(hr1(-k))
#             h1 = np.block([[hk1, np.zeros(hk1.shape)], [np.zeros(hk1.shape), hk1c]]) # TODO not sure for all cases

#         if (not is_block_hr) and (not is_complex_hr):
#             h1 = hr1(k)
        
#         # constraint: h2
#         hi1, hi2 = hi_list
#         if is_block_hi:
#             hk1, hk2 = hi1(k), hi2(k)
#             h2 = np.block([[hk1, np.zeros(hk1.shape)], [np.zeros(hk2.shape), hk2]])

#         if is_complex_hi:
#             hk1 = hi1(k)
#             hk1c = np.conjugate(hi1(-k))
#             h2 = np.block([[hk1, np.zeros(hk1.shape)], [np.zeros(hk1.shape), hk1c]])
        
#         if (not is_block_hi) and (not is_complex_hi):
#             h2 = hi1(k)

#         return h1 + 1j*h2