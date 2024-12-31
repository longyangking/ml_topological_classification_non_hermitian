import numpy as np
from .topological_model import NonHermitianHamiltonian, NonHermitianTopologicalModel, Supported_Gap_Types
from .check_insulator_utils import is_insulator_real_line, is_insulator_imaginary_line, is_insulator_point
import copy

class Generator:
    def __init__(self, hamiltonian, gap_type, verbose=False):
        self.gap_type = gap_type
        if gap_type not in Supported_Gap_Types:
            raise Exception("Error: Unsupported gap type -> {0}".format(gap_type))
        self.hamiltonian = hamiltonian
        self.verbose = verbose

    def get_n(self):
        return self.hamiltonian.get_n()

    def generate_models(self, n_sample, check_insulator=False, n_cores=1):
        models = list()
        vs = np.zeros(n_sample)

        if check_insulator:
            if self.gap_type == 'point':
                check_func = is_insulator_point
            elif self.gap_type == 'real line':
                check_func = is_insulator_real_line
            elif self.gap_type == 'imaginary line':
                check_func = is_insulator_imaginary_line

        if n_cores == 1:
            for i in range(n_sample):
                model = NonHermitianTopologicalModel(self.hamiltonian, gap_type=self.gap_type, verbose=self.verbose)
                model.initiate()
                if check_insulator:
                    flag = check_func(model)
                    while not flag:
                        model.initiate()
                        flag = check_func(model)
                models.append(copy.deepcopy(model))
                vs[i] = model.hamiltonian.get_topological_invariant()
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return models, vs

    def generate(self, n_sample, check_insulator=False, n_cores=1):
        Qs, vs = list(), np.zeros(n_sample)

        if check_insulator:
            if self.gap_type == 'point':
                check_func = is_insulator_point
            elif self.gap_type == 'real line':
                check_func = is_insulator_real_line
            elif self.gap_type == 'imaginary line':
                check_func = is_insulator_imaginary_line

        if n_cores == 1:
            for i in range(n_sample):
                model = NonHermitianTopologicalModel(self.hamiltonian, gap_type=self.gap_type, verbose=self.verbose)
                model.initiate()
                if check_insulator:
                    flag = check_func(model)
                    while not flag:
                        model.initiate()
                        flag = check_func(model)
                    
                kpoints = model.get_kpoints()
                Qs.append(model.calculate_Q(kpoints=kpoints))
                vs[i] = model.hamiltonian.get_topological_invariant()
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return Qs, vs

            