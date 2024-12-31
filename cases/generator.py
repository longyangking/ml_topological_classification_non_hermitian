import sys 
sys.path.append("..") 

import numpy as np
from topology_table import NonHermitianTopologicalModel

import copy

class Generator:
    def __init__(self, hamiltonian, gap_type, verbose=False):
        self.gap_type = gap_type
        self.hamiltonian = hamiltonian
        self.verbose = verbose

    def get_n(self):
        return self.hamiltonian.get_n()

    def generate_models(self, n_sample, n_cores=1):
        models = list()
        if n_cores == 1:
            for i in range(n_sample):
                model = NonHermitianTopologicalModel(self.hamiltonian, gap_type=self.gap_type, verbose=self.verbose)
                self.hamiltonian.init()
                # check the gap
                models.append(copy.deepcopy(model))
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return models

    def generate(self, n_sample, n_cores=1, kmesh=None, parameters=None):
        if self.verbose:
            kpoints = self.hamiltonian.get_kpoints()
            print("The number of kpoints: ", len(kpoints))

        Qs = list()
        if n_cores == 1:
            for i in range(n_sample):
                if parameters is not None:
                    self.hamiltonian.set_parameter(**(parameters[i]))
                else:
                    self.hamiltonian.init()
                    # while not self.is_insulator(model):
                    #     model.init()

                model = NonHermitianTopologicalModel(self.hamiltonian, gap_type=self.gap_type, verbose=self.verbose)
                Qs.append(model.calculate_Q())
        else:
            if n_cores < 1:
                raise Exception("Parallel with a wrong setting!")
            raise Exception("Parallel is not achieved yet")

        return Qs
            