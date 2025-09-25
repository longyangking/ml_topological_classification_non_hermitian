'''
Topological comparator for non-Hermitian Hamiltonians
'''

import numpy as np
from scipy.optimize import minimize
import scipy as sp
import itertools

def sfunc_smooth_in(Q1, Q2, epsilon):
    n_k = len(Q1)
    s = 1
    a0 = None
    for i in range(n_k):
        #a = np.abs(np.linalg.det(Q1[i]+Q2[i]))
    #     if s > (1 - np.exp(-a**2/epsilon**2)):
    #         s = 1 - np.exp(-a**2/epsilon**2)
    # return s
    #     if (a0 is None) or (a0 > a):
    #         a0 = a
    # return a
        _Q = Q1[i]+Q2[i]
        a = np.min(np.abs(np.real(np.linalg.eigvals(_Q))))
        if (a0 is None) or (a0 > a):
            a0 = a
    return np.log10(a0)/epsilon

def sfunc(Q1, Q2, c_val=-10):
    n_k = len(Q1)
    for i in range(n_k):
        _Q = Q1[i]+Q2[i]
        #print(np.log10(np.abs(np.linalg.eigvalsh(_Q))))
        for v in np.real(np.linalg.eigvals(_Q)):
            if np.log10(np.abs(v)) < c_val:
                return 0    # has a cross
    return 1

def get_kpoints(n_dim, n_mesh):
    '''
    Get the k points
    '''
    if n_dim == 1:
        kpoints = [[k] for k in np.linspace(0, np.pi, n_mesh)]
    else:
        values = [np.linspace(0, np.pi, n_mesh) for i in range(n_dim)]
        kpoints = list(itertools.product(*values))

    return kpoints

def topology_comparator(topological_model1, topological_model2, perturbation=None, epsilon=0.1, n_guess=2, sfunc_smooth=None, kpoints=None, method='Nelder-Mead', iterations=1000, c_val=-10, verbose=False):
    '''
    Compare two topological models, return the similiarity
    '''
    if sfunc_smooth is None:
        sfunc_smooth = sfunc_smooth_in

    def object_func(k):
        k = k % (2*np.pi)
        Q1 = topological_model1.calculate_Q(kpoints=[k], perturbation=perturbation)
        Q2 = topological_model2.calculate_Q(kpoints=[k], perturbation=perturbation)
        return sfunc_smooth(Q1, Q2, epsilon=epsilon)

    def evaluate_func(k):
        k = k % (2*np.pi)
        Q1 = topological_model1.calculate_Q(kpoints=[k], perturbation=perturbation)
        Q2 = topological_model2.calculate_Q(kpoints=[k], perturbation=perturbation)
        return sfunc(Q1, Q2, c_val=c_val)

    n_dim = topological_model1.get_n_dim()
    if n_dim == 0:
        return evaluate_func(0)

    #vals_guess = get_kpoints(n_dim, n_guess)
    try: 
        vals_guess = topological_model1.get_kpoints() # some old realizations may not include the method ``get_kpoints()``, just in the case
    except:
        vals_guess = get_kpoints(n_dim, n_guess)
    
    if kpoints is not None:
        vals_guess = kpoints

    #print(vals_guess)

    bounds = None
    for i in range(len(vals_guess)):
        if method == "Nelder-Mead":
            res = minimize(
                    object_func, vals_guess[i], 
                    method=method, bounds=bounds, tol=1e-12,
                    options={'maxiter': iterations, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        elif method == "BFGS":
            res = minimize(
                object_func, vals_guess[i], 
                method=method, options={'gtol': 1e-12}
            )
        elif method == "Powell":
            res = minimize(
                object_func, vals_guess[i], 
                method=method, options={'xtol': 1e-12, 'ftol': 1e-12}
            )

        _kpoint = res.x

        if verbose:
            print("Find point: {0} -> value = {1}, {2}".format(
                _kpoint, object_func(_kpoint), evaluate_func(_kpoint)
                ))
            
            print("Q1: ", np.around(topological_model1.calculate_Q(kpoints=[_kpoint], perturbation=perturbation), decimals=3))
            print("Q2: ", np.around(topological_model2.calculate_Q(kpoints=[_kpoint], perturbation=perturbation), decimals=3))

        if evaluate_func(_kpoint) == 0:
            return 0

    return 1


def topology_verifier(topological_model1, topological_model2, perturbations, similarity_func=None, epsilon=0.1, sfunc_smooth=None, n_guess=2, method='Nelder-Mead', iterations=1000, c_val=-10, verbose=False):
    n_perturbation = len(perturbations)
    similarities = np.ones(n_perturbation)
    for n in range(n_perturbation):
        # check the "perturbation" is the perturbation
        # check1 = check_perturbation(topological_model1, perturbations[n], 
        #     perturbation=perturbations[n], epsilon=epsilon, sfunc_smooth=sfunc_smooth, c_val=c_val,
        #     n_guess=n_guess, method=method, iterations=iterations)
        # check2 = check_perturbation(topological_model2, perturbations[n], 
        #     perturbation=perturbations[n], epsilon=epsilon, sfunc_smooth=sfunc_smooth, c_val=c_val,
        #     n_guess=n_guess, method=method, iterations=iterations)
        
        # if check1 and check2: 
        #     similarities[n] = topology_comparator(topological_model1, topological_model2, 
        #         perturbation=perturbations[n], epsilon=epsilon, sfunc_smooth=sfunc_smooth, c_val=c_val,
        #         n_guess=n_guess, method=method, iterations=iterations, vebose=verbose)
        if similarity_func is not None:
            similarities[n] = similarity_func(topological_model1, topological_model2, perturbation=perturbations[n])
        else:
            similarities[n] = topology_comparator(topological_model1, topological_model2, 
                perturbation=perturbations[n], epsilon=epsilon, sfunc_smooth=sfunc_smooth, c_val=c_val,
                n_guess=n_guess, method=method, iterations=iterations, verbose=verbose)

        if similarities[n] == 1:
            return 1 # for replacing np.any()

    #return np.all(similarities)
    return np.any(similarities)

def obtain_phase_center_and_number(center_indices, group_number, models, perturbations, similarity_func=None, epsilon=0.1, sfunc_smooth=None, n_guess=11, method='Nelder-Mead', iterations=1000, sc=0.5, c_val=-10, verbose=False):
    center_models = [models[i] for i in center_indices]
    n_center = len(center_models)

    similarity_matrix = np.zeros((n_center, n_center))
    for i in range(n_center):
        for j in range(i, n_center):
            similarity_matrix[i,j] = topology_verifier(center_models[i], center_models[j], perturbations, similarity_func=similarity_func, epsilon=epsilon, sfunc_smooth=sfunc_smooth, n_guess=n_guess, method=method, iterations=iterations, c_val=c_val, verbose=verbose)
            similarity_matrix[j,i] = similarity_matrix[i,j]

    if verbose:
        print("Similarity matrix for the centers: ")
        print(similarity_matrix)

    new_center_indices = list()
    new_number_group = list()

    # Add the first element to the group
    new_center_indices.append(0)
    new_number_group.append(group_number[0])

    # cluster
    for i in range(1, n_center):
        flag = True
        for i_center_index in new_center_indices:
            if similarity_matrix[i_center_index, i] > sc:
                # topologically same
                new_number_group[i_center_index] += group_number[i]
                flag = False
                break
                
        # topologically different
        if flag:
            new_center_indices.append(i)
            new_number_group.append(group_number[i])
                        
    if verbose:
        print()
        print("The new group number: ")
        print(new_number_group)
        print("The number of phases: ", len(new_number_group))

    new_center_indices = [center_indices[i] for i in new_center_indices]
    return new_center_indices, new_number_group
    
def check_perturbation(topological_model, perturbation, epsilon=0.1, n_guess=5, sfunc_smooth=None, kpoints=None, method='Nelder-Mead', iterations=1000, c_val=-10):
    '''
    Check perturbation
    '''
    if sfunc_smooth is None:
        sfunc_smooth = sfunc_smooth_in

    def object_func(k):
        Q1 = topological_model.calculate_Q(kpoints=[k])
        Q2 = topological_model.calculate_Q(kpoints=[k], perturbation=perturbation)
        return sfunc_smooth(Q1, Q2, epsilon=epsilon)

    def evaluate_func(k):
        Q1 = topological_model.calculate_Q(kpoints=[k])
        Q2 = topological_model.calculate_Q(kpoints=[k], perturbation=perturbation)
        return sfunc(Q1, Q2, c_val=c_val)

    n_dim = topological_model.get_n_dim()
    if n_dim == 0:
        return evaluate_func(0)

    vals_guess = get_kpoints(n_dim, n_guess)
    if kpoints is not None:
        vals_guess = kpoints

    bounds = None
    for i in range(len(vals_guess)):
        if method == "Nelder-Mead":
            res = minimize(
                    object_func, [vals_guess[i]], 
                    method=method, bounds=bounds, tol=1e-12,
                    options={'maxiter': iterations, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        elif method == "BFGS":
            res = minimize(
                object_func, [vals_guess[i]], 
                method=method, options={'gtol': 1e-12}
            )
        elif method == "Powell":
            res = minimize(
                object_func, [vals_guess[i]], 
                method=method, options={'xtol': 1e-12, 'ftol': 1e-12}
            )

        _kpoint = res.x

        if evaluate_func(_kpoint) == 0:
            return 0

    return 1