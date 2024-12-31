import numpy as np

complex_symmetry_class = ['A', 'AIII']
real_symmetry_class = ['AI', 'AII', 'BDI', 'C', 'CI', 'CII', 'D', 'DIII']

## real line, imaginary line, point gap
non_Hermitian_symmetry_list_Hermitian_counterparts = {
    # Complex AZ class
    "A":    ['A', 'A', 'AIII'], 
    "AIII": ['AIII', 'A', 'A'],
    # Real AZ class
    "AI":   ['AI', 'D', 'BDI'],
    "BDI":  ['BDI', 'D', 'D'],
    "D":    ['D', 'D', 'DIII'],
    "DIII": ['DIII', 'A', 'AII'],
    "AII":  ['AII', 'C', 'CII'], 
    "CII":  ['CII', 'C', 'C'],
    "C":    ['C', 'C', 'CI'],
    "CI":   ['CI', 'A', 'AI'],
    "AI+":  ['AI', 'AI', 'CI'],
    "BDI+": ['BDI', 'AI', 'AI'],
    # "D+":   ['D', 'AI', 'BDI'],
    "DIII+": ['DIII', 'A', 'D'],
    "AII+": ['AII', 'AII', 'DIII'],
    "CII+": ['CII', 'AII', 'AII'],
    # "C+":   ['C', 'AII', 'CII'],
    "CI+":  ['CI', 'A', 'C'],
    # Complex AZ class with sublattice symmetry
    "A:S":      ['AIII', 'AIII', 'AIII'], 
    "AIII:S+":  ['AIII', 'AIII', 'AIII'],
    "AIII:S-":  ['A', 'A', 'A'],
    # Real AZ class with sublattice symmetry
    "BDI:S++":  ['BDI', 'BDI', 'BDI'],
    "DIII:S--":  ['DIII', 'AIII', 'DIII'],
    "CII:S++":  ['CII', 'CII', 'CII'], 
    "CI:S--":   ['CI', 'AIII', 'CI'],
    "AI:S-":    ['CI', 'DIII', 'AIII'],
    "BDI:S-+":  ['AI', 'D', 'A'],
    "D:S+":     ['BDI', 'BDI', 'AIII'],
    "CII:S-+":  ['AII', 'C', 'A'],
    "C:S+":     ['CII', 'CII', 'AIII'],
    "DIII:S++": ['AIII', 'AIII', 'CII'],
    "CI:S++":   ['AIII', 'AIII', 'BDI'],
    "AI:S+":    ['BDI', 'BDI', 'BDI'],
    "BDI:S+-":  ['D', 'D', 'D'],
    "D:S-":     ['DIII', 'DIII', 'DIII'],
    "DIII:S+-": ['AII', 'AII', 'AII'],
    "AII:S+":   ['CII', 'CII', 'CII'],
    "CII:S+-":  ['C', 'C', 'C'],
    "C:S-":     ['CI', 'CI', 'CI'],
    "CI:S+-":   ['AI', 'AI', 'AI'],
}

non_hermitian_symmetry_list = list(non_Hermitian_symmetry_list_Hermitian_counterparts.keys())

def is_complexification_line_gap(symmetry_class):
    if symmetry_class in ['DIII', 'CI', 'DIII+', 'CI+']:
        return False, True

    if symmetry_class in ['AIII:S-']:
        return True, True

    return False, False

def is_complexification_point_gap(symmetry_class):
    if symmetry_class in ['AI:S-', 'BDI:S-+', 'D:S+', 'CII:S-+', 'C:S+']:
        return True

    return False

def is_block_diagonalization_line_gap(symmetry_class):
    if symmetry_class in ['BDI', 'CII', 'BDI+', 'CII+', 'AIII']:
        return False, True

    if symmetry_class in ['AIII:S+', 'BDI:S++', 'CII:S++']:
        return True, True

    if symmetry_class in ['CI:S--', 'DIII:S--']:
        return True, False

    return False, False

def is_block_diagonalization_point_gap(symmetry_class):
    if symmetry_class in [
        'A:S', 'AIII:S-', 'AI:S+', 'BDI:S+-', 'D:S-', 
        'DIII:S+-', 'AII:S+', 'CII:S+-', 'C:S-', 'CI:S+-'
    ]:
        return True
    
    return False

def is_complexification(symmetry_class):
    real_complex, imag_complex = is_complexification_line_gap(symmetry_class)
    point_complex = is_complexification_point_gap(symmetry_class)

    return real_complex, imag_complex, point_complex