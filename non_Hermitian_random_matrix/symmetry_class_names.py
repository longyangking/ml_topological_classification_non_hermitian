non_Hermitian_symmetry_list_name = {
    # Complex AZ class
    "A":    r"A", 
    "AIII": r"AIII",
    # Real AZ class
    "AI":   r"AI",
    "BDI":  r"BDI",
    "D":    r"D",
    "DIII": r"DIII",
    "AII":  r"AII", 
    "CII":  r"CII",
    "C":    r"C",
    "CI":   r"CI",
    "AI+":  r"AI$^\dagger$",
    "BDI+": r"BDI$^\dagger$",
    # "D+":   ['D', 'AI', 'BDI'],
    "DIII+": r"DIII$^\dagger$",
    "AII+": r"AII$^\dagger$",
    "CII+": r"CII$^\dagger$",
    # "C+":   ['C', 'AII', 'CII'],
    "CI+":  r"CI$^\dagger$",
    # Complex AZ class with sublattice symmetry
    "A:S":      r"A, $\mathcal{S}$", 
    "AIII:S+":  r"AIII, $\mathcal{S}_{+}$",
    "AIII:S-":  r"AIII, $\mathcal{S}_{-}$",
    # Real AZ class with sublattice symmetry
    "BDI:S++":  r"BDI, $\mathcal{S}_{++}$",
    "DIII:S--":  r"DIII, $\mathcal{S}_{--}$",
    "CII:S++":  r"CII, $\mathcal{S}_{++}$", 
    "CI:S--":   r"CI, $\mathcal{S}_{--}$",
    "AI:S-":    r"AI, $\mathcal{S}_{-}$",
    "BDI:S-+":  r"BDI, $\mathcal{S}_{-+}$",
    "D:S+":     r"D, $\mathcal{S}_{+}$",
    "CII:S-+":  r"CII, $\mathcal{S}_{-+}$",
    "C:S+":     r"C, $\mathcal{S}_{+}$",
    "DIII:S++": r"DIII, $\mathcal{S}_{++}$",
    "CI:S++":   r"CI, $\mathcal{S}_{++}$",
    "AI:S+":    r"AI, $\mathcal{S}_{+}$",
    "BDI:S+-":  r"BDI, $\mathcal{S}_{+-}$",
    "D:S-":     r"D, $\mathcal{S}_{-}$",
    "DIII:S+-": r"DIII, $\mathcal{S}_{+-}$",
    "AII:S+":   r"AII, $\mathcal{S}_{+}$",
    "CII:S+-":  r"CII, $\mathcal{S}_{+-}$",
    "C:S-":     r"C, $\mathcal{S}_{-}$",
    "CI:S+-":   r"CI, $\mathcal{S}_{+-}$",
}

non_Hermitian_symmetry_list_name_parity = dict()
for key in non_Hermitian_symmetry_list_name:
    sym_name = non_Hermitian_symmetry_list_name[key]
    if key not in ['A', 'AIII', "A:S", "AIII:S+", "AIII:S-"]:
        sym_name = r'$\mathcal{P}$'+sym_name
    non_Hermitian_symmetry_list_name_parity[key] = sym_name
