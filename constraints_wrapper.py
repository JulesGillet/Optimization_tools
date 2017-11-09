import numpy as np

def Ineg_wrapper(valS, valI):
    """
    Function used to wrap Inequalities into a suitable form for optimisation
    valS > valI  -->  Inequality is satisfied
    valS and valI can be float or 1d array
    """
    epsilon = 1e-6
    top = 1e3
    ecart = valI - valS
    if ecart < epsilon:
        out = np.exp(ecart) * epsilon / np.exp(epsilon)
    elif ecart > top:
        out = np.log(ecart) * top / np.log(top)
    else:
        out = ecart
    return out


def Eg_wrapper(val, ref):
    """
    Function used to wrap Equalities into a suitable form for optimisation
    val = ref  -->  Equality is satisfied
    val and ref can be float or 1d array
    """
    out = Ineg_wrapper(val, ref) + Ineg_wrapper(ref, val)
    return out