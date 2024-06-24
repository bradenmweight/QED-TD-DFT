import numpy as np
from pyscf import gto, scf

import qed


def get_scQED( mol, mf, cavity_lambda, cavity_freq, cavity_epol, NSTATES ):

    cavity_epol = cavity_lambda * cavity_epol / np.linalg.norm(cavity_epol)

    # TDA-JC
    cav_model = qed.JC(mf, cavity_mode=cavity_epol, cavity_freq=cavity_freq)
    td        = qed.TDDFT(mf, cav_obj=cav_model)
    td.nroots = NSTATES
    td.kernel()

    return td.e

def main( mol, mf, cavity_lambda, cavity_freq, cavity_epol, NSTATES ):
    return get_scQED( mol, mf, cavity_lambda, cavity_freq, cavity_epol, NSTATES )

