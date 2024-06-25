import numpy as np
from pyscf import gto, scf
from matplotlib import pyplot as plt

import pQED
import scQED

def get_system():

    mol         = gto.M()
    mol.verbose = 1
    mol.atom    = '''
    H       -0.9450370725    -0.0000000000     1.1283908757
    C       -0.0000000000     0.0000000000     0.5267587663
    H        0.9450370725     0.0000000000     1.1283908757
    O        0.0000000000    -0.0000000000    -0.6771667936
    '''
    mol.unit = 'Angstrom'
    mol.basis = 'sto3g'
    mol.build()

    mf    = scf.RKS(mol)
    mf.xc = "b3lyp"
    mf.kernel()

    return mol, mf


def lambda_scan( mol, mf, NSTATES, cavity_freq, cavity_epol, LAM_LIST ):

    # GET THE DATA
    E_scQED   = np.zeros( (len(LAM_LIST), NSTATES) )
    E_pQED    = np.zeros( (len(LAM_LIST), NSTATES+1) )
    PHOT_pQED = np.zeros( (len(LAM_LIST), NSTATES+1) )
    for LAMi,LAM in enumerate(LAM_LIST):
        print( "Working on lambda = %1.3f" % LAM )
        cavity_lambda  = LAM
        E_scQED[LAMi,:] = scQED.main( mol, mf, cavity_lambda, cavity_freq, cavity_epol, NSTATES )
        E_pQED[LAMi,:], PHOT_pQED[LAMi,:]  = pQED.main(WFNS=False, NSTATES=NSTATES+1, NEL=31, NF=10, EPOL=cavity_epol, 
                                    LAMBDA=cavity_lambda, WC=cavity_freq, HAM="JC", SUBSPACE="TRUNCATED", doPHOT=True)
        #E_pQED[LAMi,:], PHOT_pQED[LAMi,:]  = pQED.main(WFNS=False, NSTATES=NSTATES+1, NEL=31, NF=10, EPOL=cavity_epol, 
        #                            LAMBDA=cavity_lambda, WC=cavity_freq, HAM="PF", SUBSPACE="FULL", doPHOT=True)
    
    # CONVERT TO eV
    E_scQED *= 27.2114
    E_pQED  *= 27.2114

    # PLOT THE DATA
    for state in range( NSTATES ):
        if ( state == 0 ):
            plt.plot( LAM_LIST, E_pQED[:,state+1] - E_pQED[:,0], "-", lw=5, c='black', label="pQED" )
            plt.plot( LAM_LIST, E_scQED[:,state], "o", lw=3, c='red', label="scQED" )
        else:
            plt.plot( LAM_LIST, E_pQED[:,state+1] - E_pQED[:,0], "-", lw=5, c='black' )
            plt.plot( LAM_LIST, E_scQED[:,state], "o", lw=3, c='red' )
    plt.xlabel( "Coupling Strength, $\lambda_\mathrm{c}$ (a.u.)" , fontsize=15 )
    plt.ylabel( "Energy (eV)", fontsize=15 )
    plt.legend()
    plt.xlim(LAM_LIST[0],LAM_LIST[-1])
    plt.ylim(9,15)
    plt.tight_layout()
    plt.savefig("pQED_scQED_lambda.jpg", dpi=300)
    plt.clf()

    # PLOT THE DIFFERENCE
    for state in range( NSTATES ):
        dE_pQED = E_pQED[:,state+1] - E_pQED[:,0]
        dE      = dE_pQED - E_scQED[:,state]
        plt.plot( LAM_LIST, dE, "-", lw=5, label=f"State {state+1}" )
    plt.xlabel( "Coupling Strength, $\lambda_\mathrm{c}$ (a.u.)" , fontsize=15 )
    plt.ylabel( "Energy Difference, $E^\mathrm{pQED} - E^\mathrm{scQED}$ (eV)", fontsize=15 )
    plt.legend()
    plt.xlim(LAM_LIST[0],LAM_LIST[-1])
    plt.tight_layout()
    plt.savefig("pQED_scQED_DIFF_lambda.jpg", dpi=300)
    plt.clf()

    # PLOT THE PHOTON NUMBER
    color_list = ["black", "red", "blue", "green", "purple", "orange", "cyan", "magenta", "brown", "pink", "gray"]
    for state in range( NSTATES ):
        plt.plot( LAM_LIST, PHOT_pQED[:,state], "-", c=color_list[state], lw=5, label=f"State {state+1}" )
    plt.xlabel( "Coupling Strength, $\lambda_\mathrm{c}$ (a.u.)" , fontsize=15 )
    plt.ylabel( "Photon Number, $\langle \hat{a}^\dag \hat{a} \\rangle$", fontsize=15 )
    plt.legend()
    plt.xlim(LAM_LIST[0],LAM_LIST[-1])
    plt.tight_layout()
    plt.savefig("pQED_scQED_PHOT_lambda.jpg", dpi=300)
    plt.clf()

def freq_scan( mol, mf, NSTATES, cavity_freq_list, cavity_epol, cavity_lambda ):

    # GET THE DATA
    E_scQED   = np.zeros( (len(cavity_freq_list), NSTATES) )
    E_pQED    = np.zeros( (len(cavity_freq_list), NSTATES+1) )
    PHOT_pQED = np.zeros( (len(cavity_freq_list), NSTATES+1) )
    for wci,cavity_freq in enumerate(cavity_freq_list):
        print( "Working on wc = %1.3f eV" % (cavity_freq*27.2114) )
        E_scQED[wci,:] = scQED.main( mol, mf, cavity_lambda, [cavity_freq], cavity_epol, NSTATES )
        E_pQED[wci,:], PHOT_pQED[wci,:]  = pQED.main(WFNS=False, NSTATES=NSTATES+1, NEL=31, NF=10, EPOL=cavity_epol, 
                                    LAMBDA=cavity_lambda, WC=cavity_freq, HAM="JC", SUBSPACE="TRUNCATED", doPHOT=True)
    
    # CONVERT TO eV
    E_scQED *= 27.2114
    E_pQED  *= 27.2114
    cavity_freq_list *= 27.2114

    # PLOT THE DATA
    for state in range( NSTATES ):
        if ( state == 0 ):
            plt.plot( cavity_freq_list, E_pQED[:,state+1] - E_pQED[:,0], "-", lw=5, c='black', label="pQED" )
            plt.plot( cavity_freq_list, E_scQED[:,state], "o", lw=3, c='red', label="scQED" )
        else:
            plt.plot( cavity_freq_list, E_pQED[:,state+1] - E_pQED[:,0], "-", lw=5, c='black' )
            plt.plot( cavity_freq_list, E_scQED[:,state], "o", lw=3, c='red' )
    plt.xlabel( "Cavity Frequency, $\omega_\mathrm{c}$ (eV)" , fontsize=15 )
    plt.ylabel( "Transition Energy (eV)", fontsize=15 )
    plt.legend()
    plt.xlim(cavity_freq_list[0],cavity_freq_list[-1])
    plt.ylim(9,15)
    plt.tight_layout()
    plt.savefig("pQED_scQED_freq.jpg", dpi=300)
    plt.clf()

    # PLOT THE DIFFERENCE
    for state in range( NSTATES ):
        dE_pQED = E_pQED[:,state+1] - E_pQED[:,0]
        dE      = dE_pQED - E_scQED[:,state]
        plt.plot( cavity_freq_list, dE, "-", lw=5, label=f"State {state+1}" )
    plt.xlabel( "Cavity Frequency, $\omega_\mathrm{c}$ (eV)" , fontsize=15 )
    plt.ylabel( "Energy Difference, $E^\mathrm{pQED} - E^\mathrm{scQED}$ (eV)", fontsize=15 )
    plt.legend()
    plt.xlim(cavity_freq_list[0],cavity_freq_list[-1])
    plt.tight_layout()
    plt.savefig("pQED_scQED_DIFF_freq.jpg", dpi=300)
    plt.clf()

    # PLOT THE PHOTON NUMBER
    color_list = ["black", "red", "blue", "green", "purple", "orange", "cyan", "magenta", "brown", "pink", "gray"]
    for state in range( NSTATES ):
        plt.plot( cavity_freq_list, PHOT_pQED[:,state], "-", c=color_list[state], lw=5, label=f"State {state+1}" )
    plt.xlabel( "Cavity Frequency, $\omega_\mathrm{c}$ (eV)" , fontsize=15 )
    plt.ylabel( "Photon Number, $\langle \hat{a}^\dag \hat{a} \\rangle$", fontsize=15 )
    plt.legend()
    plt.xlim(cavity_freq_list[0],cavity_freq_list[-1])
    plt.tight_layout()
    plt.savefig("pQED_scQED_PHOT_freq.jpg", dpi=300)
    plt.clf()

if ( __name__ == "__main__"):

    # DEFINE THE SYSTEM
    mol, mf = get_system()
    NSTATES  = 10
    cavity_freq    = np.array([11.5/27.2114])
    cavity_epol    = np.array([1, 1, 1])
    LAM_LIST = np.arange( 0.0, 0.21, 0.01 )
    #lambda_scan( mol, mf, NSTATES, cavity_freq, cavity_epol, LAM_LIST )


    # DEFINE THE SYSTEM
    mol, mf = get_system()
    NSTATES  = 10
    cavity_freq_list    = np.arange(9.0,15.05,0.25)/27.2114
    cavity_epol    = np.array([1, 1, 1])
    LAMDA = 0.05
    freq_scan( mol, mf, NSTATES, cavity_freq_list, cavity_epol, LAMDA )