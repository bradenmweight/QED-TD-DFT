from pyscf import dft, tddft, gto
import numpy as np
import subprocess as sp

def get_ELECTRONIC_ENERGY_DIPOLE( atom="geometry.xyz", unit='Angstrom', \
                      basis='6-31G*', functional="wB97XD", nstates=2):
    
    def get_STS_dipoles(mol_obj, mf_obj, td_obj, scale=2.):
        def get_attach_dm(xy1, xy2, orbv1, orbv2=None, scale=2.):
            """xy is a tuple (x, y)"""
            rdm1 = np.einsum('ia,ib->ab', xy1[0], xy2[0])
            if isinstance(xy1[1], np.ndarray):
                rdm1 += np.einsum('ia,ib->ba', xy1[1], xy2[1])
            return scale * np.einsum('pa,ab,qb->pq', orbv1, rdm1, orbv1.conj())

        def get_detach_dm(xy1, xy2, orbo1, orbo2=None, scale=2.):
            """xy is a tuple (x, y)"""
            rdm1 = np.einsum('ia,ja->ij', xy1[0], xy2[0])
            if isinstance(xy1[1], np.ndarray):
                rdm1 += np.einsum('ia,ja->ji', xy1[1], xy2[1])
            return -scale * np.einsum('pi,ij,qj->pq', orbo1, rdm1, orbo1.conj())

        def get_STS_diff_TDMAT(xy1, xy2, coeff1, coeff2=None, scale=2.):
            """xy is a tuple (x, y)"""
            o, v = xy1[0].shape
            orbo1, orbv1 = coeff1[:,:o], coeff1[:,o:]
            orbo2, orbv2 = None, None
            if isinstance(coeff2, np.ndarray):
                orbo2, orbv2 = coeff2[:,:o], coeff2[:,o:]
            rdm1  = get_attach_dm(xy1, xy2, orbv1, orbv2, scale)
            rdm1 += get_detach_dm(xy1, xy2, orbo1, orbo2, scale)
            return rdm1
        
        mo_dipoles = mol_obj.intor('int1e_r', comp=3)
        mo_coeffs  = mf_obj.mo_coeff
        xys        = td_obj.xy
        nstates    = len(xys)
        dip_mat    = np.zeros( (nstates,nstates,3) )
        for statej in range(nstates):
            for statek in range(statej, nstates):
                TDM = get_STS_diff_TDMAT(xys[statej], xys[statek], mo_coeffs, None, scale)
                dip_mat[statej,statek,:] = np.einsum('xpq,...pq->...x', mo_dipoles, TDM)
                dip_mat[statek,statej,:] = dip_mat[statej,statek,:]
        return dip_mat

    def get_Energy( mf_obj, td_obj ):
        ENERGY    = np.zeros( td_obj.nstates+1 )
        ENERGY[0]  = mf_obj.energy_tot()
        ENERGY[1:] = ENERGY[0] + td_obj.e # GS + dE
        return ENERGY

    def get_Dipole( mol_obj, mf_obj, td_obj ):
        DIP_MAT = np.zeros((td_obj.nstates+1, td_obj.nstates+1, 3)) # Include ground state
        DIP_MAT[0,0,:]   = mf_obj.dip_moment()                       # Ground-to-Ground
        DIP_MAT[0,1:,:]  = td_obj.transition_dipole(xy=td_obj.xy)    # Ground-to-Excited
        DIP_MAT[1:,0,:]  = DIP_MAT[0,1:,:]                           # Ground-to-Excited
        DIP_MAT[1:,1:,:] = get_STS_dipoles( mol_obj, mf_obj, td_obj )
        return DIP_MAT    


    assert (nstates >= 1), "Number of states should be greater than or equal to 1"

    #mol
    mol_obj = gto.M(atom=atom, basis=basis, unit=unit, verbose=4)
    
    # DFT
    mf_obj = dft.RKS(mol_obj)
    mf_obj.xc = functional
    mf_obj.kernel()

    td_obj = tddft.TDDFT(mf_obj)
    td_obj.nstates = nstates
    td_obj.kernel()
    td_obj.analyze()

    ENERGY  = get_Energy( mf_obj, td_obj )
    DIP_MAT = get_Dipole( mol_obj, mf_obj, td_obj )

    DATA_DIR = "H_electronic_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)
    np.savetxt(f'{DATA_DIR}/ENERGY.dat', ENERGY, fmt="%1.8f")
    np.savetxt(f'{DATA_DIR}/ENERGY_TRANSITION_eV.dat', (ENERGY - ENERGY[0])*27.2114, fmt="%1.4f", header="Transition Energy (eV)")
    np.save(f'{DATA_DIR}/DIPOLE.npy', DIP_MAT)
    np.savetxt(f'{DATA_DIR}/DIPOLE_X.dat', DIP_MAT[:,:,0], fmt="%1.5f")
    np.savetxt(f'{DATA_DIR}/DIPOLE_Y.dat', DIP_MAT[:,:,1], fmt="%1.5f")
    np.savetxt(f'{DATA_DIR}/DIPOLE_Z.dat', DIP_MAT[:,:,2], fmt="%1.5f")

    return ENERGY, DIP_MAT

if ( __name__ == '__main__' ):
    atom       = '''
    H       -0.9450370725    -0.0000000000     1.1283908757
    C       -0.0000000000     0.0000000000     0.5267587663
    H        0.9450370725     0.0000000000     1.1283908757
    O        0.0000000000    -0.0000000000    -0.6771667936
    '''
    basis      = 'sto3g'
    unit       = 'Angstrom'
    functional = "b3lyp"
    nstates    = 30 # Number of excited states
    ENERGY_AD, DIPOLE_AD = get_ELECTRONIC_ENERGY_DIPOLE( atom=atom, unit=unit, basis=basis, functional=functional, nstates=nstates)