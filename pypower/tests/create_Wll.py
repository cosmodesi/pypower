# copy-paste from https://github.com/fbeutler/pk_tools/blob/master/create_Wll.py

import os, sys
import numpy as np

from scipy.interpolate import interp1d
from scipy import special as sp

from hankl import P2xi, xi2P


def create_W(kbins, s_win, window, outpath=''):
    '''
    INPUT
    kbins: Array of k-bins,
    -> k' bins are automatically determined (see Nfft below)
    s_win: binning of the config-space window Q_ell(s)
    window: config-space window Q_ell(s)

    OUTPUT
    Window function W_{ell,ell'}(k,k') as given in eq. 2.5 of arxiv:2106.06324
    -> This window function needs to be averaged to the observational
    and theoretical k-bins as shown in eq. 2.16 (also see appendix E)
    '''
    smin = min(s_win)
    smax = max(s_win)
    Nfft = 1024*16
    Nk = len(kbins)
    print("Calculate window function of size %d x %d" % (Nk, Nfft))
    ds = np.log(smax/smin)/float(Nfft)
    slog = np.array([smin*np.exp(i*ds) for i in range(0, Nfft)])

    C0_n02 = [
        [ [1., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 1./5., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 1./9.] ],
        [ [0., 1., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 2./5., 0., 9./35., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 4./21., 0.] ],
        [ [0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.], [1., 0., 2./7., 0., 2./7.], [0., 0., 0., 0., 0.], [0., 0., 2./7., 0., 100./693.] ],
        [ [0., 0., 0., 1., 0.], [0., 0., 0., 0., 0.], [0., 3./5., 0., 4./15., 0.], [0., 0., 0., 0., 0.], [0., 4./9., 0., 2./11., 0.] ],
        [ [0., 0., 0., 0., 1.], [0., 0., 0., 0., 0.], [0., 0., 18./35., 0., 20./77.], [0., 0., 0., 0., 0.], [1., 0., 20./77., 0., 162./1001.] ] ];

    C0_n1 = [
        [ [0., 0., 0., 0., 0.], [0., 1./3., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 1./7., 0.], [0., 0., 0., 0., 0.] ],
        [ [0., 0., 0., 0., 0.], [1., 0., 2./5., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 9./35., 0., 4./21.], [0., 0., 0., 0., 0.] ],
        [ [0., 0., 0., 0., 0.], [0., 2./3., 0., 3./7., 0.], [0., 0., 0., 0., 0.], [0., 3./7., 0., 4./21., 0.], [0., 0., 0., 0., 0.] ],
        [ [0., 0., 0., 0., 0.], [0., 0., 3./5., 0., 4./9.], [0., 0., 0., 0., 0.], [1., 0., 4./15., 0., 2./11.], [0., 0., 0., 0., 0.] ],
        [ [0., 0., 0., 0., 0.], [0., 0., 0., 4./7., 0.], [0., 0., 0., 0., 0.], [0., 4./7., 0., 18./77., 0.], [0., 0., 0., 0., 0.] ] ];

    ###############################################
    # Compute W_{\ell,\ell'} in Fourier space
    ###############################################
    Well = {}
    for n in range(0, 2):              # n = 0,1,2
        for ell in range(0, 5):      # ell = 0,1,2,3,4
            for ellp in range(0, 5): # ell' = 0,1,2,3,4
                W = np.zeros((Nk, Nfft))
                print("n = %d, ell = %d, ell' = %d" % (n, ell, ellp))
                for ik, k in enumerate(kbins):

                    IntegrandPi = np.zeros(Nfft)
                    for L in range(0, 5): # L = 0,1,2,3,4 : Q_L

                        interp_QL = interp1d(s_win, window[n][L], kind='linear', fill_value=((1. if L == 0 else 0.),0.), bounds_error=False)

                        if n == 0 or n == 2:
                            IntegrandPi += C0_n02[ell][ellp][L]*interp_QL(slog)
                        else:
                            IntegrandPi += C0_n1[ell][ellp][L]*interp_QL(slog)

                    kp, Pi = xi2P(slog, IntegrandPi*sp.spherical_jn(ell,slog*k), l=ellp, n=0)

                    prefactor = (-1. if ell%2 == 1 else 1.)
                    prefactor *= (-1. if ell == 1 or ell == 2 else 1.)
                    prefactor *= (1. if ellp%2 == 0 else -1.)

                    W[ik] = prefactor*(2./np.pi)*(Pi.real if ellp%2 == 0 else Pi.imag)/(4.*np.pi)

                Well[(n,ell,ellp)] = W
                if outpath:
                    # Write to file W_{ell,ell'} /////////////////////////
                    filename = "%s/W%d%d%d_%d_%d.dat" % (outpath, n, ell, ellp, Nfft, Nk)
                    with open(filename, 'w') as f:
                        for ikp in range(0, Nfft):
                            for ik in range(0, Nk):
                                f.write("%0.16f " % W[ik][ikp])
                            f.write("\n")

    if outpath:
        filename = "%s/kp_%d_%d.dat" % (outpath, Nfft, Nk)
        with open(filename, 'w') as f:
            for ikp in range(0, Nfft):
                f.write("%0.16f\n" % kp[ikp])

        filename = "%s/k_%d_%d.dat" % (outpath, Nfft, Nk)
        with open(filename, 'w') as f:
            for ik in range(0, Nk):
                if isinstance(kbins[ik], float):
                    f.write("%0.16f\n" % kbins[ik])
                else:
                    f.write("%0.16f\n" % 0.)
    return Well
