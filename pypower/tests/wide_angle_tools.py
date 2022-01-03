# copy-paste from https://github.com/fbeutler/pk_tools/blob/master/wide_angle_tools.py

import numpy as np

def get_end_point_LOS_M(d, Nkth=400, kmin=0., kmax=0.4):
    '''
    Returns the transformation matrix M assuming the
    End-point LOS definition
    Input:
    d = Comoving distance to the effective redshift
    Nkth = The number of bins in kth
    kmin = The lower k-range limit
    kmax = The upper k-range limit

    -> Note that large k-bins can lead to significant error
    -> The matrix is always 3*Nkth x 5*Nkth, mapping from
    (P_0, P_2, P_4) to (P_0, P_1, P_2, P_3, P_4)
    '''
    print("Nkth =", Nkth)
    print("(kmin, kmax) =", (kmin,kmax))
    M = np.zeros((Nkth*5, Nkth*3))

    dkp_th = (kmax-kmin)/Nkth
    kp_th = np.array([kmin + i*dkp_th + dkp_th/2. for i in range(0,Nkth)])

    K1 = (3./(5.*d))*( (3./kp_th)*np.ones(Nkth) )
    K2 = (1./d)*(3./5.)*( (2./kp_th)*np.ones(Nkth) )
    K3 = (1./d)*(10./9.)*( (5./kp_th)*np.ones(Nkth) )

    # Populate matrix M
    # We start with the three unity matrices
    M[:Nkth, :Nkth] = np.identity(Nkth)
    M[2*Nkth:3*Nkth, Nkth:2*Nkth] = np.identity(Nkth)
    M[4*Nkth:5*Nkth, 2*Nkth:3*Nkth] = np.identity(Nkth)

    # Now we add the K matrices, which however have off-diagonal elements
    # We start with the diagonal elements
    M[Nkth:2*Nkth, Nkth:2*Nkth] = np.diag(K1)
    M[3*Nkth:4*Nkth, Nkth:2*Nkth] = np.diag(K2)
    M[3*Nkth:4*Nkth, 2*Nkth:3*Nkth] = np.diag(K3)

    #return M
    # Now we add the (forward) derivative
    #for ik, k in enumerate(kp_th[:-1]):
    for ik, k in enumerate(kp_th):
        # K1 derivative (see eq. 4.3)
        M = _populate_derivative(d, M.copy(), Nkth+ik, Nkth+ik, ik, 3./5., kp_th)
        # # K2 derivative (see eq. 4.4)
        M = _populate_derivative(d, M.copy(), 3*Nkth+ik, Nkth+ik, ik, -3./5., kp_th)
        # # K3 derivative (see eq. 4.5)
        M = _populate_derivative(d, M.copy(), 3*Nkth+ik, 2*Nkth+ik, ik, 10./9., kp_th)
    return M


def _populate_derivative(d, M, index1, index2, ik, pre_factor, kp_th):
    '''
    Populate the derivative part of the M matrix
    '''
    delta_k = kp_th[1] - kp_th[0]
    norm = 0
    # If we are at the edge we do a one sided derivative
    # otherwise two sided
    #ik = 1
    if ik > 0 and ik < len(kp_th)-1:
        norm = 2.*delta_k
    else:
        norm = delta_k


    if ik > 0:
        M[index1, index2-1] = -pre_factor/(d*norm)
    else:
        M[index1, index2] += -pre_factor/(d*norm)

    if ik < len(kp_th)-1:
        M[index1, index2+1] = pre_factor/(d*norm)
    else:
        M[index1, index2] += pre_factor/(d*norm)
    return M
