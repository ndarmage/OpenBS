#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module implements the homogeneous B1 flux calculation in the
infinite (homogeneous) medium. The multigroup formalism in energy is
used. The methodology follows stricly Hebert's textbook. This module
assumes that the macroscopic cross sections are here provided as known
data. The unknowns are the multigroup neutron flux and the buckling
B**2, which is the eigenvalue. We solve this non-linear eigenvalue
problem by a root-finding method, or by an ordinary linear generalized
eigenvalue problem after a suitable change of variable. This last
solving method is suggested by the polynomial approximation of the
function gamma.

.. note:: The coefficients of the polynomial approximations for gamma
          are obtained by the python package sympy (series function).


References
==========

1. Hébert A., Applied reactor physics. Presses Inter. Polytechnique
   (2009).
"""
import os
import logging as lg
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.optimize as opt

__title__ = "MG Homog B1 calculation"
__author__ = "D. Tomatis"
__date__ = "30/06/2020"
__version__ = "1.1.0"

polar2xy = lambda rho, theta: rho * (np.sin(theta) + 1j * np.cos(theta))

# Ch. 4, pp 236, Eq. (4.127) from Hebert's textbook and series expansion
# by sympy (Maclaurin - series(expr, x=None, x0=0, n=6, dir='+'))
coefs = np.array([4. / 15.,
               - 12. / 175.,
                 92. / 2625.,
             - 7516. / 336875.,
              347476 / 21896875,
          - 83263636 / 6897515625])


def gamma_approx(x, cs=coefs):
    g = cs[-1]
    for c in np.append(cs[-2::-1], [1]):
        g = g * x + c
    return g


def alpha(B2, S=1.):
    return Salpha(B2, S) / S


def Salpha(B2, S=1.):
    "alpha times Sigma function, see theory."
    BoS = np.sqrt(abs(B2)) / S
    # B2oS2 = B2 / S**2
    # print('B2', B2, np.isclose(B2, 0, atol=1e-5))
    
    a = np.where(np.isclose(BoS, 0, atol=1e-5),
        # default criteria for close proximity: rtol=1e-05, atol=1e-08
        1 - BoS**2 / 3 + BoS**4 / 5 - BoS**6 / 7 + BoS**8 / 9,
        # 1 - B2oS2 / 3 + B2oS2**2 / 5 - B2oS2**3 / 7 + B2oS2**4 / 9,
        np.arctan(BoS) / BoS if B2 > 0 else
            np.log(abs((1 + BoS)/(1 - BoS))) / BoS / 2
        )
    
    # the following works only in presence of single scalar values
    # # if np.isclose(B2, 0, atol=1e-5):
    # if np.isclose(BoS, 0, atol=1e-5):
        # # default criteria for close proximity: rtol=1e-05, atol=1e-08
        # a = 1 - BoS**2 / 3 + BoS**4 / 5 - BoS**6 / 7 + BoS**8 / 9
    # elif B2 > 0:
        # a = np.arctan(BoS) / BoS  # 1 / np.tan(BoS) / BoS
    # else:
        # a = np.log(abs((1 + BoS)/(1 - BoS))) / BoS / 2
    # print('a', a)
    return a


def Salpha_prime(B2, S=1.):
    "Derivative of the alpha times Sigma function on B2."
    BoS, S2 = np.sqrt(abs(B2)) / S, S**2
    
    # print('B2', B2, np.isclose(B2, 0, atol=1e-5))
    a = Salpha(B2, S)
    
    ap = np.where(np.isclose(BoS, 0, atol=1e-5),
        # default criteria for close proximity: rtol=1e-05, atol=1e-08
        -(1 / 3 - 2 / 5 *BoS**2 + 3 / 7 * BoS**4 - 4 / 9 * BoS**6
          + 5 / 11 * BoS**8) / S2, (- a + (
            (1 / (1 + BoS**2)) if B2 > 0 else
            (abs((1 - BoS)/(1 + BoS)) * np.sign(1 - BoS) / (1 - BoS)**2)
               )) / (2 * B2)
        )
    return ap


def beta(B2, S=1.):
    return (1 - Salpha(B2, S)) / B2


def gamma(B2, S=1.):
    try:
        n = len(S)
    except:
        n = 1
    return np.ones(n) if np.isclose(B2, 0) else \
        np.where(np.isclose(B2, -S**2), 1 / 3,
                     alpha(B2, S) / beta(B2, S) / 3 / S)


def gamma_prime(B2, S=1.):
    "First derivative of the gamma function (without approximation)."
    gp = 4 / 15 / S**2
    if not np.isclose(B2, 0):
        f, fp = Salpha(B2, S), Salpha_prime(B2, S)
        gp = gamma(B2, S) * (1 / B2 + fp / f / (1 - f))
    return gp
    

def extract_real_elements(v):
    return v[np.isclose(v.imag, 0)].real


def isfundamental(flx):
    "Check if the input flux is the fundamental one."
    return ((flx > 0).all() or (flx < 0).all())


def get_Hr(st, ss, B2, adjoint=False, g=gamma):
    "Compute the B2-dependent removal term minus isotropic scattering."
    R = B2 * np.linalg.inv(
        get_G(st, ss, B2, gamma_func=g)
                          ) - ss[0,:,:]
    np.fill_diagonal(R, st + R.diagonal())
    return (R.conj().T if adjoint else R)


def get_H(xs, B2=0, adjoint=False, g=gamma, one_over_k=1.):
    "Compute the Boltzman operator for the conservation equation."
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    P = np.outer(chi, nsf) if chi.ndim == 1 else np.dot(chi, nsf)
    if adjoint:
        P = P.transpose()
    H = get_Hr(st, ss, B2, adjoint, g) - one_over_k * P
    return H  # as a function of B2


def get_R(xs, one_over_k=1.):
    "Compute the removal matrix."
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    R = - np.array(ss[0,:,:], copy=True)
    np.fill_diagonal(R, st + R.diagonal())
    if abs(one_over_k) > 0:
        if chi.ndim == 1:
            P = np.outer(chi, nsf)
        else:
            P = np.dot(chi, nsf)
        R -= one_over_k * P
    return R


def get_T0(xs, B2=0., one_over_k=1.):
    """Compute the transport operator on B2 whose form is used for the
    inverse iterations search."""
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    G = st.size
    T = np.dot(np.diag(st * gamma(B2, st)) - ss[1,:,:],
               get_R(xs, one_over_k))
    np.fill_diagonal(T, np.full(G, B2 / 3))
    return T


def get_Tprime(xs, B2=0, one_over_k=1.):
    """Compute the first derivitaive of the transport operator on B2.
    The form of the Boltzmann eqn is the one used for the inverse
    iterations search."""
    st = xs[0]
    G = st.size
    Tp = np.dot(np.diag(st * gamma_prime(B2, st)), get_R(xs, one_over_k))
    np.fill_diagonal(Tp, np.full(G, 1 / 3))
    return Tp


def get_G(st, ss, B2, gamma_func=gamma_approx):
    "Compute the B2-dependent operator term of the current equation."
    # print('BoS', np.sqrt(abs(B2)) / st)
    # print('approx', gamma_approx(B2 / st**2))
    # print('gamma', gamma(B2, st))
    # input('ok')
    if gamma_func == gamma_approx:
        gf = gamma_func(B2 / st**2)
    elif gamma_func == gamma:
        gf = gamma_func(B2, st)
    else:
        raise ValueError('unknown input function gamma')
    G = - np.array(ss[1,:,:], copy=True)
    np.fill_diagonal(G, 3. * gf * st + G.diagonal())
    return G


def compute_deltak(B2, xs=None, k=1, f=gamma_approx):
    "Compute the k eigenvalue corresponding to the input B2."
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    k_B2, flx = compute_kpairs(xs, B2, g=f)
    if chi.shape[1] == 1:
        dk = k_B2 - k
    else:
        dk = k_B2.max() - k
    return dk


def compute_kpairs(xs, B2=0., adjoint=False, g=gamma_approx):
    "Compute the multiplication factor eigenpairs using the input B2."
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    # get the fundamental eigenpair assuming a single family of fissiles
    if adjoint:
        chi, nsf = nsf.T, chi.T
    flx = np.dot(np.linalg.inv(get_Hr(st, ss, B2, adjoint, g)), chi)
    k = np.dot(nsf, flx)
    if k.size > 1:
        # calculate the matrix whose eigenvalues are the k's and whose
        # eigenvectors are the fission rates: F.R^-1(B2).chi
        k, fiss = np.linalg.eig(k)
        idx = np.argsort(k)
        k, fiss, flx = k[idx], fiss[:, idx], flx[:, idx]
        # ...in case more isotopes than the only fissiles were included:
        idx = np.nonzero(k)
        k, fiss, flx = k[idx], fiss[:, idx], flx[:, idx]
    else:
        k = k.item(0)
    return k, flx


def power_iteration(A, toll=1.e-6, itnmax=10):
    """Power iteration method to find the eigenvalue of A with maximum
    magnitude."""
    m, n = A.shape
    if n != m:
        raise ValueError("The input matrix is not square.")
    err, i, f = 1.e+20, 1, (np.ones(n,) / n)
    while (i <= itnmax) and (err >= toll):
        fold = np.array(f, copy=True)
        f = np.dot(A, f)
        f /= np.linalg.norm(f)
        err = abs(np.where(f > 0, 1. - fold / f, f - fold)).max()
        i += 1
    return np.dot(f, np.dot(A, f)) / np.dot(f, f), f


def find_B2_spectrum(xs, one_over_k=1., nb_eigs=None, g=coefs,
                     vrbs=False):
    "Find the B2 spectrum given the input arguments."
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    # N: polynomial order after truncation of the series expansion of
    #    the gamma function
    # G: nb. of energy groups
    N, G = len(g), st.size
    if N <= 0:
        raise ValueError('Invalid input array of coefs.')
    GN = G * N
    # consider cases with trailing zeros for terms of higher degrees
    while np.isclose(g[N-1], 0):
        N -= 1
    # the first coefficient is always equal to 1, and so it does not
    # appear in the coefs of g 
    # N += 1  # ...to consider the actual polynomial order
    # WARNING: N becomes indeed N-1 hereafter!
    if vrbs:
        print('Coefficients' + str(g))
        print('Polynomial order %d' % (N + 1))
    
    # R: removal matrix
    # C: transport matrix
    R = get_R(xs, one_over_k)
    C = - np.array(ss[1,:,:], copy=True) / 3.
    np.fill_diagonal(C, st + C.diagonal())
    
    Tm1 = 1. / st
    Tm2, Tm1R = np.diag(Tm1**2), np.dot(np.diag(Tm1), R)
    # A2 = np.dot(Tm2, A1)
    # A3 = np.dot(Tm2, A2)
    # a1, a2, a3, a4 = np.dot(C, R), g1 * A1, g2 * A2, g3 * A3
    # a2 += np.identity(G) / 3.
    # M1, M2: matrices at LHS and RHS of the generalized eigenvalue
    #         problem, respectively.
    M1, M2 = np.identity(GN), np.eye(GN, k=-G if N > 1 else 0)
    M1[:G,:G], M2[:G,:G] = np.dot(C, R), \
        - (g[0] * Tm1R + np.identity(G) / 3.)
    
    TmnR = Tm1R
    for i in range(N-2):
        idx = np.arange(G) + G * (i + 2)
        TmnR = np.dot(Tm2, TmnR)
        M1[:G, idx] = g[i + 1] * TmnR
    
    if N > 1:
        TmnR = np.dot(Tm2, TmnR)
        M2[:G,-G:] = -g[-1] * TmnR
    
    if nb_eigs is None:
        # get all eigen-pairs
        B2, flx = scipy.linalg.eig(M1, M2, check_finite=False)
        flx = flx[:G,:]
    else:
        if nb_eigs == 1:
            B2, flx = power_iteration(np.dot(np.linalg.inv(M1), M2))
            B2, flx = 1. / B2, flx[:G]
            if np.all(flx < 0):
                flx *= -1
            if not isfundamental(flx):
                lg.debug("flx: " + str(flx))
                raise RuntimeError(
                    'Flux for B2=%13.6g is not fundamental' % B2)
        elif nb_eigs > 1:
            raise RuntimeError('Not available yet.')
            # use ARPACK
            B2, flx = scipy.sparse.linalg.eigs(M2, M=M1, k=nb_eigs,
                                               which='SR')
    return B2, flx


def find_B2_asymptotes(xs, cs=coefs, check_asymptotes=True):
    "Find the B2 asymptotes leading to infinite k (on the real axis)."
    B2_asympts, flx = find_B2_spectrum(xs, g=cs, one_over_k=0.)
    real_B2_asympts = extract_real_elements(B2_asympts)
    flx = flx[:, np.isclose(B2_asympts.imag, 0)]
    idx = abs(real_B2_asympts).argsort()
    real_B2_asympts, flx = real_B2_asympts[idx], flx[:, idx]
    if check_asymptotes:
        st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
        G = st.size
        lg.debug("Real asymptotes: " + str(real_B2_asympts))
        for b2 in real_B2_asympts:
            det_R = np.linalg.det(get_Hr(st, ss, b2))
            lg.debug("det(R(B2 = {:<+13.6g})) = {:>13.6e}".format(b2,
                                                                 det_R))
            np.testing.assert_almost_equal(0, det_R,
                err_msg="B2=%f does not make R singular" % b2)
    return real_B2_asympts


def find_B2(xs, nb=1, c=coefs, root_finding=False, one_over_k=1.,
            with_approx=False):
    """Find the nb eigenvalues B2 as root of the degenerate system
    equations."""
    if root_finding:
        if with_approx:
            B2_asympts = find_B2_asymptotes(xs, c)
            # idx = B2_asympts.argsort() & (B2_asympts < 0)  # sort
            # B2_asympts = B2_asympts[idx]
            # input(B2_asympts)
            infplus = np.finfo(float).max
            k = 1. / one_over_k \
                if (one_over_k > 0) else infplus
            flx, B2, eps = None, np.zeros(nb), 1.e-10
            idx = abs(B2_asympts).argmin()
            B2l, B2r = B2_asympts[idx] + 1.e-4, 1e+6
            # WARNING: this asymptotes are not the ones of the original
            # problem because of the rational approximation
            for i in range(nb):
                B2[i] = opt.brentq(compute_deltak, B2l, B2r,
                                   args=(xs, k, gamma))
                print('Search segment is [%g, %g], eig nb. %3d = %g' % 
                      (B2l, B2r, i + 1, B2[i]))
                B2l, B2r = B2_asympts[i+1] + eps, B2_asympts[i] - eps
        else:
            # Inverse iterations to solve the non-linear eigen-problem
            G = xs[0].size  # nb. of groups from st
            toll = 1.e-6
            flx, B2, err_B2, it = np.ones(G), 0, 1.e+20, 0
            v = np.full(G, 1 / np.sum(flx))
            print("{:^5s}{:^13s}{:^13s}{:^13s}".format(
                'Its.', 'B2', 'B2-err', 'flx-err'))
            
            deriv = lambda x, eps=1.e-7: (
                get_T0(xs, x, one_over_k) - 
                get_T0(xs, x + eps, one_over_k) ) / eps
            
            while abs(err_B2) > toll:
                it += 1
                flx_old, B2_old = flx, B2
                # C = deriv(B2_old)
                # D = get_Tprime(xs, B2_old, one_over_k)
                # print('C', C)
                # print('D', D)
                # input('ok')
                # Hprime = get_Tprime(xs, B2_old, one_over_k)
                Hprime = deriv(B2_old)
                flx = np.linalg.solve(
                    get_T0(xs, B2_old, one_over_k),
                    np.dot(Hprime, flx)
                    )
                wnorm = np.dot(v, flx)
                B2 -= np.dot(v, flx_old) / wnorm
                flx /= wnorm
                err_B2 = B2_old - B2
                if abs(B2) > 0:
                    err_B2 /= B2
                err_flx = max(abs(1 - flx_old / flx))
                print(("{:>4d} " + 3*"{:13.6}").format(
                    it, B2, err_B2, err_flx)); input('wait...')
    else:
        B2, flx = find_B2_spectrum(xs, one_over_k, nb_eigs=nb, g=c)
    return B2, flx


if __name__ == "__main__":

    # verify the implementation
    st = np.array([0.52610422, 1.25161422])
    ss = np.zeros((2, 2, 2),)
    ss[0,:,:] = [[0.5002712, 0.00175241], [0.01537974, 1.14883323]]
    ss[1,:,:] = [[0.66171011, 0.00236677], [0.01330484, 0.91455725]]
    nsf = np.array([0.00757154, 0.15298673])
    chi = np.array([1., 0.])
    xs = (st, ss, chi, nsf)
    kinf, flx_inf = compute_kpairs(xs)
    np.testing.assert_almost_equal(kinf, 1.1913539017168697, decimal=7,
        err_msg="kinf not verified.")
    np.testing.assert_allclose(flx_inf, [39.10711218,  5.85183328],
        err_msg="fundamental flx_inf not verified.")
    adj_kinf, adj_flx_inf = compute_kpairs(xs, adjoint=True)
    np.testing.assert_almost_equal(adj_kinf, 1.1913539017168697, decimal=7,
        err_msg="adjoint kinf not verified.")
    np.testing.assert_allclose(adj_flx_inf, [1.1913539, 1.50878553],
        err_msg="fundamental adjoint flx_inf not verified.")
    np.testing.assert_almost_equal(find_B2(xs, c=coefs[:3])[0],
        0.004184657328394975, decimal=16, err_msg="B2 not verified.")
