#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module implemets the homogeneous B1 flux calculation in the infinte
(homogeneous) medium. The multigroup formalism in energy is used. The
methodology follows stricly Hebert's textbook. This module assumes that the
macroscopic cross sections are here provided as known data. The unknowns are
the multigroup neutron flux and the buckling B**2, which is the eigenvalue.
We solve this non-linear eigenvalue problem by a root-finding method.

References
==========

1. Hébert A., Applied reactor physics. Presses Inter. Polytechnique (2009).
"""
import os
import logging as lg
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.optimize as opt

__title__ = "MG Homog B1 calculation"
__author__ = "D. Tomatis, Y. Wang"
__date__ = "28/03/2019"
__version__ = "1.0.0"

# Ch. 4, pp 236, Eq. (4.127) from Hebert's textbook
g1, g2, g3 = 4. / 15., - 12. / 175., 92. / 2625.
gamma = lambda x: 1 + g1 * x + g2 * x**2 + g3 * x**3


def extract_real_elements(v):
    return v[np.isclose(v.imag, 0)].real


def isfundamental(flx):
    "Check if the input flux is the fundamental one."
    return ((flx > 0).all() or (flx < 0).all())


def get_R(st, ss, B2, adjoint=False, g=gamma):
    "Compute the B2-dependent removal term."
    R = get_T(st, ss, B2, gamma_func=g)
    R = B2 * np.linalg.inv(R) - ss[0,:,:]
    np.fill_diagonal(R, st + R.diagonal())
    return (R.conj().T if adjoint else R)


def get_T(st, ss, B2, gamma_func=gamma):
    "Compute the B2-dependent removal term."
    w = 3. * gamma_func(B2 / st**2) * st
    R = - np.array(ss[1,:,:], copy=True)
    np.fill_diagonal(R, w + R.diagonal())
    return R


def compute_deltak(B2, xs=None, k=1):
    "Compute the k eigenvalue corresponding to the input B2."
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    k_B2, flx = compute_kpairs(xs, B2)
    if chi.ndim == 1:
        dk = k_B2 - k
    else:
        dk = k_B2.max() - k
    return dk


def compute_kpairs(xs, B2=0., adjoint=False, g=gamma):
    "Compute the multiplication factor eigenpairs using the input B2."
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    # get the fundamental eigenpair assuming a single family of fissiles
    if adjoint:
        chi, nsf = nsf.T, chi.T
    flx = np.dot(np.linalg.inv(get_R(st, ss, B2, adjoint, g)), chi)
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


def find_B2_spectrum(xs, one_over_k=1., nb_eigs=None, g=(g1, g2, g3)):
    "Find the B2 asymptotes leading to infinite k."
    g1, g2, g3 = g
    st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
    A, C = - np.array(ss[0,:,:], copy=True), \
           - np.array(ss[1,:,:], copy=True) / 3.
    np.fill_diagonal(A, st + A.diagonal())
    if abs(one_over_k) > 0:
        if chi.ndim == 1:
            ChiF = np.outer(chi, nsf)
        else:
            ChiF = np.dot(chi, nsf)
        A -= one_over_k * ChiF
    np.fill_diagonal(C, st + C.diagonal())
    Sm1, G = 1. / st, st.size
    Sm2 = np.diag(Sm1**2)
    A1 = np.dot(np.diag(Sm1), A)
    A2 = np.dot(Sm2, A1)
    A3 = np.dot(Sm2, A2)
    a1, a2, a3, a4 = np.dot(C, A), g1 * A1, g2 * A2, g3 * A3
    a2 += np.identity(G) / 3.
    G2, G3 = G * 2, G * 3
    M1, M2 = np.identity(G3), np.eye(G3, k=-G)
    M1[:G,:G], M1[:G,G:G2], M1[:G,G2:], M2[:G,G2:] = a1, a2, a3, -a4
    # M = np.dot(np.linalg.inv(M2), M1)
    # return np.linalg.eigvals(M)
    if nb_eigs is None:
        # get all eigen-pairs
        B2, flx = scipy.linalg.eig(M1, M2, check_finite=False)
    else:
        if nb_eigs == 1:
            B2, flx = power_iteration(np.dot(np.linalg.inv(M1), M2))
            B2, flx = 1. / B2, flx[:G]
            if (flx < 0).all():
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


def find_B2_asymptotes(xs, check_asymptotes=True):
    "Find the B2 asymptotes leading to infinite k (on the real axis)."
    B2_asympts, flx = find_B2_spectrum(xs, one_over_k=0.)
    real_B2_asympts = extract_real_elements(B2_asympts)
    flx = flx[:, (B2_asympts.real() == real_B2_asympts) and
                 np.isclose(B2_asympts, 0)]
    if check_asymptotes:
        st, ss, chi, nsf = xs  # unpack the macroscopic cross sections
        G = st.size
        lg.debug("Real asymptotes: " + str(real_B2_asympts))
        for b2 in real_B2_asympts:
            det_R = np.linalg.det(get_R(st, ss, b2))
            lg.debug("det(R(B2 = {:<+13.6g})) = {:>13.6e}".format(b2, det_R))
            np.testing.assert_almost_equal(0, det_R,
                err_msg="B2=%f does not make R singular" % b2)
    return real_B2_asympts


#def find_B2(xs, B2M=1., k=1., nb=1):
def find_B2(xs, nb=1):
    "Find the nb eigenvalues B2 as root of the degenerate system equations."
    # B2_asympts = find_B2_asymptotes(xs)
    # B2 = opt.brentq(compute_deltak, B2_asympts.max() + 1.e-10,
    #                 B2M, args=(xs, k))
    B2, flx = find_B2_spectrum(xs, nb_eigs=nb)
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
    np.testing.assert_almost_equal(find_B2(xs)[0], 0.004184657328394975,
        decimal=7, err_msg="B2 not verified.")
