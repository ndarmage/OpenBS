#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module contains the definitions of the objects used to build the Bateman
system of ODEs. The neutron flux is determined in the multigroup infinite
homogeneous medium by the B1 leakage model as in [2].

.. todo::
   Choose/study the ODE solver.

.. todo::
   Implement the saturation model.

.. todo::
   Interpolate the cross sections on the variables in the state vector p.

.. note::
   The test implemented below computes the contribution functions as described
   in [1] in case of a transient featuring the physics of xenon poisoning in a
   nuclear reactor.

References
----------

1. Go Chiba et al., Important fission product nuclides identification method
   for simplified burnup chain construction, Journal of Nuclear Science and
   Technology, Taylor & Francis, Vol. 52, Nos. 7--8, pp. 953--960 (2015).
2. Hébert A., Applied reactor physics. Presses Inter. Polytechnique (2009).
"""
import os
import sys
import time
import logging as lg
import numpy as np
import xarray as xr
import scipy.integrate as ni
from scipy.constants import electron_volt as eV2J

# import the module for the flux calculation
from HomogB1FlxCalc import *

# import the tools to load the nuclide chain
from chaintools import *

import multiprocessing

__title__ = "OpenBS"
__author__ = "D. Tomatis, Y. Wang"
__date__ = "26/03/2019"
__version__ = "1.5.0"

logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
lg.basicConfig(level=lg.INFO)  # filename = logfile

MeV2J = eV2J * 1.e+6  # MeV to J conversion constant
barn2cm2 = 1.e-24  # barn to cm2 conversion constant
ThEnCut = 0.625  # thermal energy cut in eV

NB_CORES = min(multiprocessing.cpu_count() -1
               , 24)  # may fail on ARM arch.


def list_sum(l1, l2):
    "return the sum of two lists"
    return sorted(set(l1) | set(l2))


def list_intersect(l1, l2):
    "return the intersection of two lists"
    return sorted(set(l1) & set(l2))


def list_diff(l1, l2):
    "return the difference of two lists (not commutative!)"
    return sorted(set(l1) - set(l2))


def macroxs(NAr, mxslib, p, xstype='Absorption'):
    """Compute the macroscopic xs with the input list of nuclides nnames and
    concentrations in N by the microlib dict storing the microscopic xs at
    the state parameter vector p."""
    if not isinstance(p, tuple):
        raise ValueError('input p is not tuple!')

    N, nnames = NAr.values, NAr.Nname.values
    microlib, ng, ao = mxslib

    if 'Scattering' in xstype:
        xshape = ng, ng, ao
    else:
        xshape = ng
    xs = np.nan_to_num(np.array([microlib[n][xstype][p]
        if xstype in microlib[n] else np.zeros(xshape) for n in nnames]
    ))
    if 'Spectrum' in xstype:
        if ng == 2:
            xs = np.array([1., 0.])
        else:
            lg.warn("Consider dealing with many fissile families yet.")
            raise RuntimeError("Many fissile families not available yet.")
    else:
        xs = np.tensordot(N, xs, 1)  # instead of np.dot(N, microxs)
    return xs


def get_macroxs(Nar, xslib, p):
    "Return all macroscopic xs from the (MPO) xslib at the point p."
    ss = macroxs(Nar, xslib, p, 'Scattering')
    st = macroxs(Nar, xslib, p, 'Total')
    chi = macroxs(Nar, xslib, p, 'FissionSpectrum')
    nsf = macroxs(Nar, xslib, p, 'NuFission')
    return st, ss, chi, nsf


def get_Energy_macroxs(Nar, xslib, p):
    "Return the macroxs of energy production."
    # the units of the microxs are MeV.barn
    sk = macroxs(Nar, xslib, p, 'FissionEnergyFission')
    sk += macroxs(Nar, xslib, p, 'CaptureEnergyCapture')
    return sk


def get_concentrations(microxs, p, evolving_nuclides, nuclides_in_chain=True):
    """Retrieve the concentrations of the nuclides present in the chain
    (or not, depending on nuclides_in_chain) from the xslib MPO data."""
    if not isinstance(p, tuple):
        raise ValueError('The input state vector p is not tuple.')
    if nuclides_in_chain:
        nuclide_list = evolving_nuclides
        Nname = 'EvolNuclideConcs'
    else:
        # nuclide_list = list_diff(microxs.keys(), evolving_nuclides)
        nuclide_list = list_diff([*microxs], evolving_nuclides)
        Nname = 'NotEvolNuclideConcs'
    # for n in nuclide_list:
    #     print (n, microxs[n]['conc'][p])
    N = np.array([microxs[n]['conc'][p] for n in nuclide_list])
    # nan concentrations (i.e. missing in N) are replaced by zeros
    return xr.DataArray(np.nan_to_num(N), name=Nname, dims=('Nname'),
                        coords={'Nname': nuclide_list})


def compute_flux(Nar, mxslib, p, P=0., B2eigv=True, vrbs=True):
    """Calculate the multigroup spectrum of the volume-integrated flux. Cross
    sections are retrieved from the MPO library at the point p. The output flux
    is normalized to have the input power P (W/cm)."""
    # calculate the macroscopic cross sections for the flux calculation
    xs = get_macroxs(Nar, mxslib, p)
    kinf, flx = compute_kpairs(xs)
    if vrbs:
        lg.debug("fundamental k-eigenpair: " + str([kinf, flx]))
    eigv = kinf

    # call the flux calculation (with the homogeneous B1 leakage model)
    if B2eigv:
        B2, flx = find_B2(xs)
        nsf, eigv = xs[-1], B2
        C = kinf / np.dot(nsf, flx)
        if vrbs:
            lg.debug("fundamental B2-eigenpair: " + str([B2, flx * C]))

    if np.any(flx < 0.):
        raise RuntimeError("Negative flux detected " + str(flx))

    # normalize the flux to fulfill the other power P
    sk = get_Energy_macroxs(Nar, mxslib, p)
    flx *= P / np.dot(sk * MeV2J, flx)
    if vrbs:
        lg.debug("Required normalization at P = {:8.3f} (W/cm)".format(P))
        lg.debug('En. production xs: ' + str(sk))
        lg.debug('Normalized flux: ' + str(flx))
        lg.debug('Power per group (W/cm): ' + str(sk * MeV2J * flx))
    return eigv, flx


def build_N_DataArray(N, Nnames, NArNot, xname='EvolNuclideConcs'):
    NAr = xr.DataArray(N, name=xname, dims=('Nname'), coords={'Nname': Nnames})
    return xr.concat([NAr, NArNot], dim='Nname')


def compute_flx_derivatives(N, evol_names, Nnot, flx_argvs, eps=1.e-3):
    """Calculate the derivatives of the flux on the nuclide concentrations in
    N by estimating the limit of the incremental ratio. These derivatives are
    stored in the output matrix dflx_dN, with derivatives per energy group by
    columns and per every input concentration by row. The derivatives of the
    reactivity is also determined on the way. The unperturbed state is also
    calculated in this function. *eps* is the perturbation on each
    concentration and it is given as a relative quantity. Nnot contains the
    concentrations of the nuclides that do not evolve in time."""
    mxslib, p, P, B2eigv = flx_argvs  # unpack
    if B2eigv:
        raise ValueError("B2eigv is not supported yet!")

    # compute the unperturbed state
    NArAll = build_N_DataArray(N, evol_names, Nnot)
    k0, flx0 = compute_flux(NArAll, *flx_argvs, vrbs=False)

    microxs, NG, AO = mxslib

    nb_evolving_nuclides = len(N)
    drho_dN = np.zeros(nb_evolving_nuclides)
    dflx_dN = np.zeros((nb_evolving_nuclides, NG),)
    Delta_N = N * eps
    eps = 1.0 + eps
    for j in range(nb_evolving_nuclides):
        N_copy = np.array(N, copy=True)
        N_copy[j] *= eps
        # get the DataArray
        NArAll = build_N_DataArray(N_copy, evol_names, Nnot)
        k, flx = compute_flux(NArAll, *flx_argvs, vrbs=False)

        drho_dN[j] = (1. / k0 - 1. / k) / Delta_N[j]
        dflx_dN[j,:] = (flx - flx0) / Delta_N[j]
    return k0, flx0, drho_dN, dflx_dN


def dN_dt(N, t, p=None, P=0., V=0., NArNot=None, nucl_chain=None, mxslib=None,
          evol_names=None, flx=None, adjoint=False):
    """Determine the change rates of the nuclides present in the array N at
    time t, that is setup the product between the evolution matrix and the
    same vector N as in the system of the Bateman equations. The neutron flux
    (n/cm2/s) is determined hereafter if flx is None. In this case, it is
    normalized to yield the input power P (W/cm). An input flux is requested
    when solving for the adjoint vector of nuclide concentrations, and it
    cannot change during the time step of integration. The order of the
    nuclides in N is given by the sorted method below. The dict with all
    nuclide instances must be provided."""
    # The warning messages related to missing xs data have been changed to
    # lg.debug to limit the output printing with level=lg.INFO.
    lg.debug(" *******************")
    lg.debug(" *** ENTER dN_dt ***")
    lg.debug(" *******************")
    lg.debug("Compute derivatives of nuclide concentrations at t=%f" % t)
    NArAll = build_N_DataArray(N, evol_names, NArNot)
    # N, evol_names = NAr.values, NAr.coords['Nname'].values
    Nnot, not_evol_names = NArNot.values, NArNot.coords['Nname'].values

    # unpack data
    microxs, NG, AO = mxslib             #anisotropy order
    nuclides, yields = nucl_chain

    if flx is None:
        eigv, flx = compute_flux(NArAll, mxslib, p, P, HB1LM)
    elif flx.size != NG:
        raise ValueError('Input flux density with wrong nb. of groups.')
    nflx = flx / V
    # input('press to continue...')

    # consider yields of the only evolvnig nuclides
    FP, FF = yields.coords['FissionProducts'].values, \
             yields.coords['FissileNuclides'].values
    evol_FP, evol_FF = np.array(list_intersect(evol_names, FP)), \
                       np.array(list_intersect(evol_names, FF))
    gammas = yields.sel(FissionProducts=evol_FP, FissileNuclides=evol_FF).values

    # get the concentrations of the fissile nuclides if direct problem or
    # fission products if adjoint
    if adjoint:
        Nconcs, evol_FX = NArAll.sel(Nname=evol_FP).values, evol_FF
        get_channels = lambda nuc_obj: nuc_obj.removal_channels
    else:
        Nconcs, evol_FX = NArAll.sel(Nname=evol_FF).values, evol_FP
        get_channels = lambda nuc_obj: nuc_obj.buildup_channels
    # get the matrix with macroscopic fission cross sections of FF nuclides
    FFsf = np.zeros((len(evol_FF), NG),)
    for i, ff in enumerate(evol_FF):
        FFsf[i, :] = np.nan_to_num(microxs[ff]['Fission'][p])
    fission_rates = np.dot(FFsf, np.diag(nflx))

    # gammas_{j in  FP, i in FF, g} element-wise * by fission_rates_{i,g} and
    # sum on groups (last axis)
    gji = np.sum(gammas * fission_rates, axis=-1)
    FXbyFYields = np.dot(Nconcs, gji) if adjoint else np.dot(gji, Nconcs)
    FXbyFYields *= barn2cm2

    # setup the right hand side of the Bateman system equations
    f = np.zeros((len(N)),)
    for i, n in enumerate(evol_names):
        nuclide = nuclides[n]
        if 'Absorption' in microxs[n]:
            # nuclides without absorption with CEAV6: He3, He4 and H3
            mRRate = np.dot(np.nan_to_num(microxs[n]['Absorption'][p]), nflx)
        else:
            mRRate = 0.
            lg.debug("Missing absorption xs for nuclide " + str(n))
        f[i] = -(nuclide.lmbda + mRRate * barn2cm2) * N[i]

        for other, chtype, bratio in get_channels(nuclide):
            Nconc = N[evol_names == other]
            # other is a son if adjoint, a parent otherwise
            parent = n if adjoint else other
            if 'DRTYP' in chtype:
                mRRate = nuclides[parent].lmbda
            elif 'REAMT' in chtype:
                if ('MT16' in chtype) or ('016' in chtype):
                    # (n, 2n): the xs is approximated by the nxcess xs
                    if 'Nexcess' in microxs[parent]:
                        mgRRate = np.nan_to_num(microxs[parent]['Nexcess'][p])
                    else:
                        #                   ...to be removed after fix of CEAV6
                        # ISSUES RELATED TO THE USE OF CEAV6
                        lg.debug("Parent " + parent + " of " +
                                 (other if adjoint else n) +
                                 " has no nexcess xs\n" + str(nuclide))
                        continue
                elif '102' in chtype:
                    # (n, gamma): radiative capture xs approximated by the
                    # difference between absorption and fission
                    # TEMPORARY FIX - BE AWARE THAT SHOULD NOT HAPPEN
                    # THIS ISSUE COMES WITH CEAV6
                    #                       ...to be removed after fix of CEAV6
                    if 'He3' or 'He4' in parent: continue
                    # He3 queried in the direct problem and He4 in the adjoint
                    mgRRate = np.nan_to_num(microxs[parent]['Absorption'][p])
                    if 'Fission' in microxs[parent]:
                        mgRRate -= np.nan_to_num(microxs[parent]['Fission'][p])
                elif '107' in chtype:
                    # (n, alpha): alpha emission of light nuclides
                    # currently Unavailable
                    raise RuntimeError(chtype + "not available yet")
                else:
                    raise RuntimeError("Unavailable or unknown " + chtype)
                mRRate = np.dot(mgRRate, nflx) * barn2cm2
            else:
                raise ValueError('Unknown type of buildup channel ' + chtype)

            f[i] += bratio * Nconc * mRRate

        # add Yields
        isFX = True if n in evol_FX else False
        if isFX:
            posFX = evol_FX == n
            if np.sum(posFX) > 1:
                raise RuntimeError('too many occurrences in posFX.')
            f[i] += FXbyFYields[posFX]

    # return xr.DataArray(f, name='NDerivatives', dims=('Nname'),
    #                     coords={'Nname': evol_names})
    lg.debug(" ******************")
    lg.debug(" *** EXIT dN_dt ***")
    lg.debug(" ******************")
    # ofile = 'append_f.txt'
    # if os.path.isfile(ofile):
    #     fold = np.loadtxt(ofile, delimiter=',')
    #     np.savetxt(ofile, np.vstack((fold,f.T)), fmt='%13.6e', delimiter=',')
    # else:
    #     np.savetxt(ofile, f.T, fmt='%13.6e', delimiter=',')

    # if adjoint one should return -f, but the adjoint problem is later solved
    # backwards in time, that is by a change of variable from the time t to -t.
    return f

import sys

def dM_dNj_dot_N(N,p=None, evol_names=None, NArNot=None, nucl_chain=None,
                 mxslib=None, djflx=None):                   #another variable p has been added inside the definition of function dM_dNj_dot_N
    """Calculate the scalar product between the derivative of the evolution
    matrix M on the j-th nuclide concentration in N and N itself. This follows
    from calculating the \Delta M term in the perturbation ODEs, for
    \[ dM / dN \cdot N = \sum_j {(dM / \phi) (\phi / dN_j) \cdot N}, \]
    where N is the vector valued array of nuclide concentrations and $\phi$ is
    the neutron flux density. $dM / d\phi$ contains only the cross sections of
    the reaction rates (with yields in case). The derivative of the flux is
    given in input. Remind that it refers only to the j-th nuclide in N.
    This function is obtained from dN_dt.
    """
   # print("SIZE N", N.nbytes)
   # print("SIZE p",sys.getsizeof(p))
   # print("SIZE dj",sys.getsizeof(djflx))
   #print("SIZE mx",sys.getsizeof(mxslib))
   #print("SIZE nuc",sys.getsizeof(nucl_chain))
   # print("SIZE Nar",sys.getsizeof(NArNot))
   #print("SIZE evol",sys.getsizeof(evol_names))
    NArAll = build_N_DataArray(N, evol_names, NArNot)
    # N, evol_names = NAr.values, NAr.coords['Nname'].values
    Nnot, not_evol_names = NArNot.values, NArNot.coords['Nname'].values

    # unpack data
    microxs, NG, AO = mxslib
    nuclides, yields = nucl_chain

    # consider yields of the only evolvnig nuclides
    FP, FF = yields.coords['FissionProducts'].values, \
             yields.coords['FissileNuclides'].values
    evol_FP, evol_FF = np.array(list_intersect(evol_names, FP)), \
                       np.array(list_intersect(evol_names, FF))
    gammas = yields.sel(FissionProducts=evol_FP, FissileNuclides=evol_FF).values

    Nconcs, evol_FX = NArAll.sel(Nname=evol_FF).values, evol_FP
    # get the matrix with macroscopic fission cross sections of FF nuclides
    FFsf = np.zeros((len(evol_FF), NG),)
    for i, ff in enumerate(evol_FF):
        FFsf[i, :] = np.nan_to_num(microxs[ff]['Fission'][p])
    fission_rates = np.dot(FFsf, np.diag(djflx))

    # gammas_{j in  FP, i in FF, g} element-wise * by fission_rates_{i,g} and
    # sum on groups (last axis)
    gji = np.sum(gammas * fission_rates, axis=-1)
    FXbyFYields = np.dot(gji, Nconcs) * barn2cm2


    # setup the right hand side of the Bateman system equations
    f = np.zeros((len(N)),)
    for i, n in enumerate(evol_names):
        nuclide=nuclides[n]                    #verify and check condition of vector p to function dM_dN (this has been verified)
        if 'Absorption' in microxs[n]:
            # nuclides without absorption with CEAV6: He3, He4 and H3
            mRRate = np.dot(np.nan_to_num(microxs[n]['Absorption'][p]), djflx)
            f[i] = - mRRate * barn2cm2 * N[i]
        # else:
        #     lg.debug("Missing absorption xs for nuclide " + str(n))

        for parent, chtype, bratio in nuclide.buildup_channels:
            # N.B.: The derivative of M does not contain decay constants
            if 'REAMT' in chtype:
                if ('MT16' in chtype) or ('016' in chtype):
                    # (n, 2n): the xs is approximated by the nxcess xs
                    if 'Nexcess' in microxs[parent]:
                        mgRRate = np.nan_to_num(microxs[parent]['Nexcess'][p])
                    else:
                        #                   ...to be removed after fix of CEAV6
                        # ISSUES RELATED TO THE USE OF CEAV6
                        lg.debug("Parent " + parent + " of " + n +
                                 " has no nexcess xs\n" + str(nuclide))
                        continue
                elif '102' in chtype:
                    # (n, gamma): radiative capture xs approximated by the
                    # difference between absorption and fission
                    # TEMPORARY FIX - BE AWARE THAT SHOULD NOT HAPPEN
                    # THIS ISSUE COMES WITH CEAV6
                    #                       ...to be removed after fix of CEAV6
                    if 'He3' or 'He4' in parent: continue
                    # He3 queried in the direct problem and He4 in the adjoint
                    mgRRate = np.nan_to_num(microxs[parent]['Absorption'][p])
                    if 'Fission' in microxs[parent]:
                        mgRRate -= np.nan_to_num(microxs[parent]['Fission'][p])
                elif '107' in chtype:
                    # (n, alpha): alpha emission of light nuclides
                    # currently Unavailable
                    raise RuntimeError(chtype + "not available yet")
                else:
                    raise RuntimeError("Unavailable or unknown " + chtype)

                Nconc = N[evol_names == parent]
                mRRate = np.dot(mgRRate, djflx) * barn2cm2
                f[i] += bratio * Nconc * mRRate

        # add Yields
        isFX = True if n in evol_FX else False
        if isFX:
            posFX = evol_FX == n
            if np.sum(posFX) > 1:
                raise RuntimeError('too many occurrences in posFX.')
            f[i] += FXbyFYields[posFX]

    return f


if __name__ == "__main__":

    # read the nuclide chain
    lg.info(' -o-'*16)
    #nchain_file = "lib/Chain_fuel.CEAV6"
    #nchain_file = "lib/DecayData_CEAV6.h5"
    #nchain_file = "lib/DecayData_CEAV2005_V3.h5"
    nchain_file = "lib/DepletionData_CEAV6_CAPTN2N.h5"
    nchain, yields = load_nucl_chain(nchain_file, False)
    # add removal channels from a nuclide to its sons (for adj. Bateman eqs.)
    for nuclide in nchain.values():
        nuclide.add_removal_channels(nchain)
        # input(str(nuclide))
    # nuclides_in_chain = sorted(nchain.keys())  # fix the order of nuclides
    nuclides_in_chain = sorted([*nchain])  # fix the order of nuclides
    lg.info("Data correctly retrieved from the nuclide chain " + nchain_file)
    lg.info("There are %d nuclides with buildup channels." %
            len(nuclides_in_chain))
    lg.info("There are %d fissile nuclides with yields." %
            len(yields.FissileNuclides.values))
    lg.info("There are %d fission products produced by yields." %
            len(yields.FissionProducts.values))
    lg.info(' -o-'*16)

    # retrieve the cross sections prepared by ap3
    sys.path.append('include/')
    from analyse_NData import readMPO
    xsdir = "lib/"
    MPOh5 = "UO2_325_AFA3G17_idt_2G_noB2.hdf"
    # xsdir = "/data/tmplca/dtomatis/ReducedDeplChains/pwruo2_fc_281gxs/"
    # MPOh5 = "UO2_325_AFA3G17_idt_281G.hdf"
    xslib = readMPO(xsdir + MPOh5, vrbs=True, load=False, check=True,
                    save=False)
    lg.info("Data correctly retrieved from the MPO " + MPOh5)
    lg.info(' -o-'*16)
    # sys.exit("END of TEST")

    homogenized_zone = 0  # there is only a single zone of homogenization
    microxs = xslib['microlib'][homogenized_zone]
    # nb of en. groups and anisotropy order
    NG, AO, Vcm2 = xslib['NG'], xslib['ANISOTROPY_PL1'], \
                   xslib['ZONEVOLUME'][homogenized_zone]
    microxst = microxs, NG, AO  # a packed tuple used later

    # Calculate the projection matrix for different En. bounds between
    # fission yields and the neutron mutligroup flux
    EnMesh_MPO = [xslib[key] for key in [*xslib] if ("EnMesh_" in key)]
    EnMeshFlx_MPO = [xslib[key] for key in [*xslib] if ("EnMesh_" in key) and
                     not ("xslib_MIC" in key)][0]

    YgG = compute_EnYconversion(EnMeshFlx_MPO,
                                np.append(yields.YieldEnergyBounds.values,
                                EnMeshFlx_MPO[-1]))
    # convert yields onto the flux energy mesh
    yields.values = np.tensordot(yields.values, YgG, 1)
    chain_data = nchain, yields

    # check that the number of nuclides in the xslib is the same as the one
    # in the chain
    # nb_nuclides_in_MPO = len(microxs.keys())
    nuclides_in_MPO = [*microxs]
    nb_nuclides_in_MPO = len(nuclides_in_MPO)
    if nb_nuclides_in_MPO == 1:
        lg.critical("This MPO file contains only one macroscopic material.")
    if nb_nuclides_in_MPO != len(nchain):
        lg.info("Nb. of nuclides in libs mismath")
        lg.info("%d nuclides in MPO" % nb_nuclides_in_MPO +
                   " != %d nucl. in chain" % len(nchain))
        lg.info("The missing nuclides in the chain will not evolve in time.")
    evol_nucl_diff = list_diff(nuclides_in_chain, nuclides_in_MPO)
    # TEMPORARY FIX - the following should not happen
    if len(evol_nucl_diff) != 0:
        lg.warning("There are nuclides in chain missing in the MPO:")
        lg.warning(str(evol_nucl_diff))
        if evol_nucl_diff[-1] == 'He3':
            nchain['He3'] = nuclide('He3', 0., 0)
        if len(evol_nucl_diff) > 1:
            raise RuntimeError('Unknown major error')
        nuclides_in_chain = sorted([*nchain])

    # prepare the initial nuclide density vector N
    Bu_pnt, TMod_pnt, Plin_pnt = 5, 0, -1
    Bu, TMod, Plin = xslib['BURN'], xslib['TMod'], xslib['Plin']
    PWcm = Plin[Plin_pnt]
    r_f = 0.4096  # outer fuel radius (cm) from ap3 input file
    cell_pitch = 1.25882  # cell ptich (cm) from ap3 input files
    Af, cell_area = np.pi * r_f**2, cell_pitch**2
    if not np.isclose(cell_area, Vcm2):
        raise ValueError("The homogenization volume differs from user data.")
    # thermal power density by nuclear reactions in the fuel pin (W/cm3)
    PWcm3 = PWcm  # / cell_area  # instead of Af
    lg.info("Use the xs from MPO at\n Bu = %.1f MWd/t, " % Bu[Bu_pnt] +
            "TMod = %.2f K, " % TMod[TMod_pnt] +
            "Plin = %.2f W/cm." % Plin[Plin_pnt])

    # state parameters vector
    p = (Bu_pnt, Bu_pnt, Plin_pnt, TMod_pnt, Plin_pnt)
    # B2eigv is used as a global variable in the called functions
    HB1LM = not(np.isclose(xslib['kinf'][p], xslib['keff'][p]))
    lg.debug("Homog. B1 leakage model in input MPO lib? " + str(HB1LM))

    # get the initial evolving nuclides
    # warning, there are nuclides present in the chain but missing in the MPO
    evolving_nuclides = list_intersect(nuclides_in_MPO, nuclides_in_chain)
    nb_N = nb_evolving_nuclides = len(evolving_nuclides)
    N0 = get_concentrations(microxs, p, evolving_nuclides)
    # get the initial not-evolving nuclides
    N0not = get_concentrations(microxs, p, evolving_nuclides, False)

    # change the state point to have different conditions during the transient
    p = list(p)
    p[2] = p[4] = 1
    p = tuple(p)

    # compute the normalization power
    N0all = xr.concat([N0, N0not], dim='Nname')
    sk = get_Energy_macroxs(N0all, microxst, p)  # MeV/cm
    zp = tuple(np.append(homogenized_zone, p))
    PWcm_xslib = np.dot(sk * MeV2J, xslib['ZoneFlux'][zp])
    hmden = xslib['HEAVY_METAL_DENSITY'][zp]
    lg.info("Thermal linear power delivered by nuclear reaction is " +
            "{:5.2f} (W/cm)".format(PWcm_xslib))
    lg.info("Heavy Metal Density (w.r.t. homog volume) is " +
            "{:9.6g} (g/cm3)".format(hmden))
    lg.info("Heavy Metal Density (w.r.t.  fuel volume) is " +
            "{:9.6g} (g/cm3)".format(hmden / Af * cell_area))
    lg.info("Mass-specific power density is {:5.2f} (W/g)".format(
            PWcm_xslib / hmden / cell_area))
    lg.debug("Flux from xslib (n/s): " + str(xslib['ZoneFlux'][zp]))
    lg.debug("k-Sigma from xslib (MeV/cm): " + str(sk))
    lg.debug('Power per group (W/cm): ' +
             str(sk * MeV2J * xslib['ZoneFlux'][zp]))
    # input("...press a key to continue!")
    # solve the Bateman system equations
    # ofile = 'append_f.txt'
    # if os.path.exists(ofile):
    #     os.remove(ofile)
    C0, evol_names = N0.values, N0.Nname.values

    # -------------------------------------------------------------------------
    # -----------------set the calculation and its options---------------------
    # -------------------------------------------------------------------------
    calculate_reference_N = True  # load from existing file if False
    calculate_flux_and_derivatives = True  # load from existing file if False
    calculate_CF = True
    use_el_as_final_condition = True
    test_dir = 'tests/'  # directory storing the reference solutions
    # ofile += "RefSteadySolution.p"
    Nfile = test_dir + "Ref25pcPowerSolution.p"
    Ffile = test_dir + "Ref25pcFlxDerivatives.p"
    # -------------------------------------------------------------------------

    lg.info(' -o-'*16)
    import pickle
    if calculate_reference_N:
        lg.info("Calculate the reference solution by resolving the non-linear")
        lg.info("Bateman equations.")
        # define the time mesh
        tbeg, tend, Deltat_min = 0.,3600*24*4 ,5.
        tmesh =np.linspace(int(tbeg), int(tend),int ((tend - tbeg) / (Deltat_min * 60.) + 1))                                #it works with linspace(0,100,5)


        def dN_dt_wrapped(t, N):
            return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst,
                         nucl_chain=chain_data, p=p, V=Vcm2,
                         evol_names=evol_names, adjoint=False)

        Nsol = ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], C0, method="Radau",
                            t_eval=tmesh
                            )
        Nsol.evol_names, Nsol.N0Not = evol_names, N0not
        # Nsol = ni.odeint(dN_dt, C0, time, args=(p, PWcm_xslib, Vcm2, N0not,
        #                  microxst, chain_data, evol_names))

        lg.info("Save ODE solution to file " + Nfile)
        with open(Nfile, "wb") as f:
            pickle.dump(Nsol, f)
    else:
        lg.info("Retrieve the reference solution from previous calculation")
        lg.info("Ref. solution from file " + Nfile)
        with open(Nfile, "rb") as f:
            Nsol = pickle.load(f)
        tmesh = Nsol.t
        tbeg, tend, I = tmesh[0], tmesh[-1], tmesh.size-1
    lg.info(' -o-'*16)


    if calculate_flux_and_derivatives:
        lg.info("Calculate the flux and the derivatives with the solution of")
        lg.info("the nuclide concentrations in time.")
        t_beg = time.time()

        flx_argvs = microxst, p, PWcm_xslib, HB1LM

        # # serial calculation
         #flxt, kt = np.zeros((NG,I+1),), np.zeros((I+1),)
         #drho_dN = np.zeros((nb_evolving_nuclides,I+1),)
         #dflx_dN = np.zeros((nb_evolving_nuclides,NG,I+1),)
         #for i, Ni in enumerate(Nsol.y[:,:I+1].T):
         #        kt[i], flxt[:,i], drho_dN[:,i], dflx_dN[:,:,i] = \
         #        compute_flx_derivatives(Ni, evol_names, N0not, flx_argvs,
                                     #eps=1.e-2)
         #lg.info("Elapsed time for the derivatives (s): %13.6g" %
         #        (time.time() - t_beg))

        # # copy the arrays to verify the parallel calculation below
        # k0, flxt0, drho_dN0, dflx_dN0 = np.array(kt[:I+1], copy=True), \
        #                                np.array(flxt[:,:I+1], copy=True), \
        #                                np.array(drho_dN[:,:I+1], copy=True),  \
        #                                np.array(dflx_dN[:,:,:I+1], copy=True)
        #t_beg = time.time()  # restart the counter

        def compute_derivs_at_i(Ni):
            # compute the k-eigenvalue, the flux and the derivatives at the
            # beginning of the i-th time step with Ni
            # return ki, flxi, drho_dNi, dflx_dNi
            return compute_flx_derivatives(Ni, evol_names, N0not,
                                           flx_argvs, eps=1.e-2)

        pool = multiprocessing.Pool(NB_CORES)
        pool_results =pool.starmap(compute_flx_derivatives,
                                [(Ni, evol_names, N0not,flx_argvs,1.e-2) for Ni in Nsol.y[:,:tmesh.size-1+1].T])
        # pool.map(compute_derivs_at_i,[Ni for Ni in Nsol.y[:,:I+1].T]) was substitued with the latter starmap
        pool.close()
        pool.join()
        kt = np.array([res[0] for res in pool_results])
        flxt = np.column_stack([res[1] for res in pool_results])
        drho_dN = np.column_stack([res[2] for res in pool_results])
        dflx_dN = np.stack([res[3] for res in pool_results], axis=2)
        lg.info("Elapsed time for the derivatives (s): %13.6g" %
            (time.time() - t_beg))

        # # verify parallel and serial calculations
        # print(np.sum(k0 == kt) == (I+1))
        # print(np.sum(flxt0 == flxt) == NG*(I+1))
        # print(np.sum(drho_dN0 == drho_dN) == nb_N*(I+1))
        # print(np.sum(dflx_dN0 == dflx_dN) == NG*nb_N*(I+1))

        lg.info("Save the derivatives to file " + Ffile)
        with open(Ffile, "wb") as f:
            pickle.dump((kt, flxt, drho_dN, dflx_dN), f)

    else:
        lg.info("Retrieve the reference flux and the derivatives in time")
        lg.info("from file " + Ffile)
        with open(Ffile, "rb") as f:
            kt, flxt, drho_dN, dflx_dN = pickle.load(f)


    if calculate_CF:
        lg.info("Calculate the contribution function CFs.")
        lg.info("%d CPUs on this computer - " % multiprocessing.cpu_count() +
                "%d used hereafter in parallel calculations" % NB_CORES)

        # The Bateman equations need to be linearized in order to derive the
        # adjoint operator solvable by dN_dt. This means using a constant flux
        # for the reaction rates in the evolution matrix. Because of this
        # approximation suggested by Go Chiba et al., the transient is then
        # divided in many steps, which are treated separately.

        Na = np.zeros((nb_evolving_nuclides,nb_evolving_nuclides,tmesh.size-1),)
        Na_hat = np.zeros((nb_evolving_nuclides,nb_evolving_nuclides,tmesh.size-1+1),)
        CF = np.zeros_like(Na)
        # add the fictitious final condition of e_l to Na_hat
        Na_hat[:,:,-1] = np.identity(nb_evolving_nuclides)
        for i in range(tmesh.size-1-1,-1,-1):

            # locate the i-th time step
            ti, tip1 = tmesh[i], tmesh[i+1]
            Dt, tbnd = tip1 - ti, [ti, tip1]

            # get the concentrations at the beginning and at the end of the
            # given time step
            Ni, Nip1 = Nsol.y[:,i], Nsol.y[:,i+1]

            # get the flux and the derivatives at the beginning of the i-th
            # time step
            flxi, drhoi_dN, dflxi_dN = flxt[:,i], drho_dN[:,i], dflx_dN[:,:,i]

            def dM_dNj_dot_Ni(dflx):
                # N.B.: The following does not serve the purpose because it
                # includes the unwanted contributions of decay constants
                # return dN_dt(Ni, 0., NArNot=N0not, P=PWcm_xslib,
                #              mxslib=microxst, nucl_chain=chain_data,
                #              p=p, V=Vcm2, evol_names=evol_names,
                #              adjoint=False, flx=dflx)
                return dM_dNj_dot_N(Ni,p,evol_names, N0not, chain_data,
                                    microxst, djflx=(dflx / Vcm2))

            t_beg = time.time()
            # calculate the Delta M term due to flux change
            # # serial calculation
            # dM_dN_dot_Ni = np.zeros((nb_N, nb_N),)
            # for j in range(nb_evolving_nuclides):
            #     dM_dN_dot_Ni[:,j] = dM_dNj_dot_Ni(dflxi_dN[j,:])
            # cc = np.array(dM_dN_dot_Ni, copy=True)  # for testing
            # lg.info("Elapsed time for the matrix (s): %13.6g" %
            #     (time.time() - t_beg)); t_beg = time.time()

            # alternative parallel computation
            # remind that the function called by multiprocessing pool must be
            # within the scope of the pool
            pool = multiprocessing.Pool(NB_CORES)
            dM_dN_dot_Ni = np.column_stack(pool.starmap(dM_dNj_dot_N ,[(Ni,p,evol_names, N0not, chain_data,microxst,(dflxi_dN[j,:]/ Vcm2)) for j in range(nb_N)]))
                #pool.map(dM_dNj_dot_Ni, [dflxi_dN[j,:] for j in range(nb_N)])) was substitued with the latter starmap

            pool.close()
            pool.join()
            lg.info("Elapsed time for the matrix (s): %13.6g" %
                (time.time() - t_beg))
            t_beg = time.time()
            # print('Matrices are equal? ',
            #       np.sum(np.isclose(cc, dM_dN_dot_Ni)) == nb_N*nb_N)

            # redefine dNa_dt for having a new flxi
            def dNa_dt_wrapped(t, N):
                return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst,
                             nucl_chain=chain_data, p=p, V=Vcm2,
                             evol_names=evol_names, adjoint=True, flx=flxi)

            # def compute_lth_adj_funcs(nuc, Na_hat_lip1):
            for l, nuc in enumerate(evol_names):
                tt_beg = time.time()
                npos = evol_names == nuc
                if use_el_as_final_condition or (i == I - 1):
                    wf = np.zeros(nb_evolving_nuclides)
                    wf[npos] = 1.
                else:
                    wf = Na_hat[l,:,i+1]  # Na_hat_lip1

                # solve the adjoint problem (which is backwards in time)
                # explicit Runge-Kutta of order 4(5), RK45 is default method
                Nasol = ni.solve_ivp(dNa_dt_wrapped, tbnd, wf, t_eval=tbnd)
                stat, err_msg = Nasol.status, Nasol.message
                if stat != 0:
                    raise RuntimeError(str(stat) + ": " + err_msg)

                # flip the solution in time to take into account the negative
                # derivative of the adjoint problem
                Nasol.t = np.flip(Nasol.t)
                Na[l,:,i] = Nasol.y[:,-1]
                # Nali = Nasol.y[:,-1]

                # compute N adjoint hat
                Na_hat[l,:,i] = Na[l,:,i] + np.dot(Na[l,:,i], dM_dN_dot_Ni)
                # Na_hatli = Nali + np.dot(Nali, dM_dN_dot_Ni)
                # np.fill_diagonal(Na_dot_dM_dNj_dot_Ni,
                #                  1 + Na_dot_dM_dNj_dot_Ni.diagonal())

                # compute the contribution function for the nuclide at npos
                CF[l,:,i] = Na_hat[l,:,i] * Ni / Nsol.y[npos,-1]
                # CFli = Na_hatli * Ni / Nsol.y[npos,-1]

                lg.info("Elapsed time to the l-th hat adjoint (s): %13.6g" %
                    (time.time() - tt_beg))
                # return Nali, Na_hatli, CFli

            # for l, nuc in enumerate(evol_names):
            #     Na[l,:,i], Na_hat[l,:,i], CF[l,:,i] = \
            #         compute_lth_adj_funcs(nuc, Na_hat[l,:,i+1])

            lg.info("Elapsed time in the loop for adjoint funcs (s): %13.6g" %
                (time.time() - t_beg))
            input(i)
    print(CF)



