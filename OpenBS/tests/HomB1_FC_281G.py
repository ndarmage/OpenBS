#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module tests the homogeneous B1 calculation with 281G cross section
data from a UO2 fuel pin cell problem.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

baseDir = os.path.join(os.getcwd(), "..")
sys.path.append(baseDir)
from HomogB1FlxCalc import *

sys.path.append(os.path.join(baseDir, "include"))
from analyse_NData import *

MPOFile = "UO2_325_AFA3G17_tdt_8G.hdf"
# MPOFile = "UO2_325_AFA3G17_idt_2G_noB2.hdf"
MPOFile = os.path.join(baseDir, "lib", MPOFile)


def get_xs_tuple(mxs, p, G=281, L=2, z=0, N=0):
    "G is nb. of groups, while L is the order of anisotropy (plus 1)."
    st = np.zeros(G)
    sa = np.zeros_like(st)
    nxe = np.zeros_like(st)
    chi = np.zeros((G, N),)
    nsf = np.zeros((N, G),)
    ss = np.zeros((L, G, G),)
    
    for g in range(G):
        st[g] = macroxszg(mxs, p, 'Total', z, g)
        sa[g] = macroxszg(mxs, p, 'Absorption', z, g)
        nxe[g] = macroxszg(mxs, p, 'Nexcess', z, g)
        chi[g, :] = macroxszg(mxs, p, 'FissionSpectrum', z, g)
        nsf[:, g] = macroxszg(mxs, p, 'NuFission', z, g)
        for gg in range(G):
            for l in range(L-1):
                ss[l, g, gg] = macroxszg(mxs, p, 'Scattering',
                                         z, g, gg, l, True)

    np.testing.assert_allclose(np.sum(chi, axis=0), np.ones(N),
        err_msg="chi's not with unitary norm.", rtol=1.e-6)
    if N==1:
        flx = np.dot(np.linalg.inv(np.diag(st)-ss[0,:,:]), chi)
        if np.any(flx < 0):
            print('any negative flux?\n' +str(flx))
    print('check', st - np.sum(ss[0,:,:], axis=0) + nxe - sa)
    np.testing.assert_allclose(st, sa + np.sum(ss[0,:,:], axis=0) - nxe,
        err_msg="st and sa+ss-sx does not match.", rtol=1.e-6)
    # print('chi', chi)
    
    # TRR = np.array([
        # RRizg(mxs, flxref[ig], p, "Absorption", i=None, z=0, g=ig, \
          # ig=None, il=None, vrbs=False) for ig in range(G)])
    
    # mst = (mxs[0]['TotalResidual_macroAll']['Total'][p] * 
           # mxs[0]['TotalResidual_macroAll']['conc'][p])
    
    # np.testing.assert_allclose(mst, st,
        # err_msg="St does not match.")
    # print(np.sum(TRR), np.dot(flxref, sa))
    # print(ss[0,:10,:10])
    # print(
        # st - np.sum(ss[0,:,:], axis=1) - sa + nxe
    # )
    # print(
        # np.all(np.isclose(st - np.sum(ss[0,:,:], axis=0), sa))
        # )
    if G < 20:
        print(ss[0,:,:])
        print(ss[1,:,:])
    print('Shape of ss:', ss.shape)

    return st, ss, chi, nsf


EnMesh_template = """EnergyMesh(   Name: grp%03d_ENE
            NGroup: %d
              Mesh:
 %s
      Microlibrary: xslib_MIC
)
"""

def equidistant_lethargie_energy_mesh(N, E0=1.1e-10, Emax=19.6403,
                                      append=True,
                                      ofile='EnergyMeshes.udf'):
    """Print to external file an equidistant lethargy mesh with bounds
    in MeV energy unit between E0 and Emax."""
    u = lambda E: np.log(Emax / E)
    umax, umin = u(E0), 0
    Du = (umax - umin) / float(N)
    umesh = np.linspace(umin, umax, N + 1)
    Emesh = Emax * np.exp(- umesh)
    str = ('\n '.join([' '.join(['{:13.6g}']*5)]*((N + 1)//5)) + '\n ' +
          ' '.join(['{:13.6g}']*((N + 1)%5)))
    with open(ofile, 'a' if append else 'w') as f:
        f.write(EnMesh_template % (N, N, str.format(*Emesh)))
        f.write('END')


def get_iso_set(mlib):
    "Return the set of available nuclides in the microlib mlib."
    return set().union(*map(set, [zxs.keys() for zxs in mlib]))


def plot_eigenspectrum(B2_list):
    fig, ax = plt.subplots()
    for i, B2 in enumerate(B2_list):
        ax.scatter(B2.real, B2.imag, c='C%d' % i)
        # print(B2)
        # print(i, len(B2))
    ax.set_xlabel('$\Re({B^2})$')
    ax.set_ylabel('$\Im({B^2})$')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    plt.show()


def plot_gamma_func(B2):
    y = [alpha(b) for b in B2]
    z = [full_gamma(b) for b in B2]
    s = [gamma(b) for b in B2]  # approximation
    # print(B2)
    # print(y)
    fig, ax = plt.subplots()
    ax.plot(B2, y, 'C0:', label=r'$\alpha$ (cm$^{-1}$)')
    ax.plot(B2, z, 'C1-', label=r'$\gamma$')
    ax.plot(B2, s, 'C2--', label=r'$\bar{\gamma}$')
    ax.legend()
    ax.set_xlabel(r'$B^2\: (\Sigma=1)$')
    # ax.set_yscale('symlog')
    # ax.set_xscale('symlog')
    plt.show()


if __name__ == "__main__":

    # calculate equidistant lethargy meshes
    # for N in [4, 8, 16, 32, 64, 128, 256]:
        # equidistant_lethargie_energy_mesh(N)

    # B2 = np.logspace(-3, 0, 100)
    # B2 = np.append(np.append(-B2[::-1], [0]), B2)
    # plot_gamma_func(B2)

    # sys.exit()
    # verify the implementation
    MPOdata = readMPO(MPOFile)
    print(MPOdata.keys())
    # print(MPOdata['BURN'])
    # print(MPOdata['ETFl'])
    # print(MPOdata['TMod'])
    nb_state_params, ng = len(MPOdata['NVALUE']), MPOdata['NG']
    
    # select the state point (nominal power, fresh fuel)
    # p = (0, 0, -1, 0)
    p = tuple(np.zeros((nb_state_params,), dtype=np.int))
    print('\n'.join(["Retrieve data at:\n",
          " Bu (MWd/t) = {:8.3f}".format(MPOdata['BURN'][p[1]]),
          "     Tf (K) = {:8.3f}".format(MPOdata['ETFl'][p[2]]),
          "     Tm (K) = {:8.3f}".format(MPOdata['TMod'][p[3]]),
          # "Plin (W/cm) = {:8.3f}".format(MPOdata['Plin'][p[4]])
          ]))
    
    microlib = MPOdata['microlib']
    isos = get_iso_set(microlib)
    keff, kinf = MPOdata['keff'], MPOdata['kinf']
    flxref = MPOdata['TotalFlux'][p]
    print('List of present isotopes:\n', isos)
    niso = len(isos)
    print(' -> for a total of %d nuclides.' % niso)
    
    kinfc = calc_kinf_from_microlib_at_p(MPOdata, p)
    np.testing.assert_almost_equal(kinfc, kinf[p], decimal=6,
        err_msg="kinf not verified by calc_kinf_from_microlib_at_p.")
    
    xs = get_xs_tuple(microlib, p, G=ng, N=niso)  #  (st, ss, chi, nsf)
    kinfc, flx_inf = compute_kpairs(xs)
    
    print('calc.ed kinf', kinfc)
    print('    ap3 kinf', kinf[p])
    print('    ap3 keff', keff[p])
    
    # very coarse tolerance k=for unknown reasons of IDT...
    np.testing.assert_almost_equal(kinfc, kinf[p], decimal=3,
        err_msg="kinf not verified by compute_kpairs.")
        
    # g1, g2, g3 = 4. / 15., - 12. / 175., 92. / 2625.
    B2_list = []
    B2_g4, flx = find_B2_spectrum(xs, one_over_k=1., g=coefs)
    B2_list.append(B2_g4)
    B2_g3, flx = find_B2_spectrum(xs, one_over_k=1., g=coefs[:-1])
    B2_list.append(B2_g3)
    B2_g2, flx = find_B2_spectrum(xs, one_over_k=1., g=coefs[:-2])
    B2_list.append(B2_g2)
    B2_g1, flx = find_B2_spectrum(xs, one_over_k=1., g=coefs[:-3])
    B2_list.append(B2_g1)
    plot_eigenspectrum(B2_list)
    
    # np.testing.assert_almost_equal(kinf, 1.1913539017168697, decimal=7,
        # err_msg="kinf not verified.")
    # np.testing.assert_allclose(flx_inf, [39.10711218,  5.85183328],
        # err_msg="fundamental flx_inf not verified.")
    # adj_kinf, adj_flx_inf = compute_kpairs(xs, adjoint=True)
    # np.testing.assert_almost_equal(adj_kinf, 1.1913539017168697, decimal=7,
        # err_msg="adjoint kinf not verified.")
    # np.testing.assert_allclose(adj_flx_inf, [1.1913539, 1.50878553],
        # err_msg="fundamental adjoint flx_inf not verified.")
    # np.testing.assert_almost_equal(find_B2(xs)[0], 0.004184657328394975,
        # decimal=7, err_msg="B2 not verified.")
