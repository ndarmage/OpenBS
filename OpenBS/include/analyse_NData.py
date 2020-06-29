#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Author: Tomatis D., Xu S.
:Date: 29/01/2018
:Company: CEA/DEN/DANS/DM2S/SERMA/LPEC

Browse the MPO xslib produced by AP3 and extract multigroup
data homogenized per zone: microscopic cross sections of all
istopes are retrieved together with their concentrations; the
multigroup neutron flux; local valeus if available.

This module agrees with the specifications provided by [1].

Parameters determining a homogenized micro cross section of a
certain isotope (U235, U238, Pu239...) of a certain reaction
(e. g., absorption, fission, scattering...) for a state point,
zone and energy group :

1. zone (more than 1 zone when pbp)
2. state point (e.g. burnup--Tm--Tf--Br--...)
3. energy group
4. order of scattering anisotropy
5. departure and arrival groups if scattering

As for the diffusion, the number of cross sections are equal to
the number of energy groups times anisotropy order + 1 because
of the terms in the Legendre polynomial expansion.

For the scattering, the number of cross sections is equal to the
square of the number of energy groups times anisotropy order + 1.
Remins that only the non-zero terms are stored in the MPO and that
these values are read by 'TRANSPROFILE'.

Important concepts of scattering:
scattering profile, scattering matrix profiles
The scattering matrix profile only contains non-zero terms
in the scattering matrix and it is stored in the column
order.
The energy group indices are the same as the common notation:
i.e. from the highest energy to the lowest energy.
For instance, with two energy groups it is:
19.6403, 6.25E-7 and 1.1E-10,
group 0 is between 19.6403 and 6.25E-7, group 1 is between 6.25E-7
and 1.1E-10.

data structure of the micro-library
-----------------------------------

The micro-library contains the following data:
\[
  (N, \sigma^{(type)}_{g[,g,l]})_i^{(n)}(\vec{p})
\]
for the isotope $i$ specified in the reduced depletion chain at
homogenization, and where $N$ and $\sigma^{(type)}$ are respectively
the concentration of the nuclide and the microscopic cross section
of a given type. The available types of reaction rates come from
the MPO as well. $n$ is the output zone of homogenization and
$\vec{p}$ is the vector of state parameters.

Data is retrieved from the MPO and arranged hereafter as:

data['microlib']=[{} for i in range(nzone)]
microlib=data['microlib']
microlib[n] = { '*isotope name*': {
                'conc': numpy.ndarray(tuple(p)),
                '*$\sigma^{(type)}$*':
                        numpy.ndarray(tuple(p+[g,($\cdot$)]))
                                  }
}
where quantities within asterisks come from the MPO.


Notes
-----
# Retrieve non 0C available isotopes in all zones and state points
## Illustrated by an example: retrieve non 0C available isotopes
    in state point 2 and zone 10
### 1. store total available isotopes in all zones
    Remark: this step is done in state point 0 since addrzi is
    independent on state points
#### 1.1 read "../zone_10/ADDRZI" and store the number as addrzi
#### 1.2 access "..output_0/info/ADDRISO"
    and use addrzi to determine two bords of isotope id list
    ADDRISO[addrzi] and ADDRISO[addrzi + 1]
#### 1.3 access "..output_0/info/ISOTOPE"
    and use ADDRISO[addrzi] and ADDRISO[addrzi + 1] to
    determine the isotope id list ISOTOPE[ADDRISO[addrzi] : ADDRISO[addrzi + 1]]
#### 1.4 access "contents/isotopes/ISOTOPENAME"
    and use ISOTOPE[ADDRISO[addrzi] : ADDRISO[addrzi + 1]]
    to determine the isotope name list ISOTOPENAME_arr
    [ISOTOPE[ADDRISO[addrzi] : ADDRISO[addrzi + 1]]]
    this list is total available isotopes
### 2. filter 0C isotopes
#### 2.1 access output/output_0/statept_2/zone_10/CONCENTRATION
    and store all the indexes of isotopes with non zero concentration
    (use np.nonzero) then store the non 0C isotopes types and their
    concentrations

.. note:: this module runs with python3.7.

References
----------

1/ F. Auffret et Al., Description of the APOLLO3 Multi-parameter Output
Library for the version AP3-2.0, DEN/DANS/DM2S/SERMA/LTSD/RT/17-6237/A

"""

import os, sys
import numpy as np
import h5py
import logging as lg
import time

logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
lg.basicConfig(level=lg.INFO)  # filename = logfile

if (sys.version_info > (3, 0)):
    # Python 3 code in this block
    import pickle
else:
    # Python 2
    import cPickle as pickle

from scipy.constants import electron_volt as eV2J
MeV2J = eV2J * 1.e+6

__version__ = "v3.0.0"

h5float = np.float32

# list of keys with data stored as ndarrays
# . functions of p-arrays, where p is the parameter vector
parrays = ["kinf", "keff"]
# . functions of the zone region and of the p-arrays
# + all LOCALVALUES are nparrays
# .functions of the zone region, of the p-arrays and of the groups
npgarrays = ["ZoneFlux", "ZonePower", "DB2"]
pgarrays = ["TotalFlux"]

one_third = 1. / 3.

#-----------------------------------------------------------------
# functions

def initdata(nb_pnts=0, nvalue=[], nzone=0, ng=1, totiso=None,
             addrzx=[], avarea=None, anis_pl1=1, locvaltp=None,
             ntp=h5float):

    """Initialize the data dictionary.
    :param nb_pnts, number of state points
    :param nvalue, state parameters of state points
    :param nzone, number of zones
    :param ng, number of energy groups
    :param totiso, tot isotope names
    :param addrzx, address covering the cross section
    :param avarea, available reaction id and name
    :param anis_pl1, scattering anistropy order + 1
    :param locvaltp, local value types
    :param ntp, numpy data type dtype
    :type nb_pnts, int
    :type nvalue, list
    :type nzone, int
    :type ng, int
    :type totiso, 2d list
    :type addrzx, list
    :type avarea, 2d list whose elements are dictionaries
    :type anis_pl1, int
    :type locvaltp, list
    :type ntp, data type object
    """

    nb_st_params = len(nvalue)
    if nb_st_params < 1: raise ValueError("missing input nvalue")
    if totiso == None: raise ValueError("missing input totiso")
    if not len(addrzx) == nzone: raise ValueError("addrzx and nzone dismatch")
    if avarea == None: raise ValueError("missing input avarea")

    # initialize the dictionary
    data = {}

    # initialization
    # IMPORTANT: dictionary determined per zone
    # remark from Daniele: the initialization can be improved by
    # dictionary + list comprehension
    data["microlib"] = [{} for iz in range(nzone)]
    microlib = data["microlib"]

    nvalueg = np.append(nvalue, [ng])
    nvaluelg = np.append(nvalue, [anis_pl1, ng])
    nvaluelgg = np.append(nvalue, [anis_pl1, ng, ng])
    znvalue = np.append([nzone], nvalue)
    znvalueg = np.append(znvalue, [ng])

    # micro-library
    for iz in range(nzone):
        # addrzx: address recovering xs, dependent only on zones
        addrzx_i = addrzx[iz]

        # browse all the isotopes
        for isoid, isonm in enumerate(totiso[iz]):
            # initialization of dictionary
            microlib[iz][isonm] = {}

            # initialization of concentration
            microlib[iz][isonm]["conc"] = np.full(nvalue, np.nan, dtype=ntp)

            # browse available reaction names
            for reaid, reanm in avarea[addrzx_i][isoid].items():
                # initialization of xs
                reanml = reanm.lower()
                if "scat" in reanml:
                    microlib[iz][isonm][reanm] = np.zeros(nvaluelgg, dtype=ntp)
                elif "diff" in reanml:
                    microlib[iz][isonm][reanm] = np.full(nvaluelg, np.nan, dtype=ntp)
                else:
                    microlib[iz][isonm][reanm] = np.full(nvalueg, np.nan, dtype=ntp)

            # add the total xs and diffusion constant
            # WARNING: the new key name must not contain diff nor scat!
            microlib[iz][isonm]["Total"] = np.full(nvalueg, np.nan, dtype=ntp)
            microlib[iz][isonm]["DffConstant"] = np.full(nvalueg, np.nan, dtype=ntp)

    # local value types
    if locvaltp is not None:
        data["LOCALVALUE"] = locvaltp
        for lvt in locvaltp:
            # for most of cases, there is only one single value
            # maybe exception, e. g., ADF Discontinuity Factors.
            data[lvt] = np.full(znvalue, np.nan, dtype=ntp)

    # others
    data["TotalFlux"] = np.full(nvalueg, np.nan, dtype=ntp)
    data["ZoneFlux"] = np.full(znvalueg, np.nan, dtype=ntp)
    data["DB2"] = np.full(znvalueg, np.nan, dtype=ntp)
    # added because LOCAL_VALUE/POWER is wrong
    data["ZonePower"] = np.full(znvalueg, np.nan, dtype=ntp)
    data["keff"] = np.full(nvalue, np.nan, dtype=ntp)
    data["kinf"] = np.full(nvalue, np.nan, dtype=ntp)
    data["BUCKLING"] = np.full(nvalue, np.nan, dtype=ntp)

    for i in range(nb_st_params):
        data["PARAM_%d"%(i)] = np.zeros((nb_pnts,), dtype=np.int8)

    return data
#-----------------------------------------------------------------


def mergeRPData(d1=None, d2=None, vrbs=True):
    """Merge the input dict d1 and d2 having MPO data from different
    Reprise Calculations. Return the sum as d1, that becomes the union of
    the input d1 and d2. Two possible cases are allowed. The first occurs
    for d1 intersection d2 equal to the empty set, and the second requests
    that d2 is (strictly) included in d1. Other cases raise an error.
    Currently, the keys PARAM_x are not updated and the correspondence
    between the state parameter vectors and the index of the calculation
    points is lost.
    """
    # the sum with None or with an empty set yields itself
    if d1 is None or len(d1) <= 0: return d2
    if d2 is None or len(d2) <= 0: return d1
    if d1 is None and d2 is None:
        raise InputError("Invalid input.")
    if len(d1) != len(d2):
        raise ValueError("input dict do not have the same number of keys.")
    
    nzone, paramname, nvalue = d1["NZONE"], d1["PARAMNAME"], d1["NVALUE"]
    nparams, ng = len(nvalue), d1["NG"]
    if nparams != len(d2["NVALUE"]):
        raise ValueError("Input dictionaries do not have the same " + \
                         "number of parameters")
    if np.setdiff1d(paramname, d2["PARAMNAME"]).size != 0:
        raise ValueError("Input dictionaries do not have the same " + \
                         "kind of parameters")
    if nzone != d2["NZONE"]:
        raise ValueError("The input dictionaries do not have the same " + \
                         "number of zones")
    if ng != d2["NG"]:
        raise ValueError("The input dictionaries do not have the same " + \
                         "number of energy groups")
    d1m, d2m = d1["microlib"], d2["microlib"]

    # find common parameters and axes to merge
    disjoint_axes = np.empty(nparams, dtype=np.bool)
    equal_axes = np.empty_like(disjoint_axes)
    np.set_printoptions(precision=1)
    for ik, k in enumerate(paramname):
        # print(ik, k)
        # print('d1 = ', d1[k])
        # print('d2 = ', d2[k])
        disjoint_axes[ik] = np.intersect1d(d1[k], d2[k]).size == 0
        equal_axes[ik] = np.setdiff1d(d1[k], d2[k]).size == 0
    
    merge = sum(disjoint_axes) > 0  # becomes 'set' d1 if False
    if sum(disjoint_axes) > 1:
        raise RuntimeError("Merge on multiple axes not verified yet.")
    
    nb_different_axes = nparams - sum(equal_axes)
    if vrbs:
        print("Nb of different axes is {:d}".format(nb_different_axes))
        print("Merge mode is " + str(merge))
        for ik, k in enumerate(paramname):
            print(".axis {:>20s} ".format(k) +
                  ("(-same--) " if equal_axes[ik] else "(differs) ") +
                  ("(disjoint)" if disjoint_axes[ik] else ''))
    if nb_different_axes == 0:
        lg.info("Requested merge of equal MPO spaces, return the first entry")
        return d1
    
    if merge:  # merge mode
        MergeAxis = zip(np.where(disjoint_axes)[0], paramname[disjoint_axes])
        for ik, k in MergeAxis:
            print(" + axis %d, param %s" % (ik, k))

            #  ** TO BE REMOVED IN LATER VERSION ***
            # WARNING, Daniele: remove the PARAM DMod which is not acting
            # here as an indepependent parameter
            #  ** TO BE REMOVED IN LATER VERSION ***
            if k.lower() == "dmod":
                MergeAxis.remove((ik, k))
                continue

            # obsolete
            if len(np.intersect1d(d1[k], d2[k])) > 0:
                raise RuntimeError("Cannot merge MPO with common calc points!")

            # check axes
            for ik2, k2 in enumerate(d1["PARAMNAME"]):
                if ik2 == ik: continue
                #  ** TO BE REMOVED IN LATER VERSION ***
                # WARNING, Daniele
                if k2.lower() == "dmod": continue
                #  ** TO BE REMOVED IN LATER VERSION ***
                if d2[k2] not in d1[k2]:
                    # only the axis k is meant to be merged, the valeus of the
                    # other axes must already be present on the axes if d1
                    print("PARAM " + k2 + " mismatch in input MPO dicts.")
                    raise ValueError("Invalid partition!")

            # form the new axis
            d1p = np.sort(np.union1d(d1[k], d2[k]))
            d2pind1p = np.argwhere( d1p == d2[k] )[0]

            for ij, j in enumerate(d2pind1p):
                #jj = ':,'*ik+str(ij)+',:'*(nb_params_d2-ik-1)

                # treat p-array functions
                for e in parrays + pgarrays:
                    dtoadd = np.take(d2[e], ij, axis=ik)
                    d1[e] = np.insert(d1[e], j, dtoadd, axis=ik)
                # treat local values (np-array), where the first index is
                # for the homogenization zone
                for e in list(d1["LOCALVALUE"]) + npgarrays:
                    dtoadd = np.take(d2[e], ij, axis=ik+1)
                    d1[e] = np.insert(d1[e], j, dtoadd, axis=ik+1)
                # treat the microlib
                for iz in range(nzone):
                    for iso in d1m[iz]:
                        for rea in d1m[iz][iso]:  # "conc" included !

                            dtoadd = np.take(d2m[iz][iso][rea], ij, axis=ik)
                            d1m[iz][iso][rea] = \
                                np.insert(d1m[iz][iso][rea], j, dtoadd, axis=ik)

            # reset the PARAM values
            # WARNING: the keys PARAM_x lose their meaning now
            d1[k] = d1p
            d1["NVALUE"][ik] = len(d1[k])

        MergedAxes = paramname[disjoint_axes]
        if vrbs:
            print("{:d} axis(or axes) merged, being:\n".format(
                  len(MergedAxes)) + str(MergedAxes))
        lg.info(" *** merge mode done ***")

    else:  # set mode
    
        # make the slicer to fit d2 into d1, with d2 \in d1.
        slicer = []
        for ik, k in enumerate(paramname):
            if equal_axes[ik]:
                s = slice(nvalue[ik])  # i.e. ':'
            else:
                # form the new axis and reset the PARAM values
                # WARNING: the keys PARAM_x lose their meaning now
                # check for inclusion!
                if np.setdiff1d(d1[k], np.union1d(d1[k], d2[k])).size > 0:
                    raise RuntimeError("d1[k] U d2[k] \ d1[k] != set() for" +
                                       " k = " + k)
                s = np.argwhere( d1[k] == d2[k] )[0]
                if len(s) == 1:
                    s = slice(s[0], s[0]+1, 1)
                else:
                    lg.warn("s = " + str(s) + ", case is not supported yet")
            slicer.append(s)
        if vrbs:
            print("p slicer is:\n" + str(slicer))
    
        # fit d2 into the new d1 given by the union of d1 and d2
        # treat p-array functions
        pslicer = tuple(slicer)
        for e in parrays:
            d1[e][pslicer] = d2[e]
    
        pgslicer = tuple(slicer + [slice(ng)])
        for e in pgarrays:
            d1[e][pgslicer] = d2[e]

        # treat local values (np-array), where the first index is
        # for the homogenization zone
        npslicer = tuple([slice(nzone)] + slicer)
        for e in d1["LOCALVALUE"]:
            d1[e][npslicer] = d2[e]

        # treat local values (np-array), where the first index is
        # for the homogenization zone
        npgslicer = tuple([slice(nzone)] + slicer + [slice(ng)])
        for e in npgarrays:
            d1[e][npgslicer] = d2[e]

        # treat the microlib
        for iz in range(nzone):
            for iso in d1m[iz]:
                for rea in d1m[iz][iso]:  # "conc" included!

                    # print(rea, d1m[iz][iso][rea].size,
                    #       d1m[iz][iso][rea].shape)
                    d1m[iz][iso][rea][pslicer] = d2m[iz][iso][rea]
    # input('check')
    d1[ "NSTATEPOINT" ] += d2[ "NSTATEPOINT" ]
    return d1
#-----------------------------------------------------------------


def macroxszg(microlib, p, xstype="Total", z=0, g=0, ig=None, il=None, \
    vrbs=False, keep_separate=False):
    """Compute the macroscopic cross section of a given type, in the zone z and
    the energy group g, at the input state parameter vector p. The second group
    index and the anisotropy order are needed in case of scattering cross
    section. If keep_separate is True, the output cross sections are still
    macroscopic but they are separate per available nuclide."""
    if not isinstance(p, tuple):
        raise ValueError("Input state parameter vector is not tuple")

    nzone, zmlib = len(microlib), microlib[z]
    if nzone <= 0: raise ValueError("Missing zones in microlib")
    if not (0 <= z < nzone): raise ValueError("input zone out of bounds")
    # ziso = [list( microlib[iz].keys() ) for iz in range(nzone)]
    isoz = list(zmlib.keys())  # list of isotopes in zone z
    niso = len(isoz)  # nb of isotopes in zone z
    if niso <= 0: raise ValueError("No isotopes in the given region.")
    # total number of available isotopes in all zones
    # tiso = set().union(*map(set, [zm.keys() for zm in microlib]))

    reatypes = list(
        set.union( *map(set, [zmlib[iso].keys() for iso in isoz]) )
    )
    if xstype not in reatypes:
        if vrbs: print(xstype + " is not available in zone {:d}".format(z))
        return 0.

    if "dffconstant" in xstype.lower():
        print("WARNING: you are requesting macro xs on diffusion constants")

    # get type of all float data in microlib
    concs = np.array([zmlib[iso]["conc"][p] for iso in isoz])
    mxs = np.zeros_like(concs)

    if ig is None:
        if il is None:
            t_p = tuple(np.append(p, [g]))
        else:
            t_p = tuple(np.append(np.append(p, [il]), [g]))
        for ci, cv in enumerate(concs):
            iso = isoz[ci]
            # if xstype in zmlib[iso]:
            #     mxs[ci] = zmlib[iso][xstype][t_p]
            # else:
            #     if vrbs: print(iso+" does not have the xs "+xstype)

            try:
                mxs[ci] = zmlib[iso][xstype][t_p]
            except KeyError:
                # it has been noticed that there're isotopes with very
                # small concentrations without the correspondent xstype
                if vrbs: print(iso + " does not have the xs " + xstype)
                pass
    else:
        if il is None:
            t_pgg = tuple(np.append(p, [g, ig]))
            for ci, cv in enumerate(concs):
                iso = isoz[ci]
                try:
                    mxs[ci] = zmlib[iso][xstype][t_pgg]
                except KeyError:
                    # it has been noticed that there're isotopes with very
                    # small concentrations without the corrispondent xstype
                    if vrbs: print(iso + " does not have the xs "+xstype)
                    pass
        else:
            t_pggl = tuple(np.append(p, [il, g, ig]))
            for ci, cv in enumerate(concs):
                iso = isoz[ci]
                try:
                    mxs[ci] = zmlib[iso][xstype][t_pggl]
                except KeyError:
                    # it has been noticed that there're isotopes with very
                    # small concentrations without the corrispondent xstype
                    if vrbs: print(iso + " does not have the xs "+xstype)
                    pass

    # print(concs)
    # print(mxs)
    # print(concs * mxs); input('ok')
    if keep_separate:
        if 'fissionspectrum' in xstype.lower():
            macroxs = np.nan_to_num(mxs)
        else:
            macroxs = np.nan_to_num(mxs * concs)
    else:
        if 'fissionspectrum' in xstype.lower():
            macroxs = np.mean(mxs[mxs > 0])
        else:
            macroxs = np.nansum(mxs * concs)
    return macroxs
#-----------------------------------------------------------------


def macroxsz(microlib, p, xstype="Total", z=0):
    """Compute the macroscopic cross section of a given type, in the zone z and
    summed over all energy groups, at the input state parameter vector p."""
    if not isinstance(p, tuple):
        raise ValueError("Input state parameter vector is not tuple")

    nzone= len( microlib )
    if nzone <= 0: raise ValueError("Missing zones in microlib")
    if not (0 <= z < nzone): raise ValueError("input zone out of bounds")

    iso0 = microlib[z].keys()[0]
    ng = len( microlib[z][iso0]["Total"][p] )

    return np.sum([macroxszg(microlib, p, xstype, z, g) for g in range(ng)])
#-----------------------------------------------------------------


def macroxsg(microlib, p, xstype="Total", g=0):
    """Compute the macroscopic cross section of a given type, in the energy
    group g and summed over all regions, at the input state parameter vector
    p."""
    if not isinstance(p, tuple):
        raise ValueError("Input state parameter vector is not tuple")

    nzone = len(microlib)
    if nzone <= 0: raise ValueError("Missing zones in microlib")

    isos = microlib[0].keys()
    ng = len(microlib[0][isos[0]]["Total"][p])
    if not (0 <= g < ng): raise ValueError("Invalid input en. group number")

    return np.sum([macroxszg(microlib, p, xstype, z, g) for z in range(nzone)])
#-----------------------------------------------------------------


def macroxs(microlib, p, xstype="Total"):
    """Compute the macroscopic cross section of a given type, at the
    input state parameter vector p."""
    if not isinstance(p, tuple):
        raise ValueError("Input state parameter vector is not tuple")

    nzone = len(microlib)
    if nzone <= 0: raise ValueError("Missing zones in microlib")

    isos = microlib[0].keys()
    ng = len(microlib[0][isos[0]]["Total"][p])

    return np.sum([macroxsg(microlib, p, xstype, g) for g in range(ng)])
#-----------------------------------------------------------------


def RRizg(microlib, flx, p, xstype="Total", i=None, z=0, g=0, \
          ig=None, il=None, vrbs=False):
    """Compute the reaction rate for the input isotope i, in the zone
    z, in the group g and at the state parameter vector p. In case of
    scattering reaction rates, the second group index ig and the order
    of anisotropy il are also needed. If i is None, the
    total reaction rates (summed over all isotopes) will be computed.
    The input flux is expected as an np.ndarray of shape(nzone, ng)."""
    macroRR = True if i is None else False
    if not isinstance(p, tuple):
        raise ValueError("Input state parameter vector is not tuple")

    nzone = len(microlib)
    if nzone <= 0: raise ValueError("Missing zones in microlib")

    # get the set of all existing isotopes
    # remind that they may not be all present in some zone
    ziso = [list(microlib[iz].keys()) for iz in range(nzone)]
    niso = [len(ziso[iz]) for iz in range(nzone)]
    # total number of available isotopes in all zones
    tiso = set().union(*map(set, ziso))
    if not (i in tiso or macroRR):
        if vrbs: print(i + " missing in zone %d" % z)
        return 0.
    # all show the total xs, so we use it to get ng in the first zone
    ng = len(microlib[0][ziso[0][0]]["Total"][p])

    # nzone_flx, ng_flx = flx.shape
    # if ng != ng_flx: raise ValueError("Number of energy group mismatch")
    # if nzone != nzone_flx: raise ValueError("Number of zones mismatch")

    if not (0 <= g < ng): raise ValueError("Invalid input en. group number")
    if not (0 <= z < nzone): raise ValueError("input zone out of bounds")

    reatypes = set().union(
        *map(set, [microlib[z][iso].keys() for iso in ziso[z]])
    )
    if not xstype in reatypes:
        if vrbs:
            lg.warn(xstype + " is not available in zone {:d}".format(z))
        return 0.
    if "dffconstant" in xstype.lower():
        lg.warn("you are requesting macro xs on diffusion constants")

    # WARNING: there are cases where macroscopic cross sections come with a
    # non-unitary concentration
    if not macroRR:
        if not i in microlib[z].keys():
            ValueError("Missing input isotope in microlib.")
        conc = np.nan_to_num(microlib[z][i]["conc"][p])

    # this requires that all zones have the same isotopes (also with
    # nan conc) !
    if "scat" in xstype.lower():
        if ig == None: raise ValueError("Missing 2nd group index")
        if il == None: raise ValueError("Missing anysotropy order")
        if not (0 <=ig < ng): raise ValueError("Invalid input 2nd group index")
        if macroRR:
            mxs = macroxszg(microlib, p, xstype, z, g, ig, il)
        else:
            peg = np.append(p, [il,g,ig])
            mxs = microlib[z][i][xstype][tuple(peg)]
            mxs*= conc
    else:
        if macroRR:
            mxs = macroxszg(microlib, p, xstype, z, g)
        else:
            pg = np.append(p, [g])
            mxs = microlib[z][i][xstype][tuple(pg)]
            mxs*= conc

    return ( mxs * flx )
#-----------------------------------------------------------------


def calc_kinf_from_microlib_at_p(data, p):
    """Calculate the kinf with the xs at p from the microlib."""
    # memento, if ng = 1 and nzone = 1
    # kinfc = ["NuFission"][p] / (["Absorption"][p] - ["Nexcess"][p])
    xslib, zflx = data["microlib"], data["ZoneFlux"]
    ng, nzone = data["NG"], data["NZONE"]
    P, A, pp = 0., 0., np.insert(np.append(p, 0), 0, 0)
    # try:
        # with_leakage_model = not np.isnan(data["BUCKLING"][p])
    # except KeyError:
        # # BUCKLING and others are removed from data if missing
        # with_leakage_model = False
    for iz in range(nzone):
        #zp = np.append([ iz ], p)
        pp[0] = iz
        for ig in range(ng):
            pp[-1] = ig
            #zpg = tuple(np.append(zp, [ ig ]))
            zpg = tuple(pp)
            flxzpg = zflx[zpg]
            P += RRizg(xslib, flxzpg, p, xstype="NuFission" , z=iz, g=ig)
            A += RRizg(xslib, flxzpg, p, xstype="Absorption", z=iz, g=ig)
            A -= RRizg(xslib, flxzpg, p, xstype="Nexcess"   , z=iz, g=ig)

    # info: in case of leakage model, the kinf is still calculated as
    # ratio between production and removal but with the spectrum
    # obtained by the leakage model. This means that if the following
    # is uncommented, the calculated value should be 'very close' to 1
    # up to the machine precision.
#        if with_leakage_model:
#            zp = tuple(pp[:-1])
#            A += np.dot(data["DB2"][zp], zflx[zp])

    return P / A  # kinfc
    
#  (old version, potentially obsolete)
#            # memento, if ng = 1 and nzone = 1
#            # kinfc = ["NuFission"][p] / (["Absorption"][p] - ["Nexcess"][p])
#            P, A = 0., 0.
#            for iz in range(nzone):
#                #zp = np.append([ iz ], p)
#                pp[0] = iz
#                for ig in range(ng):
#                    pp[-1] = ig
#                    #zpg = tuple(np.append(zp, [ ig ]))
#                    zpg = tuple(pp)
#                    flxzpg = data[ "ZoneFlux" ][zpg]
#                    P += RRizg(microlib, flxzpg, p, xstype="NuFission" , z=iz, g=ig)
#                    A += RRizg(microlib, flxzpg, p, xstype="Absorption", z=iz, g=ig)
#                    A -= RRizg(microlib, flxzpg, p, xstype="Nexcess"   , z=iz, g=ig)
#
#            if with_leakage_model:
#                # it means a leakage model was used in the lattice calc.
#                A+= np.dot( data["DB2"][p], data["TotalFlux"][p])
#
#            kinfc = P / A
#-----------------------------------------------------------------


def fill_microlibz(microlibz, avaiso, avarea_addrzx_i, p, anis_pl1, ng, \
                   totisoconc, transp_avails_addrzx_i, pos_avails_addrzx_i, \
                   transprofile, xs):
    """Fill microlib[iz] with MPO xs-data."""

    # browse available isotopes
    for isoid, isonm in avaiso:
        # concentrations
        microlibz[ isonm ][ "conc" ][p] = totisoconc[isoid]

        for reaid, reanm in avarea_addrzx_i[isoid].items():
            reanm_l = reanm.lower()
            if "scat" in reanm_l:
                # determine the first value in the TRANSPROFILE
                TP_0 = transp_avails_addrzx_i[isoid]
                # determine fag and adr
                fag = transprofile[TP_0 : TP_0 + ng]
                adr = transprofile[TP_0 + ng : TP_0 + 2 * ng + 1]

                for ianis in range(anis_pl1):
                    pa = np.append(p, [ ianis ])
                    for idg in range(ng):
                        for iag in range(ng):

                            # verify if scale in the adr
                            scale = adr[idg] + iag - fag[idg]

                            if adr[idg] <= scale < adr[idg + 1]:
                                # state parameter for scattering
                                #pada = tuple(np.append(pa, [ idg, iag ]))
                                # transpose the matrix to get a lower tringaular-like form
                                pada = tuple(np.append(pa, [ iag, idg ]))

                                microlibz[ isonm ][ reanm ][pada] = \
                                    xs[pos_avails_addrzx_i[isoid][reaid] + \
                                       adr[ng] * ianis + scale]
                            #else:
                            # init to 0 the scattering matrix (already done by initdata!)
                            #    microlibz[ isonm ][ reanm ][pada] = 0.

            elif "diff" in reanm_l:
                for ianis in range(anis_pl1):
                    pa = np.append(p, [ ianis ])
                    for ig in range(ng):
                        pae = tuple(np.append(pa, [ ig ]))
                        microlibz[ isonm ][ reanm ][pae] = \
                            xs[pos_avails_addrzx_i[isoid][reaid] + ianis * ng + ig]
            else:
                for ig in range(ng):
                    pe = tuple(np.append(p, [ ig ]))
                    microlibz[ isonm ][ reanm ][pe] = \
                        xs[pos_avails_addrzx_i[isoid][reaid] + ig]

        # determine the total xs
        for ig in range(ng):
            p0e = tuple(np.append(p, [ 0, ig ]))
            scat0 = np.sum(microlibz[ isonm ][ "Scattering" ][p0e])

            pe = tuple(np.append(p, [ ig ]))
            microlibz[ isonm ][ "Total" ][pe] = scat0 + \
                microlibz[ isonm ][ "Absorption" ][pe]
            # the Total xs must contain the nxn contribution
            # if "Nexcess" in microlibz[ isonm ]:
            #     microlibz[ isonm ][ "Total" ][pe] -= \
            #         microlibz[ isonm ][ "Nexcess" ][pe]

            # diffusion constant = 1/3/(total - scattering_1)
            tmp = np.array(microlibz[ isonm ][ "Total" ][pe], copy=True)
            if anis_pl1 >= 1:
                p1e = tuple(np.append(p, [ 1, ig ]))
                scat1 = np.sum(microlibz[ isonm ][ "Scattering" ][p1e])
                scat1 *= one_third
                tmp -= scat1
            microlibz[ isonm ][ "DffConstant" ][pe] = one_third / tmp
    pass
#-----------------------------------------------------------------


def fill_microlibz_v2(microlibz, avaiso, avarea_addrzx_i, p, anis_pl1, ng, \
                   totisoconc, transp_avails_addrzx_i, pos_avails_addrzx_i, \
                   transprofile, xs):
    """Fill microlib[iz] with MPO xs-data. Attempt to boost the code
    performance by having only once the allocations of the ndarrays.
    Warning: many conversions to tuple-s."""

    # allocate memory for the tuples only once, in order to avoid many
    # copies within the loops
    if not isinstance(p, tuple):
        raise ValueError("Input state parameter vector is not tuple")

    nb_params, pp = len(p), np.append(p, np.zeros((3), dtype=np.int8))

    # browse available isotopes
    for isoid, isonm in avaiso:

        # concentrations
        microlibz[isonm]["conc"][p] = totisoconc[isoid]

        for reaid, reanm in avarea_addrzx_i[isoid].items():
            reanm_l = reanm.lower()
            if "scat" in reanm_l:
                # determine the first value in the TRANSPROFILE
                TP_0 = transp_avails_addrzx_i[isoid]
                # determine fag and adr
                fag = transprofile[TP_0 : TP_0 + ng]
                adr = transprofile[TP_0 + ng : TP_0 + 2 * ng + 1]

                for ianis in range(anis_pl1):
                    pp[nb_params] = ianis
                    for idg in range(ng):
                        for iag in range(ng):

                            # verify if scale in the adr
                            scale = adr[idg] + iag - fag[idg]

                            if adr[idg] <= scale < adr[idg + 1]:
                                # state parameter for scattering
                                # transpose the matrix to get a lower tringaular-like form
                                #pp[nb_params+1], pp[nb_params+2]= idg, iag
                                pp[nb_params+1], pp[nb_params+2]= iag, idg

                                microlibz[isonm][reanm][tuple(pp)] = \
                                    xs[pos_avails_addrzx_i[isoid][reaid] + \
                                       adr[ng] * ianis + scale]
                            #else:
                            # init to 0 the scattering matrix (already done by initdata!)
                            #    microlibz[isonm][reanm][pada] = 0.

            elif "diff" in reanm_l:
                for ianis in range(anis_pl1):
                    #pa = np.append(p, [ianis])
                    pp[nb_params] = ianis
                    pos0 = pos_avails_addrzx_i[isoid][reaid] + ianis * ng
                    microlibz[isonm][reanm][tuple(pp[:-2])] = xs[pos0 : pos0 + ng]
            else:
                pos0 = pos_avails_addrzx_i[isoid][reaid]
                microlibz[isonm][reanm][tuple(pp[:-3])] = xs[pos0 : pos0 + ng]

        # determine the total xs
        pp[nb_params], xsk = 0, "Scattering"
        # warning: remind that the sum is on axis 0 because the scattering
        # matrices are lower triangular-like
        ##scat0 = np.sum(microlibz[ isonm ][ xsk ][tuple(pp[:-2])], axis=0)
        ##microlibz[ isonm ][ "Total" ][tuple(pp[:-3])] = scat0
        ## if "Nxcess" in microlibz[ isonm ]:
        ##     microlibz[ isonm ][ "Total" ][tuple(pp[:-3])] -= \
        ##         microlibz[ isonm ][ "Nxcess" ][tuple(pp[:-3])]
        microlibz[ isonm ][ "Total" ][tuple(pp[:-3])] = \
            microlibz[ isonm ][ "Diffusion" ][tuple(pp[:-2])]
        if "Absorption" in microlibz[ isonm ]:
            microlibz[ isonm ][ "Total" ][tuple(pp[:-3])] += \
                microlibz[ isonm ][ "Absorption" ][tuple(pp[:-3])]
        if anis_pl1 >= 1:
            pp[nb_params] = 1
            scat1 = np.sum(microlibz[ isonm ][ xsk ][tuple(pp[:-2])], axis=0)
            scat1 *= one_third
        else:
            scat1 = np.zeros_like(scat0)
        microlibz[ isonm ][ "DffConstant" ][tuple(pp[:-3])] = one_third / \
            ( microlibz[ isonm ][ "Total" ][tuple(pp[:-3])] - scat1 )
    pass
#-----------------------------------------------------------------


def readMPO(H5file="", vrbs=False, load=True, check=True, save=True,
            remove_time=False, output_nb=-1):
    """Read homog data from the H5file.

    :param H5file: input H5 file
    :param vrbs: verbosity options (True for more verbosity)
    :param load: load existing pickle lib if existing (True/False)
    :param check: check the dependency of considered parameters with state points and/or zone
    :param save: store data on a serialized pickle file
    :param remove_time: remove Time from the list of independent variables
    :param output_nb: nb of specific OUTPUT folder to fetch data from
    :type H5file: string
    :type vrbs: boolean
    :type load: boolean
    :type check: boolean
    :type save: boolean
    :type remove_time: boolean
    :type output_nb: int
    :return: the dict containing informations in each state point and zone
    :rtype: dict

    parameters demanded as input in the initialization function
    1. nb_pnts
    2. nvalue
    3. nzone
    4. ng
    5. totiso
    6. addrzx
    7. avarea,
    8. anis_pl1
    9. locvaltp
    """
    if check: time_in_readMPO = time.perf_counter()

    if load:
        extpos = H5file.rfind('.')
        pfile  = H5file[:extpos] + ".p"
        if not os.path.exists(pfile):
            lg.warn("The input pickle file %s is missing." %pfile+ \
                    " --> continue with the HDF5 file " + H5file)
        else:
            data  = pickle.load(open(pfile, "rb"))
            ofile = data["H5file"]
            if ofile[:ofile.rfind('.')] != H5file[:H5file.rfind('.')]:
                raise ValueError("returned data of " + ofile + \
                    " that does not match with input " + H5file)
            return data

    if not os.path.exists(H5file):
        raise ValueError("The input HDF5 file %s is missing." % H5file)

    hf = h5py.File(H5file,'r')

    # 0. number of total state points
    nb_pnts = hf[ "parameters/tree/NSTATEPOINT" ][()].item()
    lg.info("The number of state point NSTATEPOINT is {:d}.".format(nb_pnts))

    # 1. state parameters
    nvalue = hf["parameters/info/NVALUE"][()]
    paramname = np.char.strip(hf["parameters/info/PARAMNAME"][()].astype(str))
    paramtype = np.char.strip(hf["parameters/info/PARAMTYPE"][()].astype(str))

    # remove the axis TIME, keep only burnup
    if 'TIME' in paramtype and 'BURNUP' in paramtype and remove_time:
        lg.info("Remove TIME as independent variable.")
        it = np.where(paramtype == 'TIME')[0]
        nvalue, paramname = np.delete(nvalue, it), np.delete(paramname, it)
    else:
        lg.warn("Requested TIME remove but no TIME variable in PARAMNAME")
        remove_time = False

    nb_prod = np.prod(nvalue)
    if nb_pnts != nb_prod:
        lg.info("Tot. Nb. of expected points: %d" % nb_prod)
        lg.warn("Nb of calc. points mismatch. There are "+ \
                "params not used to build the xslib domain.")

    nparams = len(nvalue)
    lg.info("Nb. of values per PARAM: " + str(nvalue))
    
    # 2. number of different output groups
    noutput = hf["output/NOUTPUT"][()].item()
    if output_nb == -1:    
        if noutput > 1:
            raise RuntimeError("Multi-output_x MPO not supported yet." +
                               "Please select an input OUTPUT group.")
        else:
            output_nb = 0
    elif not (0 <= output_nb < noutput):
        raise ValueError("The OUTPUT group nb {:d} is missing".format(
                         output_nb))
        
    # output_nb is used to retrieve common data to all state points
    baseadrx = "output/output_{:d}/".format(output_nb)
    lg.info("-> Process data in group output_{:d}".format(output_nb))
    nb_pntx = len(hf[baseadrx]) - 1  # only in given output_x

    # 3. number of zones
    geoadrx = "geometry/geometry_{:d}".format(output_nb)
    nzone = hf[geoadrx + "/NZONE"][()].item()
    lg.info("The number of zones NZONE is {:d}.".format(nzone))
    # volumes of the homogenization zones in cm2
    nvols = hf[geoadrx + "/ZONEVOLUME"][()].astype(h5float)
    znames = hf[geoadrx + "/ZONE_NAME"][()].astype(str)

    # 4. number of energy groups
    # verification: number of energy groups independent on state points
    if check:
        ng_ls = [len(
            hf[baseadrx + "statept_{:d}/flux/TOTALFLUX".format(isp)][()]
                    ) for isp in range(nb_pntx)]
        # verify if all the ng_ls[i] are the same
        if all(i == ng_ls[0] for i in ng_ls):
            lg.debug("The number of energy groups for all the state" +
                     " points is the same.")
        else:
            raise ValueError("Attention! There are different energy " +
                             "meshes among state points, please check.")

    ng = len(hf[baseadrx + "statept_0/flux/TOTALFLUX"][()])
    lg.info("Nb. of en. groups in the (homog) xslib = {:d}".format(ng))

    # parameters indenpedent with state points and zones
    # which are retrieved directly from the hdf file
    addriso = hf[baseadrx + "info/ADDRISO"][()]
    isotope = hf[baseadrx + "info/ISOTOPE"][()]
    niso = hf[baseadrx + "info/NISO"][()].item()

    # ADDRXS (i) [NREA+3,NISO,NADDRXS] contains the C indexes of
    # the first cross-section value in CROSSECTION (f) and TRANSPROFILE (i) arrays.
    addrxs = hf[baseadrx + "info/ADDRXS"][()]
    reaction = hf[baseadrx + "info/REACTION"][()]
    nrea = hf[baseadrx + "info/NREA"][()].item()
    transprofile = hf[baseadrx + "info/TRANSPROFILE"][()]

    # list of isotope names
    isoname = np.char.strip(hf["contents/isotopes/ISOTOPENAME"][()].astype(str))
    # list of reaction names
    reaname = np.char.strip(hf["contents/reactions/REACTIONAME"][()].astype(str))
    reanmlst = reaname[reaction]

    if vrbs:
        print("ADDRXS, position in ADDRXS/index in CROSSECTION")
        print("*ADDRXS of the 1st isotope in the 1st zone*")
        print('-'.join(["%2d"%i for i in range(np.shape(addrxs)[-1])]))
        print('|'.join(["%2d"%i for i in addrxs[0,0]]) + '\n')
        l=max([len(i) for i in reaname])
        fmt="{:^"+str(l)+"s}  {:s}"
        print("-------------------------------")
        print(fmt.format("Reactions","ADDRXS"))
        print("-------------------------------")
        fmt="{:^"+str(l)+"s}   {:>2d}"
        for i in range(nrea):
            print(fmt.format(reaname[i],addrxs[0,0,i]))
        print("-------------------------------")

    # addrzi: address recovering the isotopes
    # verification: addrzi depends on zone, not on state points
    if check:
        addrzi_ls = np.array([[ \
            int(hf[baseadrx + "statept_{:d}/zone_{:d}/ADDRZI".format(isp, iz)][()])
                           for iz in range(nzone)] for isp in range(nb_pntx)])
        # verify if arrays are the same at all state points
        if (np.diff(np.vstack(addrzi_ls).reshape(len(addrzi_ls),-1),axis=0)==0).all():
            lg.debug("The address used to recover the isotope addresses are the " +
                  "same for all state points.")
        else:
            raise ValueError("The address used to recover the isotope " +
                             "addresses are not the same for all the state " +
                             "points, please check the MPO!")

    addrzi_ls = np.array([
        int(hf[baseadrx + "statept_0/zone_{:d}/ADDRZI".format(iz)][()])
            for iz in range(nzone)])

    # 5. total isotope id and name: including 0c (zero concentration) and n0c
    totiso = [[] for iz in range(nzone)]
    for iz in range(nzone):
        # two bords of isotope id list ADDRISO[addrzi] and ADDRISO[addrzi + 1]
        isoidlstbord1, isoidlstbord2 = addriso[addrzi_ls[iz]], addriso[addrzi_ls[iz] + 1]

        # isotope id list ISOTOPE[ADDRISO[addrzi] : ADDRISO[addrzi + 1]]
        isoidlst = isotope[isoidlstbord1 : isoidlstbord2]

        # isotope name list ISOTOPENAME_arr[ISOTOPE[ADDRISO[addrzi] : ADDRISO[addrzi + 1]]]
        isonmlst = isoname[isoidlst]

        # store the total isotope names
        totiso[iz] = isonmlst

    # 6. address recovering the xs
    # verification: addrzx depends on zone, not on state points
    # ADDRZX (i) is the C index in the ADDRXS (i) used to recover the cross sections addresses.
    if check:
        addrzx = np.array([ \
            [int(hf[ baseadrx + "statept_{:d}/zone_{:d}/ADDRZX".format(isp, iz)][()])
                for iz in range(nzone)] for isp in range(nb_pntx)])
        # verify if arrays are the same all state points
        if (np.diff(np.vstack(addrzx).reshape(len(addrzx),-1),axis=0) == 0).all():
            lg.debug("The address used to recover the cross section addresses are "+ \
                  "the same for all state points.")
        else:
            raise ValueError("The address used to recover the cross section addresses "+ \
                             "are not the same for all state points, please check!")

    addrzx = np.array([ \
        int(hf[baseadrx + "statept_0/zone_{:d}/ADDRZX".format(iz) ][()])
            for iz in range(nzone)])

    # available index (>=0) in addrxs
    addrzx_len = np.shape(addrxs)[0]

    idx_avails = np.array([[np.argwhere( addrxs[iadz, iiso, :nrea] >= 0 )
                        for iiso in range(niso)] for iadz in range(addrzx_len)])

    # 7. available reaction id + name
    avarea = [[{} for iiso in range(niso)] for iadz in range(addrzx_len)]
    for iadz in range(addrzx_len):
        for iiso in range(niso):
            idx_rea = np.squeeze(idx_avails[iadz][iiso])
            for irea in idx_rea:
                avarea[iadz][iiso][irea] = reanmlst[irea]

    # id and first value in CROSSECTION
    pos_avails = [ [ {} for iz in range(niso) ] for isp in range(addrzx_len) ]
    for iadz in range(addrzx_len):
        for iiso in range(niso):
            idx_rea = np.squeeze(idx_avails[iadz][iiso])
            for irea in idx_rea:
                pos_avails[iadz][iiso][ irea ] = addrxs[iadz, iiso, irea]

    # first value in TRANSPROFILE
    transp_avails = np.array([[np.squeeze(addrxs[iadz, iiso, nrea + 2])
                             for iiso in range(niso)] for iadz in range(addrzx_len)])

    # 8. scattering anisotropy
    # verification: anisotropy order + 1 independent on state points and zones
    if check:
        anis_pl1_ls = np.array([addrxs[addrzx[iz], 0, nrea]
                               for iz in range(nzone)]).flatten()
        # verify if all the anisotropy orders are the same
        if all(i == anis_pl1_ls[0] for i in anis_pl1_ls):
            print("The scattering anisotropy order + 1 is the same "+ \
                  "for all the state points and zones")
        else:
            raise ValueError("There are different scattering anisotropy"+ \
                   " orders among state points and zones, please check.")

    anis_pl1 = addrxs[addrzx[0], 0, nrea]
    lg.info("Order of scattering anisotropy = {:d}".format(anis_pl1 - 1))

    # 9. local value types
    locvaltp = np.char.strip(hf["/local_values/LOCVALTYPE"][()]).astype(str)
    lg.info("Types of local values: \n" + str(locvaltp))

    data = initdata(nb_pnts, nvalue, nzone, ng, totiso, addrzx, avarea, anis_pl1, locvaltp)

    # verification: it may happen that in some zones, the isotope
    # types change among the state points
    if check:
        nonzero_iso_len_ls = np.array([[np.count_nonzero( \
            hf[ baseadrx + "statept_{:d}/zone_{:d}/CONCENTRATION".format(isp, iz) ][()])
                for iz in range(nzone)] for isp in range(nb_pntx)])

        # boolean value
        iso_zones_boo = np.all(nonzero_iso_len_ls == nonzero_iso_len_ls[0, :], axis=0)

        # invariant zone id
        iso_zones_invar = np.arange(nzone)[iso_zones_boo]
        lg.debug("Out of the total zones:")
        lg.debug("{:d} zones have the same number of isotopes: {:s}".format( \
            np.size(iso_zones_invar), str(iso_zones_invar)))

        # variant zone id
        iso_zones_var = np.arange(nzone)[~iso_zones_boo]
        lg.debug("{:d} zones do not have the same number of isotopes: {:s}".format( \
            np.size(iso_zones_var), str(iso_zones_var)))

    # 10. energy meshes
    energyMeshNames = np.char.strip(hf["energymesh/ENERGYMESH_NAME"][()].astype(str))
    for i, eMname in enumerate(energyMeshNames):
        data["EnMesh_" + eMname] = hf["energymesh/energymesh_%d/ENERGY" % i][()]

    # store the n0c isotopes in all the zones for all the state points
    avaiso = [[[] for iz in range(nzone)] for isp in range(nb_pnts)]

    # store param data
    data["NSTATEPOINT"] = nb_pnts
    data["NVALUE"] = nvalue
    data["PARAMNAME"] = paramname

    for ip, vp in enumerate(paramname):
        data[vp] = hf["/parameters/values/PARAM_{:d}".format(
            ip + 1 if remove_time else ip)][()]
        print("{:>20s}, nb. of values: {:3d}".format(vp, nvalue[ip]))
    if remove_time and vrbs:
        print("Remind that the PARAM Time was removed as independent variable.")

    # store zones data
    data["NZONE"] = nzone
    data["ZONEVOLUME"] = nvols
    data["ZONENAME"] = znames
    # alea...
    data["NG"] = ng
    data["REACTIONAME"] = reaname
    data["ANISOTROPY_PL1"] = anis_pl1

    # add the filename to the output dict
    data["H5file"] = H5file

    microlib = data["microlib"]

    time_stpnts_loop = time.perf_counter()
    lg.info("Start fetching data from state points")
    for isp in range(nb_pnts):
        time_per_isp = time.perf_counter()
        bb = baseadrx + "statept_{:d}".format(isp)

        # state parameters
        p = hf[bb + "/PARAMVALUEORD"][()].astype(np.int8)
        if remove_time:
            p = np.delete(p, 0)  # discard the first index of Time PARAM

        # store the indices of params in case some are not independent
        # from the others
        for r in range(nparams):
            data["PARAM_{:d}".format(r)][isp] = p[r]

        # attempt to avoid multiple np.append in loops
        pp, p = np.insert(np.append(p, 0), 0, 0), tuple(p)

        for iz in range(nzone):
            time_per_zone = time.perf_counter()
            bbz = bb + "/zone_{:d}".format(iz)

            # total isotope concentrations
            totisoconc = hf[bbz + "/CONCENTRATION"][()]
            # n0c isotope id
            avaisoid = np.nonzero(totisoconc)[0]
            # n0c isotope names
            avaisonm = totiso[iz][avaisoid]
            # store in dictionaries
            avaiso[isp][iz] = zip(avaisoid, avaisonm)
            
            # addrzx
            addrzx_i = addrzx[iz] # since it depends only on zone

            # cross section
            xs = hf[bbz + "/CROSSECTION"][()]

            # put xs data in microlib[iz], together with the available concs
            time_microlib_per_zone = time.perf_counter()
            fill_microlibz_v2(microlib[iz], avaiso[isp][iz], \
                avarea[addrzx_i], p, anis_pl1, ng, totisoconc, \
                transp_avails[addrzx_i], pos_avails[addrzx_i], \
                transprofile, xs)
            time_microlib_per_zone = time.perf_counter() - time_microlib_per_zone

            # tuple of iz and p
            pp[0] = iz
            zp = tuple(pp[:-1])
            #zp = tuple(np.append([ iz ], p))

            # zone flux
            data["ZoneFlux"][zp] = hf[bbz + "/ZONEFLUX"][()]

            # zone power
            for ig in range(ng):
                pp[-1] = ig
                zpg = tuple(pp)
                flxzpg = data["ZoneFlux"][zpg]
                data["ZonePower"][zpg] = (RRizg(microlib, flxzpg, p,
                         xstype="FissionEnergyFission", z=iz, g=ig)
                                        + RRizg(microlib, flxzpg, p,
                         xstype="CaptureEnergyCapture", z=iz, g=ig)) * MeV2J

            # zone leakage (sigma-leak as DB2)
            try:
                data["DB2"][zp] = hf[bbz + "/leakage/DB2"][()]
            except KeyError:
                # no leakage model
                pass

            # local value addresses
            locvaladd = hf[bbz + "/LOCVALADDR"][()]

            # browse and fetch all the local value types
            for idt, lvt in enumerate(locvaltp):
                lvd = range(locvaladd[idt], locvaladd[idt + 1])
                len_lvd = len(lvd)
                if len_lvd == 0:
                    # the data is not available in this state point; contrarily to xs
                    # and conc, we set it to 0 since this data is expected anyway
                    data[lvt][zp] = 0.
                elif len_lvd == 1:
                    # we have only one value per zone
                    data[lvt][zp] = hf[bbz + "/LOCALVALUE"][lvd[0]]
                else:
                    raise RuntimeError("This case is not supported yet!")
            
            if check:
                time_per_zone = time.perf_counter() - time_per_zone
                print("{0:>9.6g}s time spent for zone {1:>4d}".format( 
                    time_per_zone, iz) + " ({:5.3f}% in microlibz)".format(
                    time_microlib_per_zone / time_per_zone * 100))

        # total flux
        data["TotalFlux"][p] = hf[bb + "/flux/TOTALFLUX"][()]

        # keff and kinf
        data["keff"][p], data["kinf"][p] = hf[bb + "/addons/KEFF"][()], \
                                           hf[bb + "/addons/KINF"][()]
        keff, kinf  = data["keff"][p], data["kinf"][p]

        # check for the use of a leakage model
        with_leakage_model = not np.isclose(keff, kinf)
        if with_leakage_model:
            # all zones share the same dataset, which should be moved to addons
            data["BUCKLING"][p] = hf[bb + "/zone_0/leakage/BUCKLING"][()]

        if check or vrbs:
            print("time spent in state point {:d} (s): {:f}".format(isp, \
                time.perf_counter() - time_per_isp))
            kinfc = calc_kinf_from_microlib_at_p(data, p)
            dkinf = (kinf - kinfc) * 1.e5
            outpt = (2 * " {:8.6f},"+" {:+8.3f}").format(kinf, kinfc, dkinf)
            if vrbs:
                print("kinf: hdf, computed, diff. in pcm,\n" + outpt)
            if abs(dkinf) > 1.:
                lg.warn("Delta kinf is {:.3f} at state point nb {:d}".format( \
                    dkinf,isp)+'\n'+outpt)
    """
    if check:
        print("+ loop on state points takes %f (s)".format( \
            time.perf_counter() - time_stpnts_loop))
    """

    hf.close()
    if not with_leakage_model:
        if np.nansum(data["BUCKLING"]) > 0:
            raise RuntimeError("leakage model disabled, but BUCKLING > 0!")
        del data["BUCKLING"], data["DB2"]  # to save storage

    # backup data
    if save:
        extpos = H5file.rfind('.')
        pfile  = H5file[:extpos]+".p"
        extpos = H5file.rfind('/')
        # save the pfile locally
        pfile = os.path.dirname(sys.argv[0]) + pfile[extpos:]
        pickle.dump(data, open(pfile, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    if check:
        lg.debug("Total time spent in readMPO (s): {:f}".format( \
            time.perf_counter() - time_in_readMPO))
    return data


if __name__=='__main__':

    # test the module
    # H5file = "UO2_325_AFA3G17_idt_2G.hdf"
    H5file = "/data/tmplca/dtomatis/MPExpCoupling/pwrfuelcell/" \
        + "UO2_325_AFA3G17_idt.hdf"
    # H5file = "e650_gd32_pbp_MPO_20G_FA.hdf"

    data = readMPO(H5file, vrbs=True, load=False, check=True)
