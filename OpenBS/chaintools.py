#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
This module contains the tools to retrieve the nuclide chain from external
files. Both HDF5 and text-ascii formats of MENDEL files are supported.
"""
import os
# import platform
import numpy as np
import xarray as xr
import h5py
import logging as lg

__title__ = "chaintools"
__author__ = "D. Tomatis"
__date__ = "26/03/2019"
__version__ = "1.0.0"

int8_nan = np.iinfo(np.int16).min

# read the decay codes
DCodes = np.genfromtxt('lib/decay_reaction_codes.info', delimiter=':',
    dtype=[('Decay', 'U8'), ('DRTYP', np.float16), ('DeltaA', np.int8),
           ('DeltaZ', np.int8), ('DeltaS', np.int8)], max_rows=19,
           missing_values='?', filling_values=int8_nan)


# read the reaction DCodes
RCodes = np.genfromtxt('lib/decay_reaction_codes.info', delimiter=':',
    dtype=[('Decay', 'U8'), ('MT', np.uint8), ('DeltaA', np.int8),
           ('DeltaZ', np.int8), ('DeltaS', np.int8)], skip_header=20,
           missing_values=('?'), filling_values=int8_nan)


# read the file with information about (Z, El, Element)
ZElements = np.genfromtxt('lib/ZElements.dat', delimiter=',',
    dtype=[('Z', 'i4'), ('El', 'U2'), ('Element', 'U16')],
#                          converters={1: lambda s: s.upper()}
                          )


def get_A(name):
    "Get the atomic mass number from the input nuclide name"
    return int(''.join([c for c in name if c.isdigit()]))


def get_S(name):
    "Get the isomeric metastable state."
    Ss, S = name[-1], 0
    if (Ss == 'M'):
        S = 1
    elif (Ss == 'N'):
        S = 2
    elif (not Ss.isdigit()) and (Ss != 'G'):
        raise ValueError('Unknown S state '+Ss)
    return S


def get_Z(e):
    "Get the proton number of the element e."
    return ZElements[ZElements['El'] == e]['Z'][0]


def get_element(name):
    "Get the symbol of the chemical element in upper case letters."
    el = name[:name.index(str(get_A(name)))]
    if len(el) > 1:
        el = el[0] + el[1].lower()
    return el


def get_id(Z, A, S=0):
    "Get the nuclide id from its name."
    return (Z*1000 + A*10 + S)


def get_data_from_line(l):
    "Function for data extraction for decay channels from parents"
    v0, v1, v2 = l.split()  # retreved as string
    v1 = float(v1) if 'e' in v1 else v1
    v2 = float(v2) if 'e' in v2 else int(v2)
    return v0.strip(), v1.strip(), v2


class nuclide:
    "Nuclide object"
    def __init__(self, name='', lmbda=0., nb=0):
        # regularize the elements with training G, that is stable in the chain
        # if name[-1] == 'G':
        #     name = name[:-1]
        self.name = name
        self.Element = get_element(name)
        self.Z, self.A, self.S = get_Z(self.Element), get_A(name), get_S(name)
        self.id = get_id(self.Z, self.A, self.S)
        self.lmbda = lmbda  # decay constant (1/s)
        self.buildup_channels = [] if (nb <= 0) else [None] * nb
        self.removal_channels = None  # to be set on demand

    def set_buildup_channel(self, data, i=0):
        """Input data for a buildup channel is (parent_name,
        type_of_reaction_or_decay, branching_ratio)"""
        self.buildup_channels[i] = data

    def get_parents(self, only_decay=True):
        parents, nb_parents = [], len(self.buildup_channels)
        if nb_parents > 0:
            if only_decay:
                parents = [self.buildup_channels[i][0]
                           for i in range(nb_parents)
                           if 'DRTYP' in self.buildup_channels[i][1]]
            else:
                parents = [self.buildup_channels[i][0]
                           for i in range(nb_parents)]
        return parents

    def get_sons(self, n_dict):
        sons = []
        nuclide_with_buildup_channels = [n_dict[n] for n in n_dict if
                                         (len(n_dict[n].buildup_channels) > 0)]
        for nuclide in nuclide_with_buildup_channels:
            if self.name in nuclide.get_parents():
                sons.append(nuclide.name)
        if len(sons) != len(set(sons)):
            raise RuntimeError('Duplicate sons detected')
        return sons

    def add_removal_channels(self, n_dict):
        """The default attribute buildup_channels contains all channels from
        parent nuclides; this function adds the attribute removal_channels
        to all sons in the input chain n_dict."""
        rchannels = []
        for nname, nuclide in n_dict.items():
            if not(nuclide.buildup_channels) or (nname == self.name):
                continue
            parents = nuclide.get_parents(only_decay=False)
            if self.name in parents:
                idx = parents.index(self.name)
                bchannel = list(nuclide.buildup_channels[idx][1:])
                rchannels.append([nname] + bchannel)

        self.removal_channels = rchannels

    def branching_ratios_sum_to_1(self, nuclides, vrbs=True):
        "Verify that the branching ratios sum to 1 if the nuclide is unstable."
        sons = self.get_sons(nuclides)
        nb_sons, sum_br = len(sons), 1
        if nb_sons > 0:
            bratios = np.zeros((nb_sons),)
            for i, n in enumerate(sons):
                bchannels = nuclides[n].buildup_channels
                nb_buildup_channels = len(bchannels)
                bratios[i] = [bchannels[j][2]
                              for j in range(nb_buildup_channels)
                              if (bchannels[j][0] == self.name)][0]
            sum_br = np.sum(bratios)
        else:
            if vrbs:
                lg.warning('The nuclide %s has no decay sons.' % self.name)
        return sum_br

    def __str__(self):
        o  = "Nuclide: " + self.name
        o += "\nElement: " + self.Element + ", id: %d\n" % self.id
        o += "Z, A, S = %3d, %3d, %3d\n" % (self.Z, self.A, self.S)
        o += "decay const (1/s) = %13.6g\n" % self.lmbda
        if self.buildup_channels:
            o += ("{:^13s}" * 3 + '\n').format(
                "*Parent*", "Buildup Ch.", "Branching R.")
            for i in self.buildup_channels:
                o += (("{:^13s}" * 2) + "{:^13.6g}\n").format(*i)
        if self.removal_channels:
            o += ("{:^13s}" * 3 + '\n').format(
                "*Son*", "Removal Ch.", "Branching R.")
            for i in self.removal_channels:
                o += (("{:^13s}" * 2) + "{:^13.6g}\n").format(*i)
        return o


def isbinary_file(file_name):
    try:
        # isbinary = not 'ASCII text' in sys_command('file '+ifile))
        with open(file_name, 'tr') as check_file:  # try open file in text mode
            check_file.read()
        isbinary = False
    except:  # if fail then file is non-text (binary)
        isbinary = True
    return isbinary


def sys_command(cmd):
    "Get the output of a command executed on the linux sys."
    OS = platform.system()
    if OS == 'Linux':
        p = subprocess.Popen(cmd, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = p.communicate()
    elif OS == 'Windows':
        raise ValueError('Not yet available')
    else:
        raise ValueError('Unsupported OS')
    return output


def parse_ascii_chain(ifile):
    """Parse the ASCII text file containing the data of the nuclide chain with
    the fission yields."""
    with open(ifile, 'r') as f:
        lines = f.readlines()

    nuclides = dict()
    for line in lines:
        if line.strip() == '':
            continue
        elif (len(line.split()) == 1):
            nb_fissile_nuclides = int(line.split()[0])
            break
        if line.startswith('//') or line.startswith('/*'):
            continue

        data = get_data_from_line(line)
        # the first 8 characters identify a nuclide name
        if line[:8].strip() != '':
            nname = data[0]
            nuclides[nname], i = nuclide(*data), 0
        else:
            # the nuclide has already been found and we need to complete
            # its decay data record; data contains then (parent_name,
            # type_of_reaction_or_decay, branching_ratio)
            nuclides[nname].set_buildup_channel(data, i)
            i += 1

    last_lines = lines.index(line)
    fissile_nuclides, fission_products = lines[last_lines+1].split(), []
    if len(fissile_nuclides) != nb_fissile_nuclides:
        raise RuntimeError('nb fissile nuclides mismatch')
    nb_en_groups = int(lines[last_lines+2].split()[0]) - 1
    en_bounds = [float(v) for v in lines[last_lines+3].split()][::-1]
    nb_fission_products = int(lines[last_lines+5].split()[0])
    yields = np.zeros((nb_fission_products,
                       nb_fissile_nuclides,
                       nb_en_groups),)
    fp = -1
    for line in lines[last_lines+7:]:
        if line.strip() == '':
            continue
        vals = line.split()
        if len(vals) == 1:
            g = nb_en_groups - 1
            fission_products.append(vals[0])
            fp += 1
            continue
        else:
            yields[fp, :, g] = np.array([float(v) for v in vals])
            g -= 1
    if fp + 1 != nb_fission_products:
        raise RuntimeError('filling yields ndarray failed')
    if len(fission_products) != nb_fission_products:
        raise RuntimeError('filling fission_products ndarray failed')
    return nuclides, xr.DataArray(yields, name='FissionYields',
        dims=('FissionProducts', 'FissileNuclides', 'YieldEnergyBounds'),
        coords={'FissionProducts': fission_products,
                'FissileNuclides': fissile_nuclides,
                'YieldEnergyBounds': en_bounds[:-1]})


def cast_h5dataset(d, dtype):
    v = d.astype(dtype)
    if dtype is str:
        v = np.char.strip(v)
    return v


def has_yields(y, fp):
    "Check if the chain y contains the yields for the fission products fp."
    if fp in y.FissionProducts.values:
        print(y.sel(FissionProducts=(fp)))
    else:
        print(fp + ': missing yields')
    pass


def read_hdf5_chain(ifile):
    """Get the nuclide chain data with the fission yields from the input HDF5
    file."""
    h5 = h5py.File(ifile, 'r')
    nucl_names = cast_h5dataset(h5['Chain/ISOTOPNAME'][()], str)
    decay_consts = h5['Chain/DECAYCONST'][()]

    nuclides = dict()
    for i, nname in enumerate(nucl_names):
        base = 'Chain/' + nname + '/'
        nb_parents = h5[base + 'NBPARENT'][()].item()
        nuclides[nname] = nuclide(nname, decay_consts[i], nb_parents)
        if nb_parents > 0:
            bchannels = cast_h5dataset(h5[base + 'PARENTNAME'][()], str), \
                        cast_h5dataset(h5[base + 'REACTIONID'][()], str), \
                        h5[base + 'BRANCHRATIO'][()]
            for j, bchannel in enumerate(zip(*bchannels)):
                nuclides[nname].set_buildup_channel(bchannel, j)

    fission_products = cast_h5dataset(h5['Yields/FsPrNames'][()], str)
    fissile_nuclides = cast_h5dataset(h5['Yields/FsIsNames'][()], str)
    en_bounds = h5['Yields/YieldEnMshInMeV'][()]
    nb_fission_products, nb_fissile_nuclides, nb_en_groups = \
        len(fission_products), len(fissile_nuclides), len(en_bounds) - 1
    yields = h5['Yields/FsYields'][()].reshape(
        nb_fission_products, nb_fissile_nuclides, nb_en_groups)

    return nuclides, xr.DataArray(yields, name='FissionYields',
        dims=('FissionProducts', 'FissileNuclides', 'YieldEnergyBounds'),
        coords={'FissionProducts': fission_products,
                'FissileNuclides': fissile_nuclides,
                'YieldEnergyBounds': en_bounds[:-1]})


def load_nucl_chain(ifile='', check_br=True):
    "Read data from input chain of nuclides"
    if not os.path.isfile(ifile):
        raise ValueError('Missing input file with the nuclide chain')

    if isbinary_file(ifile):
        chain_data = read_hdf5_chain(ifile)
    else:
        chain_data = parse_ascii_chain(ifile)

    # verify that the sum of the branching ratios is equal to 1
    if check_br:
        nuclides = chain_data[0]
        for n in nuclides:
            sum_br = nuclides[n].branching_ratios_sum_to_1(nuclides, False)
            if not np.isclose(sum_br, 1.0):
                lg.warning('Sum of branching ratios of %s = %f' % (n, sum_br))
    return chain_data


def compute_EnYconversion(EnMeshFlx_MPO, YieldEnergyBounds):
    """This function computes a matrix providing the projections of different
    energy meshes when the yields constants and the neutron flux do not have
    the same energy mesh."""
    ngf, ngy = len(EnMeshFlx_MPO) - 1, len(YieldEnergyBounds) - 1
    if not np.isclose(EnMeshFlx_MPO[0], YieldEnergyBounds[0]):
        raise ValueError("The input meshes have different upper bounds.")
    if not np.isclose(EnMeshFlx_MPO[-1], YieldEnergyBounds[-1]):
        raise ValueError("The input meshes have different lower bounds.")
    if np.any(EnMeshFlx_MPO < 0.):
        raise ValueError('Mesh 1 has negative values.')
    if np.any(YieldEnergyBounds < 0.):
        raise ValueError('Mesh 2 has negative values.')
    if not np.all(np.diff(EnMeshFlx_MPO) < 0):
        raise ValueError('Mesh 1 is not strictly decreasing')
    if not np.all(np.diff(YieldEnergyBounds) < 0):
        raise ValueError('Mesh 2 is not strictly decreasing')

    # find the groups in mesh 1 holding the bunnds of mesh 2
    if ngf != 2:
        lg.warn('this func is poorly tested for NG != 2')
    C, t, tx = np.zeros((ngy, ngf),), -1, 0.
    for j in range(ngy):
        b = np.where(
            (EnMeshFlx_MPO[:-1] >  YieldEnergyBounds[j+1]) &
            (EnMeshFlx_MPO[1: ] <= YieldEnergyBounds[j+1])
        )[0][0]
        bx = (EnMeshFlx_MPO[b] - YieldEnergyBounds[j+1]) / \
             (EnMeshFlx_MPO[b] - EnMeshFlx_MPO[b+1])

        C[j,t], C[j,b] = tx, bx
        if b > t + 1:
            C[j,t+1:b] = 1.
        t, tx = b, 1. - bx

    if not np.all(np.isclose(np.sum(C, axis=0), np.ones(ngf))):
        raise RuntimeError('Something is wrong, C does not sum to 1 by cols!')
    return C


if __name__ == "__main__":

    # read the nuclide chain
    #nchain_file = "lib/Chain_fuel.CEAV6"
    #nchain_file = "lib/DecayData_CEAV6.h5"
    #nchain_file = "lib/DecayData_CEAV2005_V3.h5"
    nchain_file = "lib/DepletionData_CEAV6_CAPTN2N.h5"
    nchain, yields = load_nucl_chain(nchain_file, False)
    # add removal channels from a nuclide to its sons (for adj. Bateman eqs.)
    for nuclide in nchain.values():
        nuclide.add_removal_channels(nchain)
        # input(str(nuclide))
    nuclides_in_chain = sorted([*nchain])  # fix the order of nuclides
    print("Data correctly retrieved from the nuclide chain " + nchain_file)
    print("There are %d nuclides with buildup channels." %
          len(nuclides_in_chain))
    print("There are %d fissile nuclides with yields." %
          len(yields.FissileNuclides.values))
    print("There are %d fission products produced by yields." %
          len(yields.FissionProducts.values))
