#!/usr/bin/env python3
# --*-- coding:utf-8 --*--
"""
Study and plot the function gamma and its polynomial approximations.
"""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
# possible settings
# https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('font',**{'family':'serif', 'sans-serif':['Helvetica'], 'size': 12})
rc('text', usetex=True)

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, '..'))
from HomogB1FlxCalc import alpha, gamma, gamma_approx, coefs, \
                           gamma_prime, Salpha, Salpha_prime

from itertools import cycle
lines = ["--", "-.", ":"]
linecycler = cycle(lines)

def plot_gamma(B2, coefs, filenm):
    # plots are valid only for S = 1.
    BoS = np.sqrt(abs(B2))
    y = [alpha(v) for v in B2]
    z = [gamma(v) for v in B2]
    s = [[gamma_approx(v, cs=coefs[:i+1]) for v in B2]
         for i in range(len(coefs))]  # approximation
    # print(B2)
    # print(y)
    fig, ax = plt.subplots()
    ax.plot(B2, y, 'C0-', label=r'$\Sigma\alpha$')
    ax.plot(B2, z, 'C1-', label=r'$\gamma$', lw=2)
    for i, v in enumerate(s):
        ax.plot(B2, v, 'C%d' % (i+2) + next(linecycler),
                label=r'$\gamma^{(%d)}$' % (i*2))
    ax.legend(ncol=4, prop={'size': 14}, handletextpad=0.4,
              columnspacing=.8)
    ax.set_xlabel(r'$x^2$', fontsize=14)
    # ax.set_xlabel(r'$(| B | / \Sigma)^2$', fontsize=14)
    # ax.set_xlabel(r'$\lvert B \rvert / \Sigma$')
    ax.set_yscale('log')
    # ax.set_xscale('symlog')
    # plt.show()
    fig.savefig(filenm)


def plot_gamma_prime(B2, coefs, filenm):
    # plots are valid only for S = 1.
    BoS = np.sqrt(abs(B2))
    z = [gamma(v) for v in B2]
    s = [gamma_prime(v) for v in B2]
    # print('B2', B2)
    # print('z', z)
    # print('zp', s)
    fig, ax = plt.subplots()
    ax.plot(B2, z, 'C0-', label=r'$\gamma$', lw=2)
    ax.plot(B2, s, 'C1--', label=r"$\gamma' \Sigma^2$", lw=2)
    ax.legend(ncol=2, prop={'size': 14}, handletextpad=0.4,
              columnspacing=.8)
    ax.set_xlabel(r'$x^2$', fontsize=14)
    # ax.set_xlabel(r'$(| B | / \Sigma)^2$', fontsize=14)
    # ax.set_xlabel(r'$\lvert B \rvert / \Sigma$')
    # ax.set_yscale('symlog')
    # ax.set_xscale('symlog')
    # plt.show()
    fig.savefig(filenm)


if __name__ == "__main__":

    # functions depend on the ratio B / S, with different branches
    # according to the sign of B2; hence we fix S equal to 1 and
    # set the following expected ranges for B2.     
    B2 = np.logspace(-3, 0, 1000)
    B2 = np.append(np.append(-B2[::-1], [0]), B2)
    filenm = os.path.join("..", "..", "docs", "NET2020",
                          "figures", "gamma_poly.pdf")
    plot_gamma(B2, coefs, filenm)
    filenm = os.path.join("..", "..", "docs", "NET2020",
                          "figures", "gamma_prime.pdf")
    plot_gamma_prime(B2, coefs, filenm)