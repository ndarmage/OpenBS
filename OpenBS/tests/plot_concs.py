#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle

# ifile = "RefSteadySolution.p"
ifile = "Ref25pcPowerSolution.p"
with open(ifile, "rb") as f:
    N = pickle.load(f)

t, Nnames = N.t, N.evol_names
# print(t/3600.)
print(N.status, N.message)
nucs = ["XE135", "I135", "U235", "U238"]

t /= 3600.
fig = plt.figure()
ax = fig.subplots()
ax.plot(t, N.y[Nnames == "Xe135",:][0], 'r-', label="Xe135")
ax.plot(t, N.y[Nnames == "I135",:][0], 'b:', label="I135")
ax.set_xlabel(r'$t$ (h)')
ax.set_ylabel(r'conc. (p/barn/cm)')
# ax.set_xscale('symlog')
# ax.plot(t, N.y[Nnames == "U235",:])
# ax.plot(t, N.y[Nnames == "U238",:])
ax.legend()
plt.show()
