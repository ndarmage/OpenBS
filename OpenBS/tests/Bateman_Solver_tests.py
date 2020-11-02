
"""
In Bateman_Solver_tests.py we test our nuclide evolution solver in order to understand
if it correctly resolves the non linear ordinary differential equation system.
Firstly we retrive the nuclear data and then we procede with the tests here summarised:

1)System evolution with time for different power levels (100-75-50-25-0)% Pn

2)Further check of nuclear data--------> nuclide concentration variation with BU

3)Nuclide transient evolution (from full power level to zero power level for different BU points)

4)Nuclide transient evolution (from full power level to low power levels for a fixed BU value)

5)Nuclide transient evoltuion (from low power level to full power level for a fixed BU value)

6)Perturbation analysis on nuclide concentration (few cases)

Note: We have been used some equilibrium concentration as initial concentrations in test 4,5,6
which can be retrived by changing the Pn conditions and Burnup value in previous tests 1,2,3
(those concentrations are anyway given as input values as already saved)

"""


import sys
from OpenBS import *
from HomogB1FlxCalc import *
from analyse_NData import readMPO

import numpy as np
import xarray as xr
import scipy.integrate as ni
import matplotlib.pyplot as plt
#________________________________________________________________________________________________________ START_______________________________________________________________________________________________________________________________

# loading of all data useful for the test module and evaluation of main data functions
xsdir = "lib"
MPOh5 = "UO2_325_AFA3G17_idt_2G_noB2.hdf"
xslib = readMPO(os.path.join(xsdir, MPOh5), vrbs=True, load=False,
                    check=True, save=False)
homogenized_zone = 0
microxs = xslib['microlib'][homogenized_zone]
NG, AO, Vcm2 = xslib['NG'], xslib['ANISOTROPY_PL1'], \
                   xslib['ZONEVOLUME'][homogenized_zone]

microxst = microxs, NG, AO
Bu_pnt, TMod_pnt, Plin_pnt =12,0,-1
Bu, TMod, Plin = xslib['BURN'], xslib['TMod'], xslib['Plin']
PWcm = Plin[Plin_pnt]

p = (Bu_pnt, Bu_pnt, Plin_pnt, TMod_pnt, Plin_pnt)

nuclides_in_MPO = [*microxs]
nchain_file = "lib/DepletionData_CEAV6_CAPTN2N.h5"
nchain, yields = load_nucl_chain(nchain_file, False)
nuclides_in_chain = sorted([*nchain])
evolving_nuclides = list_intersect(nuclides_in_MPO, nuclides_in_chain)

N0 = get_concentrations(microxs, p, evolving_nuclides)
N0not = get_concentrations(microxs, p, evolving_nuclides, False)
N0all = xr.concat([N0, N0not], dim='Nname')
sk = get_Energy_macroxs(N0all, microxst, p)  # MeV/cm
zp = tuple(np.append(homogenized_zone, p))
PWcm_xslib = np.dot(sk * MeV2J, xslib['ZoneFlux'][zp])

HB1LM = not(np.isclose(xslib['kinf'][p], xslib['keff'][p]))
flx_argvs = microxst, p, PWcm_xslib, HB1LM
mxslib, p, P, B2eigv = flx_argvs

microxs, NG, AO = mxslib
microlib, ng, ao = mxslib
chain_data = nchain, yields

nuclide_list = evolving_nuclides
N = np.array([microxs[n]['conc'][p] for n in nuclide_list])
C0, evol_names = N0.values, N0.Nname.values
NArAll = build_N_DataArray(N, evol_names, N0not)
Nnames = evol_names
xname='EvolNuclideConcs'
NAr = xr.DataArray(N, name=xname, dims=('Nname'), coords={'Nname': Nnames})
M =  macroxs(NAr, mxslib, p, xstype='Absorption')
GetM = get_macroxs(NAr, mxslib, p)
GetMEn = get_Energy_macroxs(NAr, mxslib, p)
COMFL = compute_flux(NAr, mxslib, p, P=0., B2eigv=True, vrbs=True)
COMFLvec = np.array(COMFL)
COMFLDer = compute_flx_derivatives(N, evol_names, N0not, flx_argvs, eps=1.e-3)

#Time mesh
tbeg, tend = 0., 3.6e+3 * 24 * 4  # up to four days
Deltat_t = 5. * 60.  # time step in sec
I = int((tend - tbeg) / Deltat_t)  # nb. of times steps
tmesh = np.linspace(tbeg, tend, I + 1)
#_______________________________________________________________________________________________ Set Batemann solver ______________________________________________________________________________________________________________
#1)FIRST TRIAL (RESOLVING BAT.equation WITH THE FUNCTION WRAPPED)
eigv, flx = compute_flux(NArAll, mxslib, p, P, HB1LM)

def dN_dt_wrapped(t, N):
     return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx=flx,adjoint=False)

#Radau Method uses a cubic polynomial for the dense output(this may affect the solution proximate at zero (negative values encountered)-check it out
Sol = ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], C0, method="Radau",t_eval=tmesh)
Sy = Sol.y
S1vec = np.array(Sy)

#______________________________________________________________________________________________Alternative way of solving __________________________________________________________________________________________________________________________
#2)ALTERNATIVE METHOD WITH VECTORIZATION BUT SAME RESULT
   #def Sol(t):
   #return ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], C0, method="Radau",t_eval=tmes)

#t =np.linspace(tbeg,tend,5000)
#Solver = np.vectorize(Sol(t))


#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#________________________________________________________________________________Show results for isotopes Xe135,I135,Sm149,Pm149 for Pn =100%___________________________________________________________________________________
#100 %  Full power level
S1VECiodine = S1vec[60]
np.savetxt('Iodine concentration time steps 100%Pn ',S1VECiodine)

S1VECXenon = S1vec[149]
np.savetxt('Xenon concentration time steps 100%Pn',S1VECXenon)

S1VECSamarium = S1vec[119]
np.savetxt('Samarium concentration time steps 100%Pn',S1VECSamarium)

S1VECPromethium = S1vec[95]
np.savetxt('Promethium concentration time steps 100%Pn',S1VECPromethium)

t = Sol.t
#it's better to express time in terms of hours for nuclides evolution
th = t/3600

yI = S1VECiodine
yXe = S1VECXenon
ySm = S1VECSamarium
yPm = S1VECPromethium

#I135
fig = plt.figure()
plt.plot(th,yI,label ='Iodine concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Xe135
fig2 = plt.figure()
plt.plot(th,yXe,label='Xenon concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Sm149
fig3 = plt.figure()
plt.plot(th,ySm,label='Samarium concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Pm149
fig4 = plt.figure()
plt.plot(th,yPm,label='Promethium concentration with time')
plt.xlabel('Time(hours)')
plt.legend()
plt.show()
#______________________________________________________________________________Bateman Solution for different values of Pn (I135,Xe135,Sm149,Pm149)_________________________________________________________________________________________________
# 0.75 Plin

PWcm_xslibv = [PWcm_xslib*0.75, PWcm_xslib*0.5,PWcm_xslib*0.25,PWcm_xslib*0]
A = PWcm_xslibv[0]
eigv, flx = compute_flux(NArAll, mxslib, p, A, HB1LM)
def dN_dt_wrappe1(t, N):
     return dN_dt(N, t, NArNot=N0not, P=A, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

Sol2= ni.solve_ivp(dN_dt_wrappe1, [tbeg, tend], C0, method="Radau",t_eval=tmesh)
print(Sol2)
Sy = Sol2.y
S2vec = np.array(Sy)

S2VECiodine = S2vec[60]
S2VECXenon = S2vec[149]
S2VECSamarium = S2vec[119]
S2VECPromethium= S2vec[95]

np.savetxt('Iodine concentration time steps 75%Pn',S2VECiodine)

np.savetxt('Xenon concentration time steps 75%Pn',S2VECXenon)

np.savetxt('Samarium concentration time steps 75%Pn',S2VECSamarium)

np.savetxt('Promethium concentration time steps 75%Pn',S2VECPromethium)

#PLOTTING CONCENTRATIONS WITH TIME  (for Pn = 75%)
t = Sol.t
yI_75 = S2VECiodine
yXe_75 = S2VECXenon
ySm_75 = S2VECSamarium
yPm_75 = S2VECPromethium

#I135
fig = plt.figure()
plt.plot(th,yI_75,label ='iodine concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Xe135
fig2 = plt.figure()
plt.plot(th,yXe_75,label='Xenon concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Sm149
fig3 = plt.figure()
plt.plot(th,ySm_75,label='Samarium concentration with time')
plt.xlabel('time(hours)')
plt.legend()

#Pm149
fig4 = plt.figure()
plt.plot(th,yPm_75,label='Promethium concentration with time')
plt.xlabel('time(hours)')
plt.legend()
plt.show()

#_____________________________________________________________________________________________
# 0.50 Plin
B = PWcm_xslibv[1]
eigv, flx = compute_flux(NArAll, mxslib, p, B, HB1LM)

def dN_dt_wrappe2(t, N):
     return dN_dt(N, t, NArNot=N0not, P=B, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

Sol3= ni.solve_ivp(dN_dt_wrappe2, [tbeg, tend], C0, method="Radau",t_eval=tmesh)
print(Sol3)
Sy = Sol3.y
S3vec = np.array(Sy)

#verify solution dimensions
print(S3vec)
print(len(S3vec))    #ok, 158
print(S3vec[1])
print(len(S3vec[1])) #ok, 1153


S3VECiodine = S3vec[60]
S3VECXenon = S3vec[149]
S3VECSamarium = S3vec[119]
S3VECPromethium = S3vec[95]

np.savetxt('Iodine concentration time steps 50%Pn',S3VECiodine)

np.savetxt('Xenon concentration time steps 50%Pn',S3VECXenon)

np.savetxt('Samarium concentration time steps 50%Pn',S3VECSamarium)

np.savetxt('Promethium concentration time steps 50%Pn',S3VECPromethium)

#PLOTTING CONCENTRATIONS WITH TIME  (for Pn = 50%)
t = Sol.t
yI_50= S3VECiodine
yXe_50 = S3VECXenon
ySm_50 = S3VECSamarium
yPm_50 = S3VECPromethium

#I135
fig = plt.figure()
plt.plot(th,yI_50,label ='iodine concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Xe135
fig2 = plt.figure()
plt.plot(th,yXe_50,label='Xenon concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Sm149
fig3 = plt.figure()
plt.plot(th,ySm_50,label='Samarium concentration with time')
plt.xlabel('time(hours)')
plt.legend()

#Pm149
fig4 = plt.figure()
plt.plot(th,yPm_50,label='Promethium concentration with time')
plt.xlabel('time(hours)')
plt.legend()
plt.show()
#____________________________________________________________________________________________
# 0.25 Plin
C = PWcm_xslibv[2]
eigv, flx = compute_flux(NArAll, mxslib, p, C, HB1LM)

def dN_dt_wrappe3(t, N):
     return dN_dt(N, t, NArNot=N0not, P=C, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

Sol4= ni.solve_ivp(dN_dt_wrappe3, [tbeg, tend], C0, method="Radau",t_eval=tmesh)
print(Sol4)
Sy = Sol4.y
S4vec = np.array(Sy)

#verify solution dimensions
print(S4vec)
print(len(S4vec))    #ok, 158
print(S4vec[1])
print(len(S4vec[1])) #ok, 1153

S4VECiodine = S4vec[60]
S4VECXenon = S4vec[149]
S4VECSamarium = S4vec[119]
S4VECPromethium= S4vec[95]

np.savetxt('Iodine  concentration time steps 25%Pn',S4VECiodine)

np.savetxt('Xenon concentration time steps 25%Pn',S4VECXenon)

np.savetxt('Samarium concentration time steps 25%Pn',S4VECSamarium)

np.savetxt('Promethium concentration time steps 25%Pn',S4VECPromethium)

#PLOTTING CONCENTRATIONS WITH TIME  (for Pn = 25%)
t = Sol.t
yI_25= S4VECiodine
yXe_25 = S4VECXenon
ySm_25 = S4VECSamarium
yPm_25 = S4VECPromethium

#I135
fig = plt.figure()
plt.plot(th,yI_25,label ='iodine concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Xe135
fig2 = plt.figure()
plt.plot(th,yXe_25,label='Xenon concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Sm149
fig3 = plt.figure()
plt.plot(th,ySm_25,label='Samarium concentration with time')
plt.xlabel('time(hours)')
plt.legend()

#Pm149
fig4 = plt.figure()
plt.plot(th,yPm_25,label='Promethium concentration with time')
plt.xlabel('time(hours)')
plt.legend()
plt.show()
#__________________________________________________________________________________________
# 0.0 Plin

D = PWcm_xslibv[3]
eigv, flx = compute_flux(NArAll, mxslib, p,D, HB1LM)

def dN_dt_wrappe4(t, N):
     return dN_dt(N, t, NArNot=N0not, P=D, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

Sol5= ni.solve_ivp(dN_dt_wrappe4, [tbeg, tend], C0, method="Radau",t_eval=tmesh)
print(Sol5)
Sy = Sol5.y
S5vec = np.array(Sy)

#verify solution dimensions
print(S5vec)
print(len(S5vec))    #ok, 158
print(S5vec[1])
print(len(S5vec[1])) #ok, 1153

S5VECiodine = S5vec[60]
S5VECXenon = S5vec[149]
S5VECSamarium = S5vec[119]
S5VECPromethium = S5vec[95]

np.savetxt('Iodine  concentration time steps 0%Pn',S5VECiodine)

np.savetxt('Xenon concentration time steps 0%Pn',S5VECXenon)

np.savetxt('Samarium concentration time steps 0%Pn',S5VECSamarium)

np.savetxt('Promethium concentration time steps 0%Pn',S5VECPromethium)

#PLOTTING CONCENTRATIONS WITH TIME  (for Pn = 0%)
t = Sol.t
yI_0= S5VECiodine
yXe_0 = S5VECXenon
ySm_0 = S5VECSamarium
yPm_0 = S5VECPromethium

#I135
fig = plt.figure()
plt.plot(th,yI_0,label ='iodine concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Xe135
fig2 = plt.figure()
plt.plot(th,yXe_0,label='Xenon concentration with time')
plt.xlabel('Time(hours)')
plt.legend()

#Sm149
fig3 = plt.figure()
plt.plot(th,ySm_0,label='Samarium concentration with time')
plt.xlabel('time(hours)')
plt.legend()

#Pm149
fig4 = plt.figure()
plt.plot(th,yPm_0,label='Promethium concentration with time')
plt.xlabel('time(hours)')
plt.legend()
plt.show()
#_____________________________________________________________________________________________________Final plot (all together)_________________________________________________________________________________________

#For I135
fig = plt.figure()
t = Sol.t
plt.plot(th,yI,th,yI_75,th,yI_50,th,yI_25,th,yI_0,'b-')
lineObjects=plt.plot(th,yI,th,yI_75,th,yI_50,th,yI_25,th,yI_0,'b-')
plt.xlabel('Time(hours)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend(iter(lineObjects),('100% Pn', '75% Pn','50% Pn','25% Pn','0% Pn'))
plt.title("Time evolution of I135 for Burnup= 300.0MWd_tHM")

#For Xe135
fig2 = plt.figure()
t = Sol.t
plt.plot(th,yXe,th,yXe_75,th,yXe_50,th,yXe_25,th,yXe_0,'b-')
lineObjects=plt.plot(th,yXe,th,yXe_75,th,yXe_50,th,yXe_25,th,yXe_0,'b-')
plt.xlabel('Time(hours)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend(iter(lineObjects),('100% Pn', '75% Pn','50% Pn','25% Pn','0% Pn'))
plt.title("Time evolution of Xe135 for Burnup= 300.0MWd_tHM")

#For Sm149
fig3 = plt.figure()
t = Sol.t
plt.plot(th,ySm,th,ySm_75,th,ySm_50,th,ySm_25,th,ySm_0,'b-')
lineObjects=plt.plot(th,ySm,th,ySm_75,th,ySm_50,th,ySm_25,th,ySm_0,'b-')
plt.xlabel('Time(hours)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend(iter(lineObjects),('100% Pn', '75% Pn','50% Pn','25% Pn','0% Pn'))
plt.title("Time evolution of Sm149 for Burnup= 300.0MWd_tHM")
#For Pm149
fig4 = plt.figure()
t = Sol.t
plt.plot(th,yPm,th,yPm_75,th,yPm_50,th,yPm_25,th,yPm_0,'b-')
lineObjects=plt.plot(th,yPm,th,yPm_75,th,yPm_50,th,yPm_25,th,yPm_0,'b-')
plt.xlabel('Time(hours)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend(iter(lineObjects),('100% Pn', '75% Pn','50% Pn','25% Pn','0% Pn'))
plt.title("Time evolution of Pm149 for Burnup= 300.0MWd_tHM")
plt.show()
#All right
#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#___________________________________________________________________________________________Plot of Xe135,I135,Sm149,Pm149 with Burnup _____________________________________________________________________________________________
#initialization of Burnup vector
a =(np.linspace(0,20,21))
A = a.astype(int)

#Recovering concentrations with Burnup

#Xe135________________________________________________________________________________________
Concentr_atBurnupvalues_Xe = np.zeros((21, 1))
for i in A:
  Bu_pnt = i
  Bu, TMod, Plin = xslib['BURN'], xslib['TMod'], xslib['Plin']
  PWcm = Plin[Plin_pnt]
  p = (Bu_pnt, Bu_pnt, Plin_pnt, TMod_pnt, Plin_pnt)
  N0 = get_concentrations(microxs, p, evolving_nuclides)
  Xevec = np.array(N0[149])
  #np.savetxt('concentrations variation xith burnup values',Xevec)
  Concentr_atBurnupvalues_Xe[i]=Xevec

x = Bu
x = list(x)
x.pop(0)
y = np.array(Concentr_atBurnupvalues_Xe)
y = list(y)
y.pop(0)
plt.figure()

plt.plot((x),y,'v--',label ='Xe135')
plt.xlabel('Burnup (MWd/t_hm)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend()
plt.xscale("log")
plt.title("Burnup evolution of Xe135")
plt.show()

#I135_________________________________________________________________________________________
Concentr_atBurnupvalues_I = np.zeros((21, 1))
for i in A:
  Bu_pnt = i
  Bu, TMod, Plin = xslib['BURN'], xslib['TMod'], xslib['Plin']
  PWcm = Plin[Plin_pnt]
  p = (Bu_pnt, Bu_pnt, Plin_pnt, TMod_pnt, Plin_pnt)
  N0 = get_concentrations(microxs, p, evolving_nuclides)
  Ivec = np.array(N0[60])
  #np.savetxt('concentrations variation xith burnup values',Xevec)
  Concentr_atBurnupvalues_I[i]=Ivec

y1 = np.array(Concentr_atBurnupvalues_I)
y1 =list(y1)
y1.pop(0)

plt.figure()
plt.plot((x),y1,'v--',label ='I135')
plt.xlabel('Burnup (MWd/t_hm)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend()
plt.xscale("log")
plt.title("Burnup evolution of I135")

#plot together I and Xe
plt.figure()
plt.plot((x),y1,'v--',x,y,'^--')
plt.xlabel('Burnup (MWd/t_hm)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects=plt.plot((x),y1,'v--',(x),y,'^--')
plt.legend(iter(lineObjects),('I135','Xe135'))
plt.xscale("log")
plt.title("Burnup evolution of I135 and Xe135")
plt.show()

#Sm149_________________________________________________________________________________________
Concentr_atBurnupvalues_Sm = np.zeros((21, 1))
for i in A:
  Bu_pnt = i
  Bu, TMod, Plin = xslib['BURN'], xslib['TMod'], xslib['Plin']
  PWcm = Plin[Plin_pnt]
  p = (Bu_pnt, Bu_pnt, Plin_pnt, TMod_pnt, Plin_pnt)
  N0 = get_concentrations(microxs, p, evolving_nuclides)
  Smvec = np.array(N0[119])
  #np.savetxt('concentrations variation xith burnup values',Xevec)
  Concentr_atBurnupvalues_Sm[i]=Smvec

y2 = np.array(Concentr_atBurnupvalues_Sm)
y2 =list(y2)
y2.pop(0)

plt.figure()
plt.plot((x),y2,'v--',label ='Sm149')
plt.xlabel('Burnup (MWd/t_hm)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend()
plt.xscale("log")
plt.title("Burnup evolution of Sm149")
plt.show()

#Pm149_________________________________________________________________________________________
Concentr_atBurnupvalues_Pm = np.zeros((21, 1))
for i in A:
  Bu_pnt = i
  Bu, TMod, Plin = xslib['BURN'], xslib['TMod'], xslib['Plin']
  PWcm = Plin[Plin_pnt]
  p = (Bu_pnt, Bu_pnt, Plin_pnt, TMod_pnt, Plin_pnt)
  N0 = get_concentrations(microxs, p, evolving_nuclides)
  Pmvec = np.array(N0[95])
  #np.savetxt('concentrations variation xith burnup values',Xevec)
  Concentr_atBurnupvalues_Pm[i]=Pmvec


y3 = np.array(Concentr_atBurnupvalues_Pm)
y3 =list(y3)
y3.pop(0)

plt.figure()
plt.plot((x),y3,'v--',label ='Pm149')
plt.xlabel('Burnup (MWd/t_hm)')
plt.ylabel('nuclide concentration (n/barn/cm)')
plt.legend()
plt.xscale("log")
plt.title("Burnup evolution of Pm149")
plt.show()

plt.figure()
plt.plot((x),y3,'v--',x,y2,'^--')
plt.xlabel('Burnup (MWd/t_hm)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects=plt.plot((x),y3,'v--',(x),y2,'^--')
plt.legend(iter(lineObjects),('Pm149','Sm149'))
plt.xscale("log")
plt.title("Burnup evolution of Pm149 and Sm149")
plt.show()






#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#________________________________________________________________________________________Let's see how the solver responds to some abruptly power changes_________________________________________________________________________

#files starting with IBu,XeBu,SmBu and PmBu are calculated in advanced by using the previous bateman solver and by changing abruptly the value of Power level (Plin_pnt)
#In this test, we change the power level from 100% to 0%
#Selected BurnUP points are 4 [(300,2000,6000,10000) MWd/t_hm].

#I135________________________________________________________________________________________
fileI7= open('IBu7','r')
A= map(lambda s: s.strip(), fileI7)
A=list(A)
for i in range(0, len(A)):
    A[i] = float(A[i])

fileI12= open('IBu12','r')
B= map(lambda s: s.strip(), fileI12)
B=list(B)
for i in range(0, len(B)):
    B[i] = float(B[i])

fileI16= open('IBU16','r')
C= map(lambda s: s.strip(), fileI16)
C=list(C)
for i in range(0, len(C)):
    C[i] = float(C[i])

fileI20= open('IBU20','r')
D= map(lambda s: s.strip(), fileI20)
D=list(D)
for i in range(0, len(D)):
    D[i] = float(D[i])

#Xe135_____________________________________________________________________________________
fileXe7= open('XeBu7','r')
E= map(lambda s: s.strip(), fileXe7)
E=list(E)
for i in range(0, len(E)):
    E[i] = float(E[i])

fileXe12= open('XeBu12','r')
F= map(lambda s: s.strip(), fileXe12)
F=list(F)
for i in range(0, len(F)):
    F[i] = float(F[i])

fileXe16= open('XeBu16','r')
G= map(lambda s: s.strip(), fileXe16)
G=list(G)
for i in range(0, len(G)):
    G[i] = float(G[i])

fileXe20= open('XeBu20','r')
H= map(lambda s: s.strip(), fileXe20)
H=list(H)
for i in range(0, len(H)):
    H[i] = float(H[i])

#Sm149________________________________________________________________________________________
fileSm7= open('SmBu7','r')
I= map(lambda s: s.strip(), fileSm7)
I=list(I)
for i in range(0, len(I)):
    I[i] = float(I[i])

fileSm12= open('SmBu12','r')
L= map(lambda s: s.strip(), fileSm12)
L=list(L)
for i in range(0, len(L)):
    L[i] = float(L[i])

fileSm16= open('SmBu16','r')
M= map(lambda s: s.strip(), fileSm16)
M=list(M)
for i in range(0, len(M)):
    M[i] = float(M[i])

fileSm20= open('SmBu20','r')
N= map(lambda s: s.strip(), fileSm20)
N=list(N)
for i in range(0, len(N)):
    N[i] = float(N[i])

#Pm149_________________________________________________________________________________________
filePm7= open('PmBu7','r')
J= map(lambda s: s.strip(), filePm7)
J=list(J)
for i in range(0, len(J)):
    J[i] = float(J[i])

filePm12= open('PmBu12','r')
K= map(lambda s: s.strip(), filePm12)
K=list(K)
for i in range(0, len(K)):
    K[i] = float(K[i])

filePm16= open('PmBu16','r')
O= map(lambda s: s.strip(), filePm16)
O=list(O)
for i in range(0, len(O)):
    O[i] = float(O[i])

filePm20= open('PmBu20','r')
P= map(lambda s: s.strip(), filePm20)
P=list(P)
for i in range(0, len(P)):
    P[i] = float(P[i])

#PLOT OF NUCLIDE'S CONCENTRATIONS FROM 100 % TO 0% OF POWER LEVEL FOR DIFFERENT BURNUP VALUES (4 VALUES IN ANALYSIS)
#Xe135 AND I135
fig = plt.figure()
plt.plot(th,A,th,B,th,C,th,D,th,E,th,F,th,G,th,H)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(th,A,th,B,th,C,th,D,th,E,th,F,th,G,th,H)
plt.legend(iter(lineObjects),('Iodine nuclide concentration for BU = 300 MWd/t_hm', 'Iodine nuclide concentration for BU = 2000 MWd/t_hm','Iodine nuclide concentration for BU = 6000 MWd/t_hm','Iodine nuclide concentration for BU = 10000 MWd/t_hm','Xe nuclide concentration for BU = 300 MWd/t_hm','Xe nuclide concentration for BU = 2000 MWd/t_hm','Xe nuclide concentration for BU = 6000 MWd/t_hm','Xe nuclide concentration for BU = 10000 MWd/t_hm'))
plt.title("Time evolution of 135I and 135Xe from 100%PN to 0%PN for different burnup points")
#Sm149
fig2 = plt.figure()
plt.plot(th,I,th,L,th,M,th,N)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects =plt.plot(th,I,th,L,th,M,th,N)
plt.legend(iter(lineObjects),('Sm nuclide concentration for BU = 300 MWd/t_hm','Sm nuclide concentration for BU = 2000 MWd/t_hm','Sm nuclide concentration for BU = 6000 MWd/t_hm','Sm nuclide concentration for BU = 10000 MWd/t_hm'))
plt.title("Time evolution of Sm149 from 100%PN to 0%PN for different burnup points")
#Pm149
fig3 = plt.figure()
plt.plot(th,J,th,K,th,O,th,P)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects =plt.plot(th,J,th,K,th,O,th,P)
plt.legend(iter(lineObjects),('Pm nuclide concentration for BU = 300 MWd/t_hm','Pm nuclide concentration for BU = 2000 MWd/t_hm','Pm nuclide concentration for BU = 6000 MWd/t_hm','Pm nuclide concentration for BU = 10000 MWd/t_hm'))
plt.title("Time evolution of Pm149 from 100%PN to 0%PN for different burnup points")
plt.show()
#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Transient evolution  from full power level to lower power level for BurnUP = 300 MWd/t_hm (Xe,I,Sm,Pm)___________________________________________________________________________________________________________________________
#Files starting with IBu,XeBu,SmBu and PmBu are calculated in advanced by using the previous bateman solver and by changing abruptly the value of Power level (Plin_pnt)
#The number 7 stands for the Bu_pnt which is the one corresponding to 300 MWd/t_hm
fileI7= open('IBu7','r')
A= map(lambda s: s.strip(), fileI7)
A=list(A)
for i in range(0, len(A)):
    A[i] = float(A[i])

fileI7_25 = open('IBu7_25%','r')
A_25= map(lambda s: s.strip(), fileI7_25)
A_25=list(A_25)
for i in range(0, len(A_25)):
    A_25[i] = float(A_25[i])

fileI7_50 = open('IBu7_50%','r')
A_50= map(lambda s: s.strip(), fileI7_50)
A_50=list(A_50)
for i in range(0, len(A_50)):
    A_50[i] = float(A_50[i])

fileI7_75 = open('IBu7_75%','r')
A_75= map(lambda s: s.strip(), fileI7_75)
A_75=list(A_75)
for i in range(0, len(A_75)):
    A_75[i] = float(A_75[i])

figI= plt.figure()
plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.legend(iter(lineObjects),('0%Pn','25%Pn','50%Pn','75%Pn'))
plt.title("Time evolution of 135I from full power level to lower power level for Burnup= 300.0MWd/tHM")
plt.show()
##################################################################################################################################################
#Xe135
fileXe7= open('XeBu7','r')
A= map(lambda s: s.strip(), fileXe7)
A=list(A)
for i in range(0, len(A)):
    A[i] = float(A[i])

fileXe7_25 = open('Xe7_25%','r')
A_25= map(lambda s: s.strip(),fileXe7_25)
A_25=list(A_25)
for i in range(0, len(A_25)):
    A_25[i] = float(A_25[i])

fileXe7_50 = open('Xe7_50%','r')
A_50= map(lambda s: s.strip(), fileXe7_50)
A_50=list(A_50)
for i in range(0, len(A_50)):
    A_50[i] = float(A_50[i])

fileXe7_75 = open('Xe7_75%','r')
A_75= map(lambda s: s.strip(),fileXe7_75)
A_75=list(A_75)
for i in range(0, len(A_75)):
    A_75[i] = float(A_75[i])

figX= plt.figure()
plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.legend(iter(lineObjects),('0%Pn','25%Pn','50%Pn','75%Pn'))
plt.title("Time evolution of Xe135 from full power level to a lower power level for Burnup= 300.0MWd/tHM")
plt.show()
##############################################################TRIAL################################################################################
#Sm149
fileSm7= open('SmBu7','r')
A= map(lambda s: s.strip(), fileSm7)
A=list(A)
for i in range(0, len(A)):
    A[i] = float(A[i])

fileSm7_25 = open('SmBu7_25','r')
A_25= map(lambda s: s.strip(), fileSm7_25)
A_25=list(A_25)
for i in range(0, len(A_25)):
    A_25[i] = float(A_25[i])

fileSm7_50 = open('SmBu7_50','r')
A_50= map(lambda s: s.strip(), fileSm7_50)
A_50=list(A_50)
for i in range(0, len(A_50)):
    A_50[i] = float(A_50[i])

fileSm7_75 = open('SmBu7_75','r')
A_75= map(lambda s: s.strip(), fileSm7_75)
A_75=list(A_75)
for i in range(0, len(A_75)):
    A_75[i] = float(A_75[i])

figSm= plt.figure()
plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.legend(iter(lineObjects),('0%Pn','25%Pn','50%Pn','75%Pn'))
plt.title("Time evolution of Sm149 from full power level to a lower power level for Burnup= 300.0MWd/tHM")
plt.show()

#####################################################################################################################################################
#Pm149
filePm7= open('PmBu7','r')
A= map(lambda s: s.strip(), filePm7)
A=list(A)
for i in range(0, len(A)):
    A[i] = float(A[i])

filePm7_25 = open('PmBu7_25','r')
A_25= map(lambda s: s.strip(), filePm7_25)
A_25=list(A_25)
for i in range(0, len(A_25)):
    A_25[i] = float(A_25[i])

filePm7_50 = open('PmBu7_50','r')
A_50= map(lambda s: s.strip(), filePm7_50)
A_50=list(A_50)
for i in range(0, len(A_50)):
    A_50[i] = float(A_50[i])

filePm7_75 = open('PmBu7_75','r')
A_75= map(lambda s: s.strip(), filePm7_75)
A_75=list(A_75)
for i in range(0, len(A_75)):
    A_75[i] = float(A_75[i])

figPm= plt.figure()
plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,A,t,A_25,t,A_50,t,A_75)
plt.legend(iter(lineObjects),('0%Pn','25%Pn','50%Pn','75%Pn'))
plt.title("Time evolution of Pm149 from full power level to a lower power level for Burnup= 300.0MWd/tHM")
plt.show()
#________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#Now we do the case from different power levels to zero power levels (we have already done it for full power level, let's estend it to the other power levels)_______________________________________________
#FROM Pn*100% to zero_____________________________________________________________________________________________________________________
#Start /Data dedicated to transiet analysis
#Initial concentration is already known
p = list(p)
p[2] = p[4] = 0
p = tuple(p)

#initial concentration CO
Sol1= ni.solve_ivp(dN_dt_wrappe1, [tbeg, tend], C0, method="Radau",t_eval=tmesh)
Sy = Sol1.y
S1vec = np.array(Sy)

S1VECiodine = S1vec[60]
S1VECXenon = S1vec[149]
S1VECSamarium = S1vec[119]
S1VECPromethium = S1vec[95]
#________________________________________________________________________________________________________________________________________
#FROM Pn*75% to zero_____________________________________________________________________________________________________________________
#Start /Data dedicated to transiet analysis
a =(np.linspace(0,157,158))
A = a.astype(int)

S2final = np.empty(158)
for i in A:
     Sforeach = S2vec[i]
     S2final[i]=Sforeach[1152]

def dN_dt_wrappe1(t, N):
     return dN_dt(N, t, NArNot=N0not, P=A, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

#we changed the initial concentration (S2final)
Sol2= ni.solve_ivp(dN_dt_wrappe1, [tbeg, tend], S2final, method="Radau",t_eval=tmesh)
Sy = Sol2.y
S2vec = np.array(Sy)

S2VECiodine = S2vec[60]
S2VECXenon = S2vec[149]
S2VECSamarium = S2vec[119]
S2VECPromethium = S2vec[95]
#__________________________________________________________________________________________________________________________________________
#from Pn*50% to zero________________________________________________________________________________________________________________________
S3final = np.empty(158)
for i in A:
     Sforeach = S3vec[i]
     S3final[i]=Sforeach[1152]

def dN_dt_wrappe1(t, N):
     return dN_dt(N, t, NArNot=N0not, P=A, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

#we changed the initial concentration (S3final)
Sol3= ni.solve_ivp(dN_dt_wrappe1, [tbeg, tend], S3final, method="Radau",t_eval=tmesh)
Sy = Sol3.y
S3vec = np.array(Sy)

S3VECiodine = S3vec[60]
S3VECXenon = S3vec[149]
S3VECSamarium = S3vec[119]
S3VECPromethium = S3vec[95]
#____________________________________________________________________________________________________________________________________________
#from Pn*25% to zero__________________________________________________________________________________________________________________________
S4final = np.empty(158)
for i in A:
     Sforeach = S4vec[i]
     S4final[i]=Sforeach[1152]

def dN_dt_wrappe1(t, N):
     return dN_dt(N, t, NArNot=N0not, P=A, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

#we changed the initial concentration (S4final)
Sol4= ni.solve_ivp(dN_dt_wrappe1, [tbeg, tend], S4final, method="Radau",t_eval=tmesh)
Sy = Sol4.y
S4vec = np.array(Sy)

S4VECiodine = S4vec[60]
S4VECXenon = S4vec[149]
S4VECSamarium = S4vec[119]
S4VECPromethium = S4vec[95]

#From 0% to zero (we just retrive the equlibrium concentration)
S5final = np.empty(158)
for i in A:
     Sforeach = S5vec[i]
     S5final[i]=Sforeach[1152]

#Plottings
fig6= plt.figure()
plt.plot(t,S1VECiodine,t,S2VECiodine,t,S3VECiodine,t,S4VECiodine)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,S1VECiodine,t,S2VECiodine,t,S3VECiodine,t,S4VECiodine)
plt.legend(iter(lineObjects),('100%Pn','75%Pn','50%Pn','25%Pn'))
plt.title("Time evolution of 135I from different power levels to zero power level for Burnup=6000MWd/tHM")

fig7 = plt.figure()
plt.plot(t,S1VECXenon,t,S2VECXenon,t,S3VECXenon,t,S4VECXenon)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,S1VECXenon,t,S2VECXenon,t,S3VECXenon,t,S4VECXenon)
plt.legend(iter(lineObjects),('100%Pn','75%Pn','50%Pn','25%Pn'))
plt.title("Time evolution of Xe135 from different power levels to zero power level for Burnup=6000MWd/tHM")
plt.show()
#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#Now we do the case from low power level to full power level for Xe and I
#The matter here is to retrive the concentrantions after the first transiet [just if Plin is different from the maxium (100%)] and then to see if Bateman does his job correctly for an abrupt power level change from those new stable concentrantions
#NOTE:right before, we started always from the same initial concentrations and we didn't need concentration at new equilibrium point

#retrive concentrations from previous calculations (new equilibrium points)
#S2final,S3final,S4final,S5final
#Once retrived those concentrations, we set the Bateman solver by imposing a sudden increase in power level up to 100% Pn
#up to here the state parameter has power level zero (from previous calculation); Since we want to study the transitory to full power level we set the new value of Pn up to 100 %
p = list(p)
p[2] = p[4] = -1
p = tuple(p)

HB1LM = not(np.isclose(xslib['kinf'][p], xslib['keff'][p]))
flx_argvs = microxst, p, PWcm_xslib, HB1LM
k, flx = compute_flux(NArAll, *flx_argvs, vrbs=False)
#check if the flux is the one which belongs to power value to 100%

def dN_dt_wrapped(t, N):
     return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

#FROM 75%--------------------------------------->100%
S75_100= ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], S2final, method="Radau",t_eval=tmesh)
Sy = S75_100.y
S75_100vec = np.array(Sy)
S75_100VECiodine = S75_100vec[60]
S75_100VECXenon = S75_100vec[149]

#FROM 50%--------------------------------------->100%
S50_100 = ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], S3final, method="Radau",t_eval=tmesh)
Sy = S50_100.y
S50_100vec = np.array(Sy)
S50_100VECiodine = S50_100vec[60]
S50_100VECXenon = S50_100vec[149]

#FROM 25%--------------------------------------->100%
S25_100 = ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], S4final, method="Radau",t_eval=tmesh)
Sy = S25_100.y
S25_100vec = np.array(Sy)
S25_100VECiodine = S25_100vec[60]
S25_100VECXenon = S25_100vec[149]

#FROM 0%--------------------------------------->100%
S0_100 = ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], S5final, method="Radau",t_eval=tmesh)
Sy = S0_100.y
S0_100vec = np.array(Sy)
S0_100VECiodine = S0_100vec[60]
S0_100VECXenon = S0_100vec[149]

A = S75_100VECiodine
B = S50_100VECiodine
C = S25_100VECiodine
D = S0_100VECiodine
E =S75_100VECXenon
F =S50_100VECXenon
G =S25_100VECXenon
H =S0_100VECXenon

#Plotting
fig8= plt.figure()
plt.plot(t,A,t,B,t,C,t,D)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,A,t,B,t,C,t,D)
plt.legend(iter(lineObjects),('75%Pn','50%Pn','25%Pn','0%Pn'))
plt.title("Time evolution of 135I from lower power level to full power level for Burnup=2000MWd/tHM")

fig9 = plt.figure()
plt.plot(t,E,t,F,t,G,t,H)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,E,t,F,t,G,t,H)
plt.legend(iter(lineObjects),('75%Pn','50%Pn','25%Pn','0%Pn'))
plt.title("Time evolution of Xe135 from lower power level to full power level for Burnup=2000MWd/tHM")
plt.show()
#________________________________________________________________________________________________________End of transiet calculation______________________________________________________________________________________________
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#reset state vector (full power level)
Bu_pnt, TMod_pnt, Plin_pnt =12,0,-1
p = (Bu_pnt, Bu_pnt, Plin_pnt, TMod_pnt, Plin_pnt)
#Evolution of Xe135,I135 and Sm149 for different perturbations_______________________________________________________________________
#Let's set the calculation for the time evolution of targets nuclide for different perturbations
#This test is useful to understand qualitatively how concentrations evolve after perturbing the system

DeltaN_U235 = 0.2* C0[140]

DeltaN_I135 = 0.2* C0[60]

DeltaN_U238 = 0.2* C0[143]

#One extra perturbation point (Pm149) to see the effect on Sm149
DeltaN_Pm249 = 0.2* C0[95]

#New C0 vector #Initial perturbed concentration vectors
C0_perturbed1= np.empty(158)
a =(np.linspace(0,157,158))
A = a.astype(int)
for i in A:
     C0_perturbed1[i]=C0[i]
     C0_perturbed1[140]=C0[140]+DeltaN_U235

C0_perturbed2= np.empty(158)
a =(np.linspace(0,157,158))
A = a.astype(int)
for i in A:
     C0_perturbed2[i]=C0[i]
     C0_perturbed2[60]=C0[60]+DeltaN_I135

C0_perturbed3= np.empty(158)
a =(np.linspace(0,157,158))
A = a.astype(int)
for i in A:
     C0_perturbed3[i]=C0[i]
     C0_perturbed3[143]=C0[143]+DeltaN_U238

#One extra concentration vector (Pm149) to see the effect on Sm149 #############################
C0_perturbed4= np.empty(158)
a =(np.linspace(0,157,158))
A = a.astype(int)
for i in A:
     C0_perturbed4[i]=C0[i]
     C0_perturbed4[95]=C0[95]+DeltaN_Pm249

#Now let's set the Bateman Solver
#unperturbed solution
def dN_dt_wrapped(t, N):
     return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

Sol = ni.solve_ivp(dN_dt_wrapped, [tbeg, tend], C0, method="Radau",t_eval=tmesh)
Sy = Sol.y
S1vec = np.array(Sy)

S1VECiodine = S1vec[60]
np.savetxt('Iodine concentration time steps 100%Pn ',S1VECiodine)

S1VECXenon = S1vec[149]
np.savetxt('Xenon concentration time steps 100%Pn',S1VECXenon)

S1VECSamarium = S1vec[119]
np.savetxt('Samarium concentration time steps 100%Pn',S1VECSamarium)

S1VECPromethium = S1vec[95]
np.savetxt('Promethium concentration time steps 100%Pn',S1VECPromethium)

#1st pertubation----> +0.2 N U235
def dN_dt_wrapped(t, N):
     return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

SP1 = ni.solve_ivp(dN_dt_wrapped, [tbeg, tend],C0_perturbed1, method="Radau",t_eval=tmesh)
Sy = SP1.y
S1vec = np.array(Sy)
N_perturbed_I1=S1vec[60]
np.savetxt("Time evolution of I135 with the first perturbation vector",N_perturbed_I1)
N_perturbed_Xe1=S1vec[149]
np.savetxt("Time evolution of Xe135 with the first perturbation vector",N_perturbed_Xe1)
N_perturbed_Sm1= S1vec[119]
np.savetxt("Time evolution of Sm149 with the first perturbation vector",N_perturbed_Sm1)
N_perturbed_Pm1 = S1vec[95]
np.savetxt("Time evolution of Sm149 with the first perturbation vector",N_perturbed_Pm1)

#2nd perturbation -----> + 0.2N I135
def dN_dt_wrapped(t, N):
     return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

SP2= ni.solve_ivp(dN_dt_wrapped, [tbeg, tend],C0_perturbed2, method="Radau",t_eval=tmesh)
Sy = SP2.y
S2vec = np.array(Sy)

N_perturbed_I2=S2vec[60]
np.savetxt("Time evolution of I135 with the second perturbation vector",N_perturbed_I2)
N_perturbed_Xe2=S2vec[149]
np.savetxt("Time evolution of Xe135 with the second perturbation vector",N_perturbed_Xe2)
N_perturbed_Sm2= S2vec[119]
np.savetxt("Time evolution of Sm149 with the second perturbation vector",N_perturbed_Sm2)
N_perturbed_Pm2 = S2vec[95]
np.savetxt("Time evolution of Sm149 with the second perturbation vector",N_perturbed_Pm2)

#3rd perturbation-----> + 0.2 N U238
def dN_dt_wrapped(t, N):
     return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

SP3= ni.solve_ivp(dN_dt_wrapped, [tbeg, tend],C0_perturbed3, method="Radau",t_eval=tmesh)
Sy = SP3.y
S3vec = np.array(Sy)

N_perturbed_I3=S3vec[60]
np.savetxt("Time evolution of I135 with the third perturbation vector",N_perturbed_I3)
N_perturbed_Xe3=S3vec[149]
np.savetxt("Time evolution of Xe135 with the third perturbation vector",N_perturbed_Xe3)
N_perturbed_Sm3= S3vec[119]
np.savetxt("Time evolution of Sm149 with the third perturbation vector",N_perturbed_Sm3)
N_perturbed_Pm3 = S3vec[95]
np.savetxt("Time evolution of Sm149 with the third perturbation vector",N_perturbed_Pm3)


#4th perturbation -------> +0.2 N Pm249  (For samarium analysis only)
def dN_dt_wrapped(t, N):
     return dN_dt(N, t, NArNot=N0not, P=PWcm_xslib, mxslib=microxst, nucl_chain=chain_data, p=p, V=Vcm2,evol_names=evol_names,flx = flx ,adjoint=False)

SP4= ni.solve_ivp(dN_dt_wrapped, [tbeg, tend],C0_perturbed4, method="Radau",t_eval=tmesh)
Sy = SP4.y
S4vec = np.array(Sy)

N_perturbed_I4=S4vec[60]
np.savetxt("Time evolution of I135 with the forth perturbation vector",N_perturbed_I4)
N_perturbed_Xe4=S4vec[149]
np.savetxt("Time evolution of Xe135 with the forth perturbation vector",N_perturbed_Xe4)
N_perturbed_Sm4= S4vec[119]
np.savetxt("Time evolution of Sm149 with the forth perturbation vector",N_perturbed_Sm4)
N_perturbed_Pm4 = S4vec[95]
np.savetxt("Time evolution of Sm149 with the forth perturbation vector",N_perturbed_Pm4)

#Plotting perturbative concentrations for Iodine
fig10 = plt.figure()
plt.plot(t,S1VECiodine,t,N_perturbed_I2)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,S1VECiodine,t,N_perturbed_I2)
plt.legend(iter(lineObjects),('I135 unperturbed state evolution','I135 for increasing +0.2NO(I135)'))
plt.title("Time evolution of I135 for different pertubations")
plt.show()

#Plotting perturbative concentrations for Xenon
fig10 = plt.figure()
plt.plot(t,S1VECXenon,t,N_perturbed_Xe1,t,N_perturbed_Xe2,t,N_perturbed_Xe3)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,S1VECXenon,t,N_perturbed_Xe1,t,N_perturbed_Xe2,t,N_perturbed_Xe3)
plt.legend(iter(lineObjects),('Xe137 unperturbed state evolution','Xe135 for increasing +0.2NO(U235)','Xe135 for increasing +0.2NO(I135)','Xe135 for increasing +0.2NO(U238)'))
plt.title("Time evolution of Xe135 for different pertubations")
plt.show()

#Plotting perturbative concentrations for Samarium (as we can see iodine perturbation has no effect of Sm concentration while the greatest changes are given by the fissiles and the Promethium)
fig10 = plt.figure()
plt.plot(t,S1VECSamarium,t,N_perturbed_Sm1,t,N_perturbed_Sm2,t,N_perturbed_Sm3,t,N_perturbed_Sm4)
plt.xlabel('Time(s)')
plt.ylabel('nuclide concentration (n/barn/cm)')
lineObjects = plt.plot(t,S1VECSamarium,t,N_perturbed_Sm1,t,N_perturbed_Sm2,t,N_perturbed_Sm3,t,N_perturbed_Sm4)
plt.legend(iter(lineObjects),('Sm149 unperturbed state evolution','Sm149 for increasing +0.2NO(U235)','Sm149 for increasing +0.2NO(I135)','Sm149 for increasing +0.2NO(U238)','Sm149 for increasing +0.2NO(Pm149)'))
plt.title("Time evolution of Sm149 for different pertubations")
plt.show()
