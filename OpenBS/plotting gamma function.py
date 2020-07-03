import numpy as np
from matplotlib import pyplot as plt


#first we define the function alpha and beta and then the gamma function which is defined as a function itself of alpha and beta
#alfa function
def alpha(B2,S):
    #z = np.zeros((200))
    BoS = np.sqrt((abs(B2)))/S
    if np.isclose(B2,0):
        a = 1- BoS**2/3 + BoS**4/5 - BoS**6/7
    elif (B2) > 0:
        a = 1/(((BoS))) * (np.arctan(BoS))
    else:
        a = (0.5/((BoS))) * (((np.log((1+BoS))-np.log(1-BoS))))
    return a/S

#beta function
def beta(B2,S):
    return (1-alpha(B2,S)*S)/B2

#defining gamma function
def gamma(B2,S):
    return  1/(3*S) * ((alpha(B2,S))/(beta(B2,S)))

#try
S = 1
B2 =np.logspace(-3,0,100)
B2 = np.append(np.append(-B2[::-1], [0]), B2)

i = [alpha(b,S) for b in B2]
ii = [beta(b,S) for b in B2]
iii = [gamma(b,S) for b in B2]
#print((B2))
#print(i)
#print(ii)
#print(iii)

#plt.plot(B2/S**2,i)
#plt.plot(B2/S**2,ii)
#plt.plot(B2/S**2,iii)



#comparison with appro. polyn.
x = np.array(B2/S**2)

# n=6 (maximum order of the poly)    1 + 4*x**2/15 - 12*x**4/175 + 92*x**6/2625
def ApproX_6(x):
    return    1 + 4*x**1/15 - 12*x**2/175 + 92*x**3/2625


# n=10 (maximum order of the poly)   1 + 4*x**2/15 - 12*x**4/175 + 92*x**6/2625 - 7516*x**8/336875 + 347476*x**10/21896875
def ApproX_10(x):
    return  1 + 4*x**1/15 - 12*x**2/175 + 92*x**3/2625 - 7516*x**4/336875 + 347476*x**5/21896875

#further approximation,
# n=12 (maximum order of the poly)   1 + 4*x**2/15 - 12*x**4/175 + 92*x**6/2625 - 7516*x**8/336875 + 347476*x**10/21896875 - 83263636*x**12/6897515625
def ApproX_12(x):
       return  1 + 4*x**1/15 - 12*x**2/175 + 92*x**3/2625 - 7516*x**4/336875 + 347476*x**5/21896875 - 83263636*x**6/6897515625

plt.figure()
plt.subplot(211)
plt.plot(x,ApproX_6(x),x,iii,"--")
plt.legend(("Gamma","Poly_max_order6"))



plt.subplot(212)
plt.plot(x,ApproX_10(x),x,iii,"--")
plt.legend(("Gamma","Poly_max_order10"))
plt.xlabel("(B/$\Sigma$)^2 for $\Sigma$ = 1 ")


#facultative (Higher order approximation)
#plt.subplot(221)
#plt.plot(x,ApproX_12(x),x,iii,"--")
#plt.legend(("Gamma","Poly_max_order12"))
#plt.xlabel("(B/$\Sigma$)^2 for $\Sigma$ = 1 ")





#Calculate the relative error for each approximation
err_Approx_6= ((ApproX_6(x)-iii)/iii)*100
print(err_Approx_6)

err_Approx_10= ((ApproX_10(x)-iii)/iii)*100
print(err_Approx_10)

plt.figure()
plt.subplot(211)
plt.plot(x,err_Approx_6,color='black',linestyle = 'dashed',label="err_Poly_max_order6")
plt.legend()
plt.ylabel("% error")


plt.subplot(212)
plt.plot(x,err_Approx_10,color='green',linestyle = 'dashed',label = "err_Poly_max_order10")
plt.legend()
plt.xlabel("(B/$\Sigma$)^2 for $\Sigma$ = 1 ")
plt.ylabel("% error")
plt.show()















