
#!/usr/bin/env python3

"""
Basic description:
Single sided divorce, in which individuals gather information from offspring mass or actuvely through observation to update their belief about their partner's quality.

Parameters of interest:
Informativeness of information, cost of information, and uncertainty in the informational landscape.

Reason for model:
What is the value of information in different soci-info contexts? Wha tis the optimal infromation gathering strategy for decision like divorce?
"""

__appname__ = 'SDivorce.py'
__author__ = 'Tristan JC (tc661@exeter.ac.uk)'
__version__ = '15.0'

## imports ##
from re import I, S
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os
import scipy.stats as sc
import random
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.ticker import MaxNLocator


filename = "test.csv";
    # limits
diffval = 1; diffstrat = 1; difffit = 1; tol = 1e-7;
    # Keys
L = 0; # Low ability male
H = 1; # High ability male
start = 0; # at start of season
mid = 1; # post all information
end = 2; # post divorce decision
HD = 0
    ## Life history parameters
d = 0.2; # probability of death between breeding seasons
B = 0.5; # probability that arrivals to the population are high ability
J = 51; 
a_H = 11
a_L = 1;
sigma = 5;
O = 100
sds = 10; # number of standard deviations we want for mass info grid
N = 20; #number of grid points per standard deviation
    ## Observation parameters ##
Z = 2;
Noise = 1
    ## Costs ##
careRisk = 0.0
obsRisk = 0.0
C=0.005 
frm = 0.5
fq = 1
    # Space definitions
G = int(sds*N*sigma)
G_0 = N*(a_H-a_L);
a_f = fq*sds*sigma+(a_H-a_L)/2 # Female quality, sds*sigma
I = 2*G+G_0+1;

# Proportion of males still single from previous generation
npost = B

q = np.zeros((I-G_0));
q_z = np.zeros((I-G_0, Z));
l = np.zeros((I));
l_z = np.zeros((I, Z));
rho = np.zeros((J,3,2));
a = np.zeros((J,J,2,Z))
a2 = np.zeros((J,J,2,Z))
b = np.zeros((J,J,2))
P = np.zeros((J,J,Z,2))
PKJZ = np.zeros((J,J,Z))
psi = np.zeros((J,2),dtype=int)
x = np.zeros((J,2),dtype=int)
Pij = np.zeros((I,J))
V = np.zeros(J)
R = np.zeros((J,Z))
Vz = np.zeros((J,Z))
HC = np.zeros(J)
HM = np.zeros(J)


def Interpolate(Val, size=J-1):
    """ Linear interpolation """
    n1 = math.floor(Val*size)
    if n1 == size:
        n2=n1;
        p1 = 1;
        p2 = 0;
    else:
        n2 = n1+1;
        p2 = Val*size-n1;
        p1 = 1.-p2;
    
    return n1, n2, p1, p2

def Belief(j, s=J-1):
    """returns state j as a probability for state space size s"""
    return j/s;
    
def init__G_0():
    global G_0, I, q, l, Pij, N, G, q_z, l_z
    ## Matrices ##
    G = int(sds*N*sigma)
    G_0 = N*(a_H-a_L);
    I = 2*G+G_0+1;
    q = np.zeros((I-G_0));
    q_z = np.zeros((I-G_0, Z));
    l = np.zeros((I), dtype= np.float64);
    l_z = np.zeros((I, Z), dtype= np.float64);
    Pij = np.zeros((I,J))
 
def init_l():
    """generates likelihood ratios of high ability for all messages"""
    global l
    for i in range(I):
        
        if 2*G >= i >= G_0:
            l[i] =  q[H_i(i)]/q[i]
        
        for z in range(Z):
            if 2*G >= i >= G_0:
                l_z[i,z] =  q_z[H_i(i),z]/q_z[i,z]

def init_q():	
    """Generates the full message space, q"""
    global q

    K = 0;
    # first is q_i, or q_i after observation when z=Z-1
    for i in range(I-G_0):
        K += np.exp(-(((i-G)/N)**2)/(2.*sigma**2));
    for i in range(I-G_0):
        q[i] = (1./K)*np.exp(-(((i-G)/N)**2)/(2.*sigma**2));

    # # Next are the garbled information sources from observations where z != Z-1
    # for z in range(Z):
    #     # plt.ylim(0,0.01)
    #     q_z[:,z] = (q[:]**(O*z/(Z-1)))
        
    #     q_z[:,z] /= np.sum(q_z[:,z])
    
    # Next are the garbled information sources from observations where z != Z-1
    for z in range(Z):
        if z == 0:
            q_z[:,0] = 1/(I-G_0)
        else:
            K = 0;
            sigma_obs = sigma + Noise
            # first is q_i, or q_i after observation when z=Z-1
            for i in range(I-G_0):
                K += np.exp(-(((i-G)/N)**2)/(2.*sigma_obs**2));
            for i in range(I-G_0):
                q_z[i,z] = (1./K)*np.exp(-(((i-G)/N)**2)/(2.*sigma_obs**2));

def H_i(i):
   return i-G_0

def init_rho():
    """Sets the initial distribution of high and 
    low ability males amongst females"""
    global rho;
    rho[:,:,:] = 0
    for j in range(J):
        rho[0,start,L] = (1.-B)
        rho[J-1,start,H] = (B)

def Observation(j, i, z):
    """Update belief j through message i,z"""
    prior = Belief(j)
    if z == 0 or prior == 0 or prior == 1:
        return prior
    elif i < G_0:
        post = 0
    elif i > 2*G:
        post = 1
    else:
        post = prior*l_z[i,z]/(1-prior+prior*l_z[i,z])

    return post

def Mass(j, i):
    """Update belief j through mass message i"""
    prior = Belief(j)
    if prior == 0 or prior == 1:
        return prior
    if i < G_0:
        post = 0
    elif i > 2*G:
        post = 1
    else:
        post = prior*l[i]/(1-prior+prior*l[i])
    return post

def init_ab():
    """Calculate a and b for finding P(k|j,z,L)"""
    global a, b, a2
    a[:,:,:,:]=0
    b[:,:,:]=0
    for j in range(J):
        for i in range(I):
            for z in range(Z):
                post = Observation(j, i, z)
                n1,n2,p1,p2 = Interpolate(post)

                if i <= 2*G:
                    a[j,n1,L,z] += q_z[i,z]*p1
                    a[j,n2,L,z] += q_z[i,z]*p2
                if i >= G_0:
                    a[j,n1,H,z] += q_z[H_i(i),z]*p1
                    a[j,n2,H,z] += q_z[H_i(i),z]*p2

            post = Mass(j, i)
            n1,n2,p1,p2 = Interpolate(post)

            if i <= 2*G:
                b[j,n1,L] += q[i]*p1
                b[j,n2,L] += q[i]*p2
            if i >= G_0:
                b[j,n1,H] += q[H_i(i)]*p1
                b[j,n2,H] += q[H_i(i)]*p2
    # For Gridz
    for z in range(Z):
        if z > 1:
            for j in range(J):
                for k in range(J):
                    a2[j,k,L,z]=a[j,k,L,z]
                    a2[j,k,H,z]=a[j,k,H,z]

                    for y in range(z-1):
                        a2L = 0
                        a2H = 0
                        for k2 in range(J):
                            a2L += a2[j,k2,L,z]*a[k2,k,L,z]
                            a2H += a2[j,k2,H,z]*a[k2,k,H,z]
                        a2[j,k,L,z] = a2L
                        a2[j,k,H,z] = a2H

                    a[j,k,L,z]=a2[j,k,L,z]
                    a[j,k,H,z]=a2[j,k,H,z]
    

def init_Pkjzs():
    """Calculate the probability that Post is k, given prior is j,
    strategy is z and partner is L or H.
    Also, P(k|j,z) is the transition probability from belief j/J to 
    belief k/J, given observation strategy psi(j/J)=z and 
    offspring mass information. For Bayesian updating from 
    observation and offspring mass for all j and z"""
    global P, PKJZ
    P[:,:,:,:]=0

    for j in range(J):
        for k in range(J):
            for z in range(Z):
                for u in range(J):
                    P[k,j,z,L] += a[j,u,L,z]*b[u,k,L]
                    P[k,j,z,H] += a[j,u,H,z]*b[u,k,H]

                PKJZ[k,j,z] = Belief(j)*P[k,j,z,H] \
                    + (1.-Belief(j))*P[k,j,z,L];

def init_strat():
    """Set initial resident observationa and divorce strategy"""
    global x, psi
    for j in range(math.floor((J-1)/2)):
        x[j,0] = 0.
    psi[0,0]=0

def init_Pij():
    global Pij
    Pij[:,:] = 0

    for j in range(J):
        for i in range(I):
            if i <= 2*G:
                Pij[i,j] += (1-Belief(j))*q[i]
            if i >= G_0:
                Pij[i,j] += (Belief(j))*q[H_i(i)]

def init():
    """"""
    global V, Vz, HC, HM, a_f, q_z, l_z, a, a2, P, PKJZ, R

    print("Initialising Variables")
    init__G_0()
    a_f = fq*sds*sigma+(a_H-a_L)/2
    V = np.zeros(J)
    Vz = np.zeros((J,Z))
    HC = np.zeros(J)
    HM = np.zeros(J)
    # if gridz:
    # q_z = np.zeros((I-G_0, Z));
    # l_z = np.zeros((I, Z));
    # a = np.zeros((J,J,2,Z))
    # a2 = np.zeros((J,J,2,Z))
    # P = np.zeros((J,J,Z,2))
    # PKJZ = np.zeros((J,J,Z))
    # R = np.zeros((J,Z))
    init_strat()
    init_q()
    init_l()
    init_rho()
    init_ab()
    init_Pkjzs()
    init_Pij()
    init_R()

def check0():
    Sum_q = q[:].sum();
    if Sum_q > 1.000001:
        print("Sum q: {}\n".format(Sum_q))
        sys.exit()

def check1():
    Sum_L = 0.;
    Sum_H = 0.;
    for j in range(J):
        Sum_L += rho[j,start,L];
        Sum_H += rho[j,start,H];
    if Sum_L+Sum_H > 1.001:
        print("check1, Sum ~ L: {}\n".format(Sum_L));
        print("check1, Sum ~ H: {}\n".format(Sum_H));
        sys.exit()

def check2():
    Sum_L = 0.;
    Sum_H = 0.;
    for j in range(J):
        Sum_L += rho[j,mid,L];
        Sum_H += rho[j,mid,H];
    
    if Sum_L+Sum_H > 1.001:
        print("check2, Sum ~ L: {}\n".format(Sum_L));
        print("check2, Sum ~ H: {}\n".format(Sum_H));
        sys.exit()

def check3(D_L, D_H, M_L, M_H):
    if D_L + D_H + M_L + M_H > 2.001:
        print("check3, Sum ~ L: {}\n".format(D_L+M_L));
        print("check3, Sum ~ H: {}\n".format(D_H+M_H));
        sys.exit()

def Gen():
    """Calculate eta for a given resident strategy (x,psi), 
    one generation into the future"""
    global rho, psi, x, npost
    rho[:,mid,:] = 0

    check1();

    for k in range(J):
        for j in range(J):
            rho[k,mid,L] += rho[j,start,L]*P[k,j,psi[j,0],L]*(1-obsCost(psi[j,0]))
            rho[k,mid,H] += rho[j,start,H]*P[k,j,psi[j,0],H]*(1-obsCost(psi[j,0]))

    E_L, E_H, D_L, D_H, M_L, M_H = 0, 0, 0, 0, 0, 0

    for j in range(J):
        E_L += rho[j,start,L]*obsCost(psi[j,0])
        E_H += rho[j,start,H]*obsCost(psi[j,0])
        D_L += rho[j,mid,L]*x[j,0]
        D_H += rho[j,mid,H]*x[j,0]
        M_L += rho[j,mid,L]*(1-x[j,0])
        M_H += rho[j,mid,H]*(1-x[j,0])

    check3(D_L, D_H, M_L, M_H)

    alpha_L = (1.-d)*(D_L+E_L) + d*(1-d)*M_L + (1.-B)*d
    alpha_H = (1.-d)*(D_H+E_H) + d*(1.-d)*(M_H) + B*d
    
    npost = alpha_H/(alpha_H+alpha_L)*0.9 + npost*0.1 # to buffer against oscillations and help reach demo. stability
    # npost = alpha_H/(alpha_H+alpha_L)
    j1,j2,s1,s2 = Interpolate(npost)

    for j in range(J): # Surviving pairs
        rho[j,end,L] = rho[j,mid,L]*(1.-x[j,0])*((1-d)**2)
        rho[j,end,H] = rho[j,mid,H]*(1.-x[j,0])*((1-d)**2)
    
    # Females arriving to pairing pool
    rho[j1,end,L] += alpha_L*s1
    rho[j2,end,L] += alpha_L*s2
    rho[j1,end,H] += alpha_H*s1
    rho[j2,end,H] += alpha_H*s2

def diff():
    """Returns total difference in population structure between current
    and last generation."""
    diffval= 0

    # Equation 36
    diffval = (abs(rho[:,start,L] - rho[:,end,L]) + abs(rho[:,start,H] - rho[:,end,H])).sum();
    
    return diffval

def popx():
    """Iterating population states till convergence of eta and rho"""
    global rho
    diffval = 1
    check0();

    while diffval > tol*100:
        Gen()
        print("Line 383, eta: {}".format(npost))
        diffval=diff()

        # prep matrix for next iteration (forgetting previous iteration)
        rho[:,start,:] = rho[:,end,:];
        rho[:,end,:] = 0.;

def init_R():
    global R
    R[:,:] = 0
    for j in range(J):
        for z in range(Z):
            for i in range(I):
                # R[j,z] += ( (1-frm)*i/N + frm*(a_f*(1-C) +\
                #     C*a_f*(1.-z/(Z-1))) )*Pij[i,j]
                R[j,z] += (i/N + a_f*(1-C*(z/(Z-1))))*Pij[i,j] 

def careCost(z):
    obs = z/(Z-1)
    return (1-obs)*careRisk

def obsCost(z):
    obs = z/(Z-1)
    return (obs)*obsRisk

def Best_response():
    """Calculates the best response to a given resident strategy"""
    global Vz, V, x, psi, HC, HD, HM, difffit

    Vz[:,:]=R[:,:]
    k1, k2, u1, u2 = Interpolate(npost)

    HP = u1*V[k1] + u2*V[k2] # Fitness value of being in the pairing pool

    for k in range(J):
        HC[k] = (1.-d)*(d*HP + (1.-d)*V[k])
        HD = (1.-d)*HP
        HM[k] = max(HC[k],HD)
        if HC[k] >= HD:
            x[k,1] = 0
        else:
            x[k,1] = 1

    difffit=0 
    for j in range(J):
        for z in range(Z):
            for k in range(J):
                Vz[j,z] += (1-careCost(z))*(1-obsCost(z))*PKJZ[k,j,z]*HM[k]
        
        psi[j,1] = np.argmax(Vz[j,:])
        difffit += abs(V[j]-Vz[j,psi[j,1]])
        V[j] = Vz[j,psi[j,1]]
    return difffit

def diff3():
    """"""
    dif = 0.;
    for j in range(J):
        # Equation 
        dif += abs(x[j,0] - x[j,1]);
        dif += abs(psi[j,0] - psi[j,1]);
    
    return dif;

def DP_iter():
    """Find the ESS strategies for given context, 
    iterate strategy until b(x[j])=x[j] for all j"""
    global npost
    init()
    INo = 0
    print("Initial Resident Strategy")
    for j in range(J):
        print("(x, /psi)({}) = ({}, {})\n".format(Belief(j), x[j,0], psi[j,0]), end="")    

    diffstrat = 1
    while diffstrat > 0:
        # Generate pop from parameters and strategy x[j]
        init_rho()
        popx()
        print("\nResident eta: {}".format(npost))
        # npost = B

        difffit = 1;
        # iterate back in time to find best response b(x[j]) for all j
        
        # clear V
        V[:]=0;

        while difffit > tol:
            # find optimal strategy one step back;
            difffit = Best_response();
            # print(difffit)
        
        # find difference between best response and resident strategy x
        diffstrat = diff3();

        # Make the best response strategy the new resident strategy
        for j in range(J):
            x[j,0] = x[j,1];
            x[j,1] = 0;
            psi[j,0] = psi[j,1];
            psi[j,1] = 0;
        INo += 1
        print("Strategy Invasion No. {}".format(INo))
        for j in range(J):
            print("(x, /psi)({}) = ({}, {})\n".format(Belief(j), x[j,0], psi[j,0]), end="")       
       
    # for j in range(J):
    #     for k in range(J):
    #             print("j: {}, k: {}".format(j,k), end="");
    #             print(", {}".format(PKJZ[k][j][Z-1]), end="")
    #             print(", {}".format(PKJZ[k][j][Z-5]), end="")
    #             print(", {}".format(PKJZ[k][j][1]), end="")
    #             print(", {}".format(PKJZ[k][j][0]), end="")
    #             print("\n", end="")
    
    print("\nDone.\n")
    return npost, Vz

def Barcode():
    barcode = "d{}B{}sig{}Go{}aF{}J{}G{}Z{}C{}O{}obR{}caR{}Noi{}N{}fq{}/".format(d,B,sigma,G_0,frm,J,G,Z,C,O,obsRisk,careRisk,Noise,N,fq)
    return barcode

def OutCSV(barcode, dir):
    
    os.makedirs(dir+barcode+"/", exist_ok = True)
    np.savetxt(dir+barcode+"/x.csv", x[:,0], delimiter=",")
    np.savetxt(dir+barcode+"/psi.csv", psi[:,0], delimiter=",")
    np.savetxt(dir+barcode+"/Vz.csv", Vz, delimiter=",")
    with open(dir+barcode+"/eta.csv", 'w') as f:
        f.write("{}".format(npost))
        
    with open(dir+barcode+"/PKJZ.csv", 'w') as f:
        f.write("j,k")
        for z in range(Z):
            f.write(",{}".format(z))
        f.write("\n")
        
        for j in range(J):
            for k in range(J):
                    f.write("{},{}".format(j,k));
                    for z in range(Z):
                        f.write(",{}".format(PKJZ[k][j][z]))
                    f.write("\n")

def Run(dir, psi_S, psi_V, x_ESS, i):
    """Generates or loads data for set parameters"""
    global x, psi, Vz, npost
    init()
    barcode=Barcode()
    print(barcode)
    if os.path.exists(dir+barcode):
        #Reads and processes data for plotting if it already exists
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            global npost
            npost = float(line)
            x_ESS.append(npost)
        psi_S[:,i] = np.genfromtxt(dir+barcode+"psi.csv",delimiter=',')
        psi[:,0] = psi_S[:,i]
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv",delimiter=',')
        Vz = np.genfromtxt(dir+barcode+"Vz.csv",delimiter=',')
        for j in range(J):
            psi_V[j,i] = Vz[j,int(psi_S[j,i])] - Vz[j,0]

    else:
        #Runs model with set parameters
        npost, Vz = DP_iter()

        #Saves data
        OutCSV(barcode, dir)

        #Processes data for plotting
        x_ESS.append(npost)
        psi_S[:,i]=psi[:,0]
        for j in range(J):
            psi_V[j,i] = Vz[j,int(psi_S[j,i])] - Vz[j,0]
    
    age(dir)
    ageplt(dir, barcode)
    return psi_S, psi_V, x_ESS

def Sig(dir="./Results_obs/", sigmin=1, sigmax=15, sig_N = 1):
    """Plot ESS x and Value of information against belief"""
    global sigma,barcode
    x_ESS = []
    psi_V = np.zeros((J, (sigmax-sigmin+1)*sig_N))
    psi_S = np.zeros((J, (sigmax-sigmin+1)*+sig_N))
    sigs = np.arange(sig_N*(sigmax-sigmin+1))/sig_N+sigmin/sig_N
    
    for i in sigs:
        sigma=i

        
        psi_S, psi_V, x_ESS = Run(dir, psi_S, psi_V, x_ESS, int((i-sigmin/sig_N)*sig_N))
    t = "d = {}, beta = {}, G_0 = {}, a_f = {}, J = {}, G = {}, Z = {}, C = {}".format(d,B,G_0,a_f,J,G,Z,C)
    Plots(psi_S, psi_V, x_ESS, ds=sigs, xlab="\u03C3", dir=dir, title=t, sig_N=sig_N)

def d_(dir="./Results_obs/", dmin=1, dmax=10):
    """Plot ESS x and Value of information against belief"""
    global d,barcode
    x_ESS = []
    psi_V = np.zeros((J,dmax-dmin+1))
    psi_S = np.zeros((J, dmax-dmin+1))
    ds = np.arange(dmax-dmin+1)+1
    
    for i in ds:
        d= i/dmax
        psi_S, psi_V, x_ESS = Run(dir, psi_S, psi_V, x_ESS, i-dmin)
    
    ds = ds/ds.max()
    t = "\u03C3 = {}, B = {}, G_0 = {}, a_f = {}, J = {}, G = {}, Z = {}, C = {}".format(sigma,B,G_0,a_f,J,G,Z,C)
    
    Plots(psi_S, psi_V, x_ESS, ds=ds, xlab="d", dir=dir, title=t)

def B_(dir="./Results_obs/", dmin=0, dmax=10):
    """Plot ESS x and Value of information against belief"""
    global B,barcode
    x_ESS = []
    psi_V = np.zeros((J,dmax-dmin+1))
    psi_S = np.zeros((J, dmax-dmin+1))
    ds = np.arange(dmax-dmin+1)+dmin
    
    for i in ds:
        B= i/dmax
        
        psi_S, psi_V, x_ESS = Run(dir, psi_S, psi_V, x_ESS, i-dmin)

    t = "d = {}, \u03C3 = {}, G_0 = {}, a_f = {}, J = {}, G = {}, Z = {}, C = {}".format(d,sigma,G_0,a_f,J,G,Z,C)
    ds = ds/ds.max()
    Plots(psi_S, psi_V, x_ESS, ds=ds, xlab="B", dir=dir, title=t)

def G_0_(dir="./Results_obs/", dmin=1, dmax=10):
    """Plot ESS x and Value of information against belief"""
    global a_H,barcode, G_0
    x_ESS = []
    psi_V = np.zeros((J,dmax-dmin+1))
    psi_S = np.zeros((J,dmax-dmin+1))
    ds = np.arange(dmax-dmin+1)+dmin
    
    for i in ds:
        a_H= i
        init__G_0()
        
        psi_S, psi_V, x_ESS = Run(dir, psi_S, psi_V, x_ESS, i-dmin)
    
    t = "d = {}, \u03C3 = {}, B = {}, a_f = {}, J = {}, G = {}, Z = {}, C = {}, O = {}".format(d,sigma,B,a_f,J,G,Z,C,O)
    ds = N*(ds-a_L)
    Plots(psi_S, psi_V, x_ESS, ds=ds, xlab="G_0", dir=dir, title=t)

def Plots(psi_S, psi_V, x_ESS, ds, xlab, dir, title, sig_N=1):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] = 24
    fig, axs = plt.subplots(1, 2, figsize=(25,10))
    fig.tight_layout(pad=5.0)
    psi_S[psi_S == 0.0] = np.nan
    psi_V[psi_V == 0.0] = np.nan
    # masked_array = np.ma.array (psi_S, mask=np.isnan(psi_S))
    cmap = plt.cm.get_cmap('YlGn').copy()
    cmap = cmap.reversed()
    cmap.set_bad('grey',1.)
    step = (ds[1]-ds[0])/2

    img = axs[0].imshow(np.flip(psi_S,0)/(Z-1), interpolation='none', extent=[ds.min()-step,ds.max()+step,0,1], cmap= cmap, aspect='auto')
    axs[0].plot(ds, x_ESS, color="red")
    cbar = plt.colorbar(img, ax=axs[0])
    cbar.set_label('Proportion of Time Spent Observing, (z/Z)', rotation=270, labelpad=40, y=0.45)
    axs[0].set_xlabel(xlab)
    axs[0].set_ylabel("Belief (j/J)")

    img2 = axs[1].imshow(np.flip(psi_V,0), extent=[ds.min()-step,ds.max()+step,0,1], cmap= cmap, aspect='auto')
    axs[1].plot(ds, x_ESS, color="red")
    cbar2= plt.colorbar(img2, ax=axs[1])
    cbar2.set_label('Canonical Cost of No Auxilliary Information', rotation=270, labelpad=40, y=0.45)
    axs[1].set_xlabel(xlab)
    axs[1].set_ylabel("Belief (j/J)")
    fig.suptitle(title, fontsize=30)
    plt.savefig(dir+xlab+title+"_result.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def age(dir):
    """Gives a female's life history for a given optised strategy"""
    global rho, psi, x, npost
    init()
    barcode = Barcode();
    check1();
    rho[:,:,:] = 0
    j1,j2,s1,s2 = Interpolate(npost)
    rho[j1,start,L] += (1-npost)*s1
    rho[j2,start,L] += (1-npost)*s2
    rho[j1,start,H] += npost*s1
    rho[j2,start,H] += npost*s2
    x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
    psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")

    # Write initial distribution to file
    with open(dir+barcode+"/age_H_start.csv",'w') as fd:
        fd.write("{}".format(rho[0,start,H]))
        for j in range(1,J):
            fd.write(",{}".format(rho[j,start,H]))
        fd.write("\n")
    with open(dir+barcode+"/age_L_start.csv",'w') as fd:
        fd.write("{}".format(rho[0,start,L]))
        for j in range(1,J):
            fd.write(",{}".format(rho[j,start,L]))
        fd.write("\n")
    with open(dir+barcode+"/age_H_mid.csv",'w') as fd:
        fd.write("{}".format(rho[0,mid,H]))
        for j in range(1,J):
            fd.write(",{}".format(rho[j,mid,H]))
        fd.write("\n")
    with open(dir+barcode+"/age_L_mid.csv",'w') as fd:
        fd.write("{}".format(rho[0,mid,L]))
        for j in range(1,J):
            fd.write(",{}".format(rho[j,mid,L]))
        fd.write("\n")
    
    popsize = [1] # proportion alive on year 1
    max_age = 0
    while popsize[-1] > 0.01: # continue if porportion left alive is greater than 0.01
        max_age += 1 # counter for the number of years have passed
        popsize.append(popsize[-1]*(1-d)) # over winter deaths
        rho[:,mid,:] = 0
        for k in range(J):
            for j in range(J):
                rho[k,mid,L] += rho[j,start,L]*P[k,j,psi[j,0],L]*(1-obsCost(psi[j,0]))
                rho[k,mid,H] += rho[j,start,H]*P[k,j,psi[j,0],H]*(1-obsCost(psi[j,0]))

        E_L, E_H, D_L, D_H, M_L, M_H = 0, 0, 0, 0, 0, 0

        for j in range(J):
            E_L += rho[j,start,L]*obsCost(psi[j,0])
            E_H += rho[j,start,H]*obsCost(psi[j,0])
            D_L += rho[j,mid,L]*x[j,0]
            D_H += rho[j,mid,H]*x[j,0]
            M_L += rho[j,mid,L]*(1-x[j,0])
            M_H += rho[j,mid,H]*(1-x[j,0])

        # 
        phi = (1 - d) * (D_H + D_L) + d

        check3(D_L, D_H, M_L, M_H)

        j1,j2,s1,s2 = Interpolate(npost)

        for j in range(J):
            rho[j,end,L] = rho[j,mid,L]*(1.-x[j,0])*((1-d))
            rho[j,end,H] = rho[j,mid,H]*(1.-x[j,0])*((1-d))
        
        rho[j1,end,L] += (1-npost)*s1*phi
        rho[j2,end,L] += (1-npost)*s2*phi
        rho[j1,end,H] += npost*s1*phi
        rho[j2,end,H] += npost*s2*phi

        # Output
        with open(dir+barcode+"/age_H_start.csv",'a') as fd:
            fd.write("{}".format(rho[0,start,H]))
            for j in range(1,J):
                fd.write(",{}".format(rho[j,start,H]))
            fd.write("\n")
        with open(dir+barcode+"/age_L_start.csv",'a') as fd:
            fd.write("{}".format(rho[0,start,L]))
            for j in range(1,J):
                fd.write(",{}".format(rho[j,start,L]))
            fd.write("\n")
        with open(dir+barcode+"/age_H_mid.csv",'a') as fd:
            fd.write("{}".format(rho[0,mid,H]))
            for j in range(1,J):
                fd.write(",{}".format(rho[j,mid,H]))
            fd.write("\n")
        with open(dir+barcode+"/age_L_mid.csv",'a') as fd:
            fd.write("{}".format(rho[0,mid,L]))
            for j in range(1,J):
                fd.write(",{}".format(rho[j,mid,L]))
            fd.write("\n")
        
        rho[:,start,:] = rho[:,end,:]
        rho[:,end,:] = 0

def pdf_plt(dir, barcode):
    # print pdf
    font = {'size'   : 50}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =55
    plt.rcParams["figure.figsize"] = (15,7)
    plt.plot(np.arange(I-G_0),q[:], label="Low",linewidth=15.0, alpha=0.7)
    plt.plot(np.arange(I-G_0)+G_0,q[:], label="High",linewidth=15.0, alpha=0.7)
    plt.xlabel("Offspring mass")
    # plt.title("G = {}, N = {}, G_0 = {}".format(G,N,G_0))
    plt.ylim(0,0.005)
    leg=plt.legend(title="Male quality")
    leg.get_frame().set_alpha(0)
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    # Remove all borders
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(dir+barcode+"q.pdf", format="pdf", bbox_inches='tight',dpi=600, transparent=True)
    plt.close()

    plt.rc('font', **font)
    plt.plot(np.arange(I-G_0),q_z[:,1], label="Low",linewidth=15.0, alpha=0.7)
    plt.plot(np.arange(I-G_0)+G_0,q_z[:,1], label="High",linewidth=15.0, alpha=0.7)
    plt.xlabel("Social Cue")
    # plt.title("G = {}, N = {}, G_0 = {}".format(G,N,G_0))
    plt.ylim(0,0.005)
    leg=plt.legend(title="Male quality")
    leg.get_frame().set_alpha(0)
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    # Remove all borders
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(dir+barcode+"qz.pdf", format="pdf", bbox_inches='tight',dpi=600, transparent=True)
    plt.close()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

def info_summary_plt(dir, barcode, lw, popsize, max_age, age_L_start, age_H_start, age_L_mid, age_H_mid):
    
    font = {'size'   : 30}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =40

    fig = plt.figure()
    fig.tight_layout()
    fig.set_size_inches(30, 20)
    # fig.suptitle(t, fontsize=30)
    gs = fig.add_gridspec(1, 3, wspace=0)
    
    gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0,0:2], wspace=0, hspace = .05)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0,2], wspace=0, hspace = .5)

    aa = plt.subplot(gs0[:,:])
    aa.plot((age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1), label="L", linewidth=lw, alpha=0.7)
    aa.plot((age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1), label="H", linewidth=lw, alpha=0.7)
    aa.set_ylabel("Probability of Observing")
    aa.legend().set_zorder(1)
    # aa.set_box_aspect(1)
    aa.set_ylim(0,1)
    aa.set_xlabel("Age of female")
    aa.set_xlim(0,max_age)
    # aa.set_title('a', y=0.45, fontsize = 30)

    a1 = plt.subplot(gs1[0,0])
    a1.plot(np.arange(I-G_0)/N,q[:],  linewidth=lw, alpha=0.7)
    a1.plot(np.arange(I-G_0)/N+G_0/N,q[:],  linewidth=lw, alpha=0.7)
    # a1.plot(np.arange(I-G_0)/N,q_z[:,Z-1], linestyle='dashed', linewidth=5, alpha=0.7)
    # a1.plot(np.arange(I-G_0)/N+G_0/N,q_z[:,Z-1], linestyle='dashed',  linewidth=5, alpha=0.7)
    a1.set_xlabel("Mass")
    plt.setp(a1.get_yticklabels(), visible=False)
    a1.set_xticks([G/N, (G+G_0)/N])
    a1.set_box_aspect(1)
    a1.set_title("Offspring Mass PDF", fontsize = 30)

    psi_mat = np.zeros((Z,J))
    x_mat = np.zeros((1,J))
    for j in range(J):
        if x[j,0] == 1:
            x_mat[0,j] = 1
        for z in range(Z):
            if psi[j,0] >= z:
                psi_mat[z,j] = 1
    
    cmap1 = colors.ListedColormap(['white', 'red'])
    cmap2 = colors.ListedColormap(['white', 'green'])
    bounds=[0,1,2]
    norm1 = colors.BoundaryNorm(bounds, cmap1.N)
    norm2 = colors.BoundaryNorm(bounds, cmap2.N)

    a4 = plt.subplot(gs1[1,0])
    a4.imshow(x_mat, extent=[0,1,0,1],cmap=cmap1, norm=norm1)
    a4.set_xlabel("Belief State")
    a4.set_aspect('equal', adjustable='box')
    a4.xaxis.set_major_locator(plt.MaxNLocator(1))
    a4.yaxis.set_major_locator(plt.MaxNLocator(1))
    a4.set_title("Divorce Strategy", fontsize = 30)

    a2 = plt.subplot(gs1[2,0], sharey=a4, sharex=a4)
    a2.imshow(psi_mat[1:,:], extent=[0,1,0,1], origin="lower",cmap=cmap2, norm=norm2)
    a2.set_xlabel("Belief State")
    a2.set_aspect('equal', adjustable='box')
    a2.set_title("Observation Strategy", fontsize = 30)

    fig.savefig(dir+barcode+"Info_summary.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def divorce_plt(dir,barcode,lw,popsize,max_age, age_L_start, age_H_start, age_L_mid, age_H_mid):
    font = {'size'   : 12}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =15
    plt.plot((age_L_mid[1:,:]*x[:,0]).sum(axis=1), linewidth=5, alpha=0.7)
    plt.plot((age_H_mid[1:,:]*x[:,0]).sum(axis=1), linewidth=5, alpha=0.7)
    plt.ylabel("Probability of divorce")
    plt.xlabel("Age of female")
    plt.ylim(0,1)
    plt.xlim(0,max_age)
    plt.savefig(dir+barcode+"DivorceRate.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def matching_plt(dir,barcode,lw,popsize,max_age, age_L_start, age_H_start, age_L_mid, age_H_mid):
    # Distribution of males bonded with females, given female death.
    Lbonded = (age_L_start[1:,:]).sum(axis=1)
    Hbonded = (age_H_start[1:,:]).sum(axis=1)
    # for agei in popsize:
    #     Hdemo[agei] *= (1-d)**agei
    #     Ldemo[agei] *= (1-d)**agei
    # Ldemo /= Ldemo.sum() #L Demographics
    # Hdemo /= Hdemo.sum() #H Demographics
    prop_H=Hbonded[:]/(Lbonded+Hbonded)
    plt.plot(prop_H, linewidth =5, label = "Female perspective", alpha=0.7, color="k")
    plt.plot(prop_H*popsize[:-1], label = "Male perspective", linewidth =5, alpha=0.2, color="k")
    plt.xlim(0,max_age)
    plt.legend().set_zorder(1)
    plt.xlabel("Age of Female")
    plt.ylabel("Probability of Pairing \nwith H male")
    plt.ylim(0,1)
    plt.savefig(dir+barcode+"MatchingRate.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def omnisummary_plt(dir,barcode,lw,popsize,max_age, age_L_start, age_H_start, age_L_mid, age_H_mid):
    #Poster Summary
    font = {'size'   : 25}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =30
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    gs = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.15)
    # gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0:4], wspace=0.25, hspace = .05)
    # gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0,4], wspace=0, hspace = .5)
    
    aa = plt.subplot(gs[0,0])
    aa.plot((age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1), color="teal", label="Low", linewidth=lw, alpha=0.7)
    aa.plot((age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1), color="orange", label="High", linewidth=lw, alpha=0.7)
    aa.set_ylabel("Probability of Observing") # based on prior beliefs
    aa.legend().set_title("Male quality")
    aa.set_ylim(0,1)
    aa.set_xlim(0,max_age-1)
    aa.xaxis.set_major_locator(MaxNLocator(integer=True))
    # aa.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    aa.set_xticklabels([])
    
    ab = plt.subplot(gs[1, 0])
    ab.plot((age_L_mid[1:,:]*x[:,0]).sum(axis=1), color="teal", linewidth=lw, alpha=0.7)
    ab.plot((age_H_mid[1:,:]*x[:,0]).sum(axis=1), color="orange", linewidth=lw, alpha=0.7)
    ab.set_ylabel("Probability of Divorce") # based on posterior beliefs
    ab.set_ylim(0,1)
    ab.set_xlabel("Age of female")
    ab.set_xlim(0,max_age-1)
    ab.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ab.yaxis.set_ticks(np.arange(0, 1.1, 0.1))

    # a1 = plt.subplot(gs1[0,0])
    # a1.plot(np.arange(I-G_0)/N,q[:],  linewidth=lw, alpha=0.7)
    # a1.plot(np.arange(I-G_0)/N+G_0/N,q[:],  linewidth=lw, alpha=0.7)
    # a1.set_xlabel("Mass")
    # plt.setp(a1.get_yticklabels(), visible=False)
    # a1.set_xticklabels([])
    # a1.set_box_aspect(1)
    # a1.set_title("Offspring Mass PDF", fontsize = 30)

    psi_mat = np.zeros((Z,J))
    x_mat = np.zeros((1,J))
    for j in range(J):
        if x[j,0] == 1:
            x_mat[0,j] = 1
        for z in range(Z):
            if psi[j,0] >= z:
                psi_mat[z,j] = 1
    
    from matplotlib import colors
    cmap = colors.ListedColormap(['white', 'gray'])
    bounds=[0,1,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    a4 = plt.subplot(gs[1,1])
    a4.imshow(x_mat, extent=[0,1,0,1],cmap=cmap, norm=norm)
    a4.set_xlabel("Belief State")
    a4.set_aspect('equal', adjustable='box')
    a4.set_yticklabels([])
    # a4.xaxis.set_major_locator(plt.MaxNLocator(1))
    # a4.yaxis.set_major_locator(plt.MaxNLocator(1))
    # a4.set_title("Divorce Strategy", fontsize = 30)

    a2 = plt.subplot(gs[0,1])
    a2.imshow(psi_mat[1:,:], extent=[0,1,0,1], origin="lower",cmap=cmap, norm=norm)
    # a2.set_xlabel("Belief State")
    a2.set_aspect('equal', adjustable='box')
    a2.set_yticklabels([])
    a2.set_xticklabels([])
    # a2.set_title("Observation Strategy", fontsize = 30)

    fig.savefig(dir+barcode+"Poster_summary.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def ageplt(dir, barcode):
    """Plots life results for a given barcode and dir"""
    lw = 10
    popsize = [1]
    max_age = 0
    while popsize[-1] > 0.01:
        max_age += 1
        popsize.append(popsize[-1]*(1-d))

    age_L_start = np.genfromtxt(dir+barcode+"/age_L_start.csv", delimiter=",")
    age_H_start = np.genfromtxt(dir+barcode+"/age_H_start.csv", delimiter=",")

    age_L_mid = np.genfromtxt(dir+barcode+"/age_L_mid.csv", delimiter=",")
    age_H_mid = np.genfromtxt(dir+barcode+"/age_H_mid.csv", delimiter=",")

    pdf_plt(dir,barcode)
    info_summary_plt(dir,barcode,lw,popsize,max_age, age_L_start, age_H_start, age_L_mid, age_H_mid)
    divorce_plt(dir,barcode,lw,popsize,max_age, age_L_start, age_H_start, age_L_mid, age_H_mid)
    matching_plt(dir,barcode,lw,popsize,max_age, age_L_start, age_H_start, age_L_mid, age_H_mid)
    omnisummary_plt(dir,barcode,lw,popsize,max_age, age_L_start, age_H_start, age_L_mid, age_H_mid)

def SimpRun(dir):
    """Generates or loads data for set parameters"""
    global x, psi, Vz, npost
    barcode=Barcode()

    if os.path.exists(dir+barcode):
        #Reads and processes data for plotting if it already exists
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            global npost
            npost = float(line)
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")

    else:
        init__G_0()
        #Runs model with set parameters
        npost, Vz = DP_iter()

        #Saves data
        OutCSV(barcode, dir)
    
    age(dir)
    ageplt(dir, barcode)
    
    return

def NofeedbackRun(dir):
    """Generates or loads data for set parameters"""
    global x, psi, Vz, npost,C

    barcode=Barcode()
    if os.path.exists(dir+barcode):
        #Reads and processes data for plotting if it already exists
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            global npost
            npost = float(line)
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")

    else:
        init__G_0()
        #Runs model with set parameters
        C=0.1
        npost, Vz = DP_iter()

        #Saves data
        OutCSV(barcode, dir)
    
    return

def full_explore(dir="./Results_obs/"):
    """Plot life history for a given parameter space"""
    global a_H,barcode, G_0, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    a_H_space = {11,15} # difference in mean
    sig_space = {5,7} # deviation about the mean
    C_space = {0.005} # Observation opportunity cost
    B_space = {0.5} # Rarity of H males
    d_space = {0.2} # Death rate
    Noi_space = {1} # Noise difference between observations and mass
    OR_space = {0} # Probability of death during observation
    CR_space = {0} # Probability of death during care
    O_space = {1} # Noise difference between high and low observation strategies
    
    for c in C_space:
        C=c
        for o in O_space:
            O = o
            for noi in Noi_space:
                Noise = noi
                for i in a_H_space:
                    a_H= i
                    for s in sig_space:
                        sigma=s
                        for cr in CR_space:
                            careRisk = cr
                            for r in OR_space:
                                obsRisk = r
                                for b in B_space:
                                    B=b
                                    for ds in d_space:
                                        d=ds
                                        init()
                                        SimpRun(dir)

def NoisyQplots(dir="./Results_obs/"):
    """Plot life history for a given parameter space"""
    global a_H, barcode, G_0, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    init()
    SimpRun(dir)

def usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab):
    
    max_age = len(age_H_start[:,0])
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =15
    plt.plot(sig_space, Observed_L, label="L", linewidth=4, alpha=0.7)
    plt.plot(sig_space, Observed_H, label="H", linewidth=4, alpha=0.7)
    # plt.plot(sig_space, Observed, label="Total", linewidth=4, linestyle= "dashed", color="k", alpha=0.7)
    plt.legend().set_zorder(1)
    plt.ylim(0,1)
    plt.xlabel(xlab)
    plt.ylabel("Proportion of seasons observed")
    plt.savefig(dir+"summary.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.savefig(dir+"summary.png", format="png", bbox_inches='tight',dpi=600)
    plt.close()

def canon_plots(dir, max_age, Canons, sig_space, etas, xlab, ylim):
    
        fig, ax1 = plt.subplots()
        max_age = 10
        # max_age = len(Canons[0])
        for i in range(1,max_age+1):
            # col = (0.,1-i/(max_age+1),0.5)
            if i > 1:
                col = 1-((1/(i*i*0.5)))
                col = (col,col,col)
            else:
                col = (0.,0.,0.)
    
            # canon = [can[i] for can in Canons if len(can)>i]
            # ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
           
            if i < max_age:
                canon = [can[i] for can in Canons if len(can)>i]
                ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed",
                  zorder=max_age - i + 1, color = col)
            else:
                canon = [can[i:].sum()/len(can[i:]) for can in Canons if len(can)>i]
                ax1.plot(sig_space[:len(canon)], canon,label="10+",linestyle="dashed",
                  zorder=max_age - i + 1, color = col)
        
        ax2 = ax1.twinx()
        entropy = []
        for i in etas:
            entropy.append(sc.entropy([i,(1-i)],base=2))
        # Total_entropy = []
        ax2.plot(sig_space, entropy, label="Eta", color="red")
        ax2.tick_params(axis='y', labelcolor="red")
        # ax1.legend(title="Female age", loc=7, fontsize='small', fancybox=True).set_zorder(max_age+5)
        ax1.set_xlabel(xlab)
        ax1.set_ylim(0.,ylim)
        ax2.set_ylim(0.,1.)
        ax1.set_xlim(min(sig_space),max(sig_space))
        ax1.set_ylabel("Value of social cue")
        ax2.set_ylabel("Shannon entropy of pairing pool quality", color = "red")
        ax1.set_frame_on(False) # make it transparent
        ax2.set_frame_on(True) # add background
        fig.tight_layout()
        ax2.patch.set_visible(False)
        legend = ax1.legend(
            title="Female age",
            loc=7,
            fontsize='small',
            fancybox=True
        )

        fig.add_artist(legend)
        legend.set_zorder(100)
        for spine in ax1.spines.values():
            spine.set_zorder(max_age+6)
        for spine in ax2.spines.values():
            spine.set_zorder(max_age+7)
    
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax1.set_facecolor((221/255, 255/255, 179/255))
        plt.savefig(dir+"canons.pdf", format="pdf", bbox_inches='tight',dpi=600)
        plt.savefig(dir+"canons.png", dpi=600, bbox_inches="tight")
        plt.close()

def H_Scarcity(dir):
    global a_H, G_0, Z, fq,  sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [s/20 for s in range(21)]
    

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []

    for b in sig_space:
        B=b
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))
        ageplt(dir, barcode)

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)

    # Plot:
    xlab = "Population-level frequency of high-quality males (β)"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.25)

def Variance(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [s/2 for s in range(1,32)]
    print(sig_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []

    for s in sig_space:
        sigma=s
        init()
        barcode=Barcode()
        print(barcode)

        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))
        ageplt(dir, barcode)

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    xlab = "Offspring mass standard deviation"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.25)

def Mean(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    print(sig_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []

    for s in sig_space:
        a_H=1+s
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))
        ageplt(dir, barcode)

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    #     Total_entropy.append(((age_H_start[1:,:]+age_L_start[1:,:])*entropy).sum(axis=1).sum())
    
    xlab = "Mean Difference (δ)"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.25)

def Mortality(dir):
    global a_H, G_0, Z, fq, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [0.001]
    # sig_space += [0.02, 0.04, 0.06, 0.08,0.10, 0.12, 0.14, 0.16, 0.18, 0.2]
    sig_space += [s/40 for s in range(1,41)]
    # sig_space += [s/20 for s in range(1,21)]
    # sig_space += [0.6,0.7,0.8,0.9,1.]
    # sig_space += [0.4,0.6,0.8,1.]
    print(sig_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []

    for s in sig_space:
        d=s
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))
        ageplt(dir, barcode)

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    #     Total_entropy.append(((age_H_start[1:,:]+age_L_start[1:,:])*entropy).sum(axis=1).sum())
    

    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, "Mortality rate (μ)")
    canon_plots(dir, 10, Canons, sig_space, etas, "Mortality rate (μ)", 0.7)

def ORisk(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [s/10000 for s in range(1,11)]
    print(sig_space)

    Observed_L = []
    Observed_H = []
    age_uncert = []
    Observed = []
    Canons = []
    etas = []

    for s in sig_space:
        obsRisk=s
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    # Plot:
    xlab = "Observation Cost ($C_{μ}$)"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.7)

def Carecost(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [s/1000 for s in range(4,15)]
    print(sig_space)

    Observed_L = []
    Observed_H = []
    age_uncert = []
    Observed = []
    Canons = []
    etas = []

    for s in sig_space:
        C=s
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    # Plot:
    xlab = "Observation Cost ($C_{O}$)"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.7)

def FemQual(dir):
    global a_H, G_0, fq, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    # sig_space = [0,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2,2.5,3,3.5,4]
    # sig_space = [0,3,3.5,4]
    # sig_space = [0,4]
    sig_space = [s/10 for s in range(41)]+[4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,10.0]

    print(sig_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []

    for s in sig_space:
        fq=s
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))
        ageplt(dir, barcode)

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    # Plot:
    xlab = "Relative Female Quality (F)"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.7)

def BothCosts(dir):
    global a_H, G_0, fq, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [0,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]
    c_space = [0,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]
    # sig_space = [0,3,3.5,4]
    # sig_space = [0,4]
    print(sig_space)
    print(c_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = np.zeros((len(sig_space),len(c_space)))
    etas = []

    for s in range(len(sig_space)):
        obsRisk=sig_space[s]
        for t in range(len(c_space)):
            C=c_space[t]
            
            init()
            barcode=Barcode()
            print(barcode)


            if not os.path.exists(dir+barcode):
                print("Finding DP solution ~")
                SimpRun(dir)
        
            #Reads and processes data for plotting if it already exists
            x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
            psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
            age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
            age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
            Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
            f = open(dir+barcode+"eta.csv")
            for line in f.readlines():
                etas.append(float(line))
            ageplt(dir, barcode)

            # Calculate total amount of observations
            Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
            Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
            Observed_L.append(Total_Observed_L)
            Observed_H.append(Total_Observed_H)
            Observed.append(Total_Observed_H+Total_Observed_L)
            canon = 0
            for j in range(J):
                canon += age_H_start[0,j]*(Vz[j,psi[j,0]] -Vz[j,0])
                canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            Canons[s,t] = canon # Adaptive value of cue for first time partners 
            
            #plot
            plt.imshow(Canons, origin='lower')
            plt.colorbar().set_label("Adaptive Value of Social Cue in First Year")
            plt.xlabel(unicodeit.replace("Opportunity costs (C_M)"))
            plt.ylabel(unicodeit.replace("Mortality costs (C_μ)"))
            plt.savefig(dir+"canons.pdf", format="pdf", bbox_inches='tight',dpi=600)
            plt.close()

def Reset():
    global filename, d, B, J, a_H, a_L, O, sigma, sds, N
    global a_f, Z, C, O, careRisk, obsRisk, Noise, fq
    global G, G_0, I, diffval, diffstrat, difffit, tol
    global L, H, start, mid, end, HD , q, q_z, l, l_z, rho, a
    global b, P, PKJZ, psi, x, Pij, V, R, Vz, HC, HM
    
        ## Life history parameters
    d = 0.2; # probability of death between breeding seasons
    B = 0.5; # probability that arrivals to the population are high ability
    J = 51; 
    a_H = 11
    a_L = 1;
    sigma = 5;
    O = 100
    sds = 10; # number of standard deviations we want for mass info
    N = 20; #number of grid points per standard deviation
        ## Observation parameters ##
    Z = 2;
    Noise = 1.
        ## Costs ##
    careRisk = 0.0
    obsRisk = 0.0
    C=0.0025 
    fq = 1

def NoFeedback_mean(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk,tol,V
    sig_space = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    print(sig_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []
    for s in sig_space:
        a_H=1+s
        init()


        barcode=Barcode()
        print(barcode)
        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            NofeedbackRun(dir) # find resident strategy state before arrival of low cost cue

        C=0.005
        init_R() #set new offspring mass for using new cue
        difffit = 1;
        # iterate back in time to find best response b(x[j]) for all j
        # clear V
        V[:]=0;
        while difffit > tol:
            # find optimal strategy one step back;
            difffit = Best_response(); #find whether an invading female ought to use the new cue
        # Make the best response strategy the new resident strategy

        for j in range(J):
            x[j,0] = x[j,1];
            x[j,1] = 0;
            psi[j,0] = psi[j,1];
            psi[j,1] = 0;
        
        OutCSV(barcode, dir)
        
        age(dir)
        ageplt(dir, barcode)
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    #     Total_entropy.append(((age_H_start[1:,:]+age_L_start[1:,:])*entropy).sum(axis=1).sum())
    
    # plt.plot(sig_space, Total_entropy)
    # plt.show()
    max_age = len(age_H_start[:,0])
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =15
    plt.plot(sig_space, Observed_L, label="L", linewidth=4, alpha=0.7)
    plt.plot(sig_space, Observed_H, label="H", linewidth=4, alpha=0.7)
    plt.plot(sig_space, Observed, label="Total", linewidth=4, linestyle= "dashed", color="k", alpha=0.7)
    plt.legend().set_zorder(1)
    plt.xlabel("Mean Difference (δ)")
    plt.ylabel("Proportions of seasons observed")
    plt.savefig(dir+"summary.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

  
    fig, ax1 = plt.subplots()
    max_age = 10
    for i in range(1,max_age+1):
        if i > 1:
            col = 1-((1/(i*i*0.5)))
        else:
            col = 0

        # canon = [can[i] for can in Canons if len(can)>i]
        # ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
       
        if i < max_age:
            canon = [can[i] for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
        else:
            canon = [can[i:].sum()/len(can[i:]) for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label="10+",linestyle="dashed", color = (col,col,col))
    
    ax2 = ax1.twinx()
    entropy = []
    for i in etas:
        entropy.append(sc.entropy([i,(1-i)],base=2))
    # Total_entropy = []
    ax2.plot(sig_space, entropy, label="Eta", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax1.legend(title="Female age", loc=7, fontsize='small', fancybox=True).set_zorder(1)
    ax1.set_xlabel("Mean Difference (δ)")
    ax1.set_ylim(0.,0.25)
    ax2.set_ylim(0.,1.)
    ax1.set_xlim(0.,25)
    ax1.set_ylabel("Value of social cue")
    ax2.set_ylabel("Shannon entropy of pairing pool quality", color = "red")
    ax1.set_zorder(1)
    ax1.set_frame_on(False) # make it transparent
    ax2.set_frame_on(True) # add background
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()
    # ax1.set_facecolor((221/255, 255/255, 179/255))
    plt.savefig(dir+"canons.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def NoFeedback_mu(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk,tol,V
    sig_space = [s/1000 for s in range(1,11)]
    print(sig_space)

    C=0.0
    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []
    for s in sig_space:
        obsRisk=s
        init()


        barcode=Barcode()
        print(barcode)
        print("Finding DP solution ~")
        NofeedbackRun(dir) # find resident strategy state before arrival of low cost cue

        C=0.0
        init_R() #set new offspring mass for using new cue
        difffit = 1;
        # iterate back in time to find best response b(x[j]) for all j
        # clear V
        V[:]=0;
        while difffit > tol:
            # find optimal strategy one step back;
            difffit = Best_response(); #find whether an invading female ought to use the new cue
        # Make the best response strategy the new resident strategy

        for j in range(J):
            x[j,0] = x[j,1];
            x[j,1] = 0;
            psi[j,0] = psi[j,1];
            psi[j,1] = 0;
        
        OutCSV(barcode, dir)
        
        age(dir)
        ageplt(dir, barcode)
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_H_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    max_age = len(age_H_start[:,0])
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =15
    plt.plot(sig_space, Observed_L, label="Low", color="teal", linewidth=4, alpha=0.7)
    plt.plot(sig_space, Observed_H, label="High", color="orange", linewidth=4, alpha=0.7)
    # plt.plot(sig_space, Observed, label="All", linewidth=4, color="k", alpha=0.7)
    plt.legend(title="Male quality")
    plt.xlabel("Observation Cost ($C_{μ}$)")
    plt.ylim(0,1)
    plt.ylabel("Proportions of seasons observed")
    plt.savefig(dir+"summary.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

  
    fig, ax1 = plt.subplots()
    max_age = 10
    for i in range(1,max_age+1):
        if i > 1:
            col = 1-((1/(i*i*0.5)))
        else:
            col = 0

        # canon = [can[i] for can in Canons if len(can)>i]
        # ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
       
        if i < max_age:
            canon = [can[i] for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
        else:
            canon = [can[i:].sum()/len(can[i:]) for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label="10+",linestyle="dashed", color = (col,col,col))
    
    ax2 = ax1.twinx()
    entropy = []
    for i in etas:
        entropy.append(sc.entropy([i,(1-i)],base=2))
    # Total_entropy = []
    ax2.plot(sig_space, entropy, label="Eta", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax1.legend(title="Female age", loc=7, fontsize='small', fancybox=True).set_zorder(1)
    ax1.set_xlabel("Observation Cost ($C_{μ}$)")
    ax1.set_ylim(0.,0.25)
    ax2.set_ylim(0.,1.)
    ax1.ticklabel_format(scilimits=(-2,-10))
    # ax1.set_xlim(0.,25)
    ax1.set_ylabel("Value of social cue")
    ax2.set_ylabel("Shannon entropy of pairing pool quality", color = "red")
    ax1.set_zorder(1)
    ax1.set_frame_on(False) # make it transparent
    ax2.set_frame_on(True) # add background
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()
    # ax1.set_facecolor((221/255, 255/255, 179/255))
    plt.savefig(dir+"canons.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def NofeedbackRunC(dir):
    """Generates or loads data for set parameters"""
    global x, psi, Vz, npost,C, obsRisk

    barcode=Barcode()
    if os.path.exists(dir+barcode):
        #Reads and processes data for plotting if it already exists
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            print(line)
            npost = float(line)
            print(float)
            
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")

    else:
        init__G_0()
        #Runs model with set parameters
        obsRisk=0.1
        npost, Vz = DP_iter()

        #Saves data
        OutCSV(barcode, dir)
    
    return

def NoFeedback_Cc(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk,tol,V
    sig_space = [s/1000 for s in range(4,15)]
    print(sig_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []
    for s in sig_space:
        C=s
        init()


        barcode=Barcode()
        print(barcode)
        print("Finding DP solution ~")
        NofeedbackRunC(dir) # find resident strategy state before arrival of low cost cue
        
        obsRisk=0.00
        init_R() #set new offspring mass for using new cue
        difffit = 1;
        # iterate back in time to find best response b(x[j]) for all j
        # clear V
        V[:]=0;
        while difffit > tol:
            # find optimal strategy one step back;
            difffit = Best_response(); #find whether an invading female ought to use the new cue
        # Make the best response strategy the new resident strategy
        eta = npost
        for j in range(J):
            x[j,0] = x[j,1];
            x[j,1] = 0;
            psi[j,0] = psi[j,1];
            psi[j,1] = 0;
        
        OutCSV(barcode, dir)
        
        age(dir)
        ageplt(dir, barcode)
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(eta)

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_H_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    max_age = len(age_H_start[:,0])
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =15
    plt.plot(sig_space, Observed_L, label="Low", color="teal", linewidth=4, alpha=0.7)
    plt.plot(sig_space, Observed_H, label="High", color="orange", linewidth=4, alpha=0.7)
    # plt.plot(sig_space, Observed, label="All", linewidth=4, color="k", alpha=0.7)
    plt.legend(title="Male quality")
    plt.xlabel("Observation Cost ($C_{O}$)")
    plt.ylim(0,1)
    plt.ylabel("Proportions of seasons observed")
    plt.savefig(dir+"summary.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

  
    fig, ax1 = plt.subplots()
    max_age = 10
    for i in range(1,max_age+1):
        if i > 1:
            col = 1-((1/(i*i*0.5)))
        else:
            col = 0

        # canon = [can[i] for can in Canons if len(can)>i]
        # ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
       
        if i < max_age:
            canon = [can[i] for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
        else:
            canon = [can[i:].sum()/len(can[i:]) for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label="10+",linestyle="dashed", color = (col,col,col))
    
    ax2 = ax1.twinx()
    entropy = []
    for i in etas:
        entropy.append(sc.entropy([i,(1-i)],base=2))
    # Total_entropy = []
    ax2.plot(sig_space, entropy, label="Eta", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax1.legend(title="Female age", loc=7, fontsize='small', fancybox=True).set_zorder(1)
    ax1.set_xlabel("Observation Cost ($C_{O}$)")
    ax1.set_ylim(0.,0.25)
    ax2.set_ylim(0.,1.)
    ax1.ticklabel_format(scilimits=(-2,-10))
    # ax1.set_xlim(0.,25)
    ax1.set_ylabel("Value of social cue")
    ax2.set_ylabel("Shannon entropy of pairing pool quality", color = "red")
    ax1.set_zorder(1)
    ax1.set_frame_on(False) # make it transparent
    ax2.set_frame_on(True) # add background
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()
    # ax1.set_facecolor((221/255, 255/255, 179/255))
    plt.savefig(dir+"canons.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def NofeedbackRunvarsigma(dir):
    """Generates or loads data for set parameters"""
    global x, psi, Vz, npost,C, obsRisk

    barcode=Barcode()
    if os.path.exists(dir+barcode):
        #Reads and processes data for plotting if it already exists
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            print(line)
            npost = float(line)
            print(float)
            
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")

    else:
        init__G_0()
        #Runs model with set parameters
        obsRisk=0.1
        npost, Vz = DP_iter()

        #Saves data
        OutCSV(barcode, dir)
    
    return

def NoFeedback_varsigma(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk,tol,V
    sig_space = [s/2 for s in range(-5,5)]

    print(sig_space)

    Observed_L = []
    Observed_H = []
    Observed = []
    Canons = []
    etas = []
    for s in sig_space:
        Noise=s
        init()


        barcode=Barcode()
        print(barcode)
        print("Finding DP solution ~")
        NofeedbackRunC(dir) # find resident strategy state before arrival of low cost cue
        
        obsRisk=0.00
        init_R() #set new offspring mass for using new cue
        difffit = 1;
        # iterate back in time to find best response b(x[j]) for all j
        # clear V
        V[:]=0;
        while difffit > tol:
            # find optimal strategy one step back;
            difffit = Best_response(); #find whether an invading female ought to use the new cue
        # Make the best response strategy the new resident strategy
        eta = npost
        for j in range(J):
            x[j,0] = x[j,1];
            x[j,1] = 0;
            psi[j,0] = psi[j,1];
            psi[j,1] = 0;
        
        OutCSV(barcode, dir)
        
        age(dir)
        ageplt(dir, barcode)
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(eta)

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_H_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    max_age = len(age_H_start[:,0])
    font = {'size'   : 10}
    plt.rc('font', **font)
    plt.rcParams['axes.labelsize'] =15
    plt.plot(sig_space, Observed_L, label="Low", color="teal", linewidth=4, alpha=0.7)
    plt.plot(sig_space, Observed_H, label="High", color="orange", linewidth=4, alpha=0.7)
    # plt.plot(sig_space, Observed, label="All", linewidth=4, color="k", alpha=0.7)
    plt.legend(title="Male quality")
    plt.xlabel("Observation Noise (𝜍)")
    plt.ylim(0,1)
    plt.ylabel("Proportions of seasons observed")
    plt.savefig(dir+"summary.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

  
    fig, ax1 = plt.subplots()
    max_age = 10
    for i in range(1,max_age+1):
        if i > 1:
            col = 1-((1/(i*i*0.5)))
        else:
            col = 0

        # canon = [can[i] for can in Canons if len(can)>i]
        # ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
       
        if i < max_age:
            canon = [can[i] for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label=i,linestyle="dashed", color = (col,col,col))
        else:
            canon = [can[i:].sum()/len(can[i:]) for can in Canons if len(can)>i]
            ax1.plot(sig_space[:len(canon)], canon,label="10+",linestyle="dashed", color = (col,col,col))
    
    ax2 = ax1.twinx()
    entropy = []
    for i in etas:
        entropy.append(sc.entropy([i,(1-i)],base=2))
    # Total_entropy = []
    ax2.plot(sig_space, entropy, label="Eta", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax1.legend(title="Female age", loc=7, fontsize='small', fancybox=True).set_zorder(1)
    ax1.set_xlabel("Observation Noise (𝜍)")
    ax1.set_ylim(0.,0.25)
    ax2.set_ylim(0.,1.)
    ax1.ticklabel_format(scilimits=(-2,-10))
    # ax1.set_xlim(0.,25)
    ax1.set_ylabel("Value of social cue")
    ax2.set_ylabel("Shannon entropy of pairing pool quality", color = "red")
    ax1.set_zorder(1)
    ax1.set_frame_on(False) # make it transparent
    ax2.set_frame_on(True) # add background
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()
    # ax1.set_facecolor((221/255, 255/255, 179/255))
    plt.savefig(dir+"canons.pdf", format="pdf", bbox_inches='tight',dpi=600)
    plt.close()

def Varsigma(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [s/2 for s in range(-5,5)]

    print(sig_space)

    Observed_L = []
    Observed_H = []
    age_uncert = []
    Observed = []
    Canons = []
    etas = []

    for s in sig_space:
        Noise=s
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    # Plot:
    xlab = r"Observation Noise ($\varsigma$)"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.7)

def GridZ(dir):
    global a_H, G_0, Z, sigma, C, obsRisk, O, B, d, Noise, careRisk, obsRisk
    sig_space = [2]+[s*10+1 for s in range(1,10)]

    print(sig_space)

    Observed_L = []
    Observed_H = []
    age_uncert = []
    Observed = []
    Canons = []
    etas = []

    for z in sig_space:
        Z=z
        init()
        barcode=Barcode()
        print(barcode)


        if not os.path.exists(dir+barcode):
            print("Finding DP solution ~")
            SimpRun(dir)
        
        #Reads and processes data for plotting if it already exists
        x[:,0] = np.genfromtxt(dir+barcode+"x.csv", delimiter=",")
        psi[:,0] = np.genfromtxt(dir+barcode+"psi.csv", delimiter=",")
        age_H_start = np.genfromtxt(dir+barcode+"age_H_start.csv", delimiter=",")
        age_L_start = np.genfromtxt(dir+barcode+"age_L_start.csv", delimiter=",")
        Vz = np.genfromtxt(dir+barcode+"Vz.csv", delimiter=",")
        f = open(dir+barcode+"eta.csv")
        for line in f.readlines():
            etas.append(float(line))

        # Calculate total amount of observations
        Total_Observed_L = (age_L_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Total_Observed_H = (age_H_start[1:,:]*psi[:,0]/(Z-1)).sum(axis=1).sum()/len(age_L_start[1:,0])
        Observed_L.append(Total_Observed_L)
        Observed_H.append(Total_Observed_H)
        Observed.append(Total_Observed_H+Total_Observed_L)
        canon = 0
        for j in range(J):
            canon += age_H_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
            canon += age_L_start[:,j]*(Vz[j,psi[j,0]] -Vz[j,0])
        Canons.append(canon)
    
    # Plot:
    xlab = "Strategy grid fineness (Z)"
    usage_plots(dir, age_H_start, sig_space, Observed_L, Observed_H, xlab)
    canon_plots(dir, 10, Canons, sig_space, etas, xlab, 0.7)

## Run ##
def main(argv):
    global filename, d, B, J, a_H, a_L, O, sigma, sds, N
    global a_f, Z, C, O, careRisk, obsRisk, Noise, fq
    global G, G_0, I, diffval, diffstrat, difffit, tol
    global L, H, start, mid, end, HD , q, q_z, l, l_z, rho, a
    global b, P, PKJZ, psi, x, Pij, V, R, Vz, HC, HM

    ### Experiments ###
    # version = "../results_no_eta/"
    version = "../results/"

    # Reset()
    # Experiment = "Noise_nofeedback/"
    # print(Experiment)
    # NoFeedback_varsigma(version+Experiment)

    # Reset()
    # Experiment = "Noise2/"
    # print(Experiment)
    # Varsigma(version+Experiment)

    # Reset()
    # Experiment = "GridZ/"
    # print(Experiment)
    # GridZ(version+Experiment)

    # Reset()
    # Experiment= "NoFeedback_mu/"
    # print(Experiment)
    # NoFeedback_mu(version+Experiment)
    
    # Reset()
    # C=0.0
    # Experiment= "Feedback_mu/"
    # print(Experiment)
    # ORisk(version+Experiment)

    # Reset()
    # C=0.0
    # Noise=-1
    # Experiment= "NoFeedback_mu_lownnoise/"
    # print(Experiment)
    # NoFeedback_mu(version+Experiment)

    # Reset()
    # Experiment= "NoFeedback_Cc/"
    # print(Experiment)
    # NoFeedback_Cc(version+Experiment)

    # Reset()
    # Experiment= "Feedback_Cc/"
    # print(Experiment)
    # Carecost(version+Experiment)



    # Reset()
    # Noise=-1
    # Experiment= "Feedback_Co_lownnoise/"
    # print(Experiment)
    # ORisk(version+Experiment)


    # Reset()
    # Experiment= "NoFeedback_mean/"
    # print(Experiment)
    # NoFeedback_mean(version+Experiment)


    Reset()
    Experiment = "H_scarcity/"
    print(Experiment)
    H_Scarcity(dir=version+Experiment)

    # Reset()
    # Noise = 10.
    # Experiment = "HighNoise/"
    # print(Experiment)
    # NoisyQplots(dir=version+Experiment)

    # Reset()
    # Experiment = "Baseline/"
    # print(Experiment)
    # full_explore(dir=version+Experiment)

    # Reset()
    # C=0.0025
    # Experiment = "Mean2/"
    # print(Experiment)
    # Mean(dir=version+Experiment)
    
    # Reset()
    # C=0.01
    # Experiment = "Mean0/"
    # print(Experiment)
    # Mean(dir=version+Experiment)

    # Reset()
    # C=0.
    # obsRisk=0.00025
    # Experiment = "MortORonly/"
    # print(Experiment)
    # Mortality(dir=version+Experiment)
    
    # Reset()
    # C=0.0025
    # obsRisk=0.
    # Experiment = "Mort/"
    # print(Experiment)
    # Mortality(dir=version+Experiment)

    # Reset()
    # C=0.
    # obsRisk=0.00025
    # fq = 3.5
    # Experiment = "HighF_and_Cmu/"
    # print(Experiment)
    # Mortality(dir=version+Experiment)
    
    # Reset()
    # C=0.0025
    # obsRisk=0.
    # fq = 2
    # Experiment = "mort_Co_fvH/"
    # print(Experiment)
    # Mortality(dir=version+Experiment)

    # Reset()
    # C=0.0025
    # obsRisk=0.0
    # Experiment = "FemQual/"
    # print(Experiment)
    # FemQual(dir=version+Experiment)

    # Reset()
    # C=0.0
    # obsRisk=0.00025
    # Experiment = "FemQualrisk/"
    # print(Experiment)
    # FemQual(dir=version+Experiment)

    # Reset()
    # C=0.
    # obsRisk=0.0005
    # Experiment = "HighCmu/"
    # print(Experiment)
    # Mortality(dir=version+Experiment)
    
    # Reset()
    # C=0.005
    # obsRisk=0.
    # Experiment = "HighCo/"
    # print(Experiment)
    # Mortality(dir=version+Experiment)

    # Not currently working due to use of faulty unicodeit package:
    # Reset()
    # d=0.1
    # Experiment = "Lowmu_costs/"
    # print(Experiment)
    # BothCosts(dir=version+Experiment)

    # Not currently working due to use of faulty unicodeit package:
    # Reset()
    # d=0.4
    # Experiment = "Highmu_costs/"
    # print(Experiment)
    # BothCosts(dir=version+Experiment)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)