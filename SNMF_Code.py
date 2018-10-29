# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:54:32 2018

@author: amiglan, sasthan1 & sroy66 (Anurag, Siddhartha & Suman)
"""
import numpy as np
from numpy import linalg as LA
#V shall be the DTM for this.
#P Shall be a random initialization

"""
while t < Max Iter do
Rt = (V P ) / (P* P.T *P ), % entrywise division
Pnew = Pt *(Rt )α, % entrywise exponentiation
Enew = V − Pnew(Pnew)T 2
F
if Enew ≥ Et then
Update P by (17): Pnew = Pt  (Rt )1/3
Enew = V − Pnew(Pnew)T 2
F
end if
Pt+1 = Pnew
Et+1 = Enew
t = t + 1
end while



"""


def SymNMF(V,K=9,alpha=0.99,maxIter=1000):
    alpha = 0.99
    epsilon = 0.0000000005
    P = np.random.rand(V.shape[0],K)#we need to initialize this one
    P=P/1000.0
    P = np.nan_to_num(P)
    P = P+epsilon
    error = LA.norm(V-np.dot(P,P.T))
    
    for iter in range(1,maxIter):
        #print(str(iter))
        VP = np.dot(V,P)
        VP = np.nan_to_num(VP)
        
        #print("line 40 : Iteration " + str(iter))
        PPt = np.dot(P,P.T)
        PPt = np.nan_to_num(PPt)
        #Todo: Why
        PPt = PPt + epsilon
        #print("line 42 : Iteration " + str(iter))
        PPtP = np.dot(PPt,P)
        PPtP = np.nan_to_num(PPtP)
        PPtP = PPtP + epsilon
        #print("line 44 : Iteration " + str(iter))
        R = VP/PPtP
        R = np.nan_to_num(R)
        R = R + epsilon
        #print("line 46 : Iteration " + str(iter))
        #Ralpha = np.exp(R,alpha)
        Ralpha = np.power(R,alpha)
        Ralpha = np.nan_to_num(Ralpha)
        Ralpha = Ralpha + epsilon
        #print("line 49 : Iteration " + str(iter))
        Pnew = P*Ralpha
        Pnew = np.nan_to_num(Pnew)
        Pnew = Pnew + epsilon
        #print("line 51 : Iteration " + str(iter))
        errornew = LA.norm(V-np.dot(Pnew,Pnew.T))
        #print("line 53 : Iteration " + str(iter))
        if(errornew >= error): #Confusion
            Ralpha = np.power(R,1.0/3)
            Ralpha = np.nan_to_num(Ralpha)
            Ralpha = Ralpha + epsilon
            #print("line 56 : Iteration " + str(iter))
            #Ralpha = np.exp(R,1.0/3)
            Pnew = P*Ralpha
            Pnew = np.nan_to_num(Pnew)
            Pnew = Pnew + epsilon
            #print("line 59 : Iteration " + str(iter))
            errornew = LA.norm(V-np.dot(Pnew,Pnew.T))
            #print("line 61 : Iteration " + str(iter))
        if(np.isnan(errornew)):
            print('nan error')
            break
        
        P = Pnew
        P= np.nan_to_num(P)
        P = P + epsilon
        if((error - errornew) < epsilon*100.0):
            print("Converge successfully! "  + str(iter))
            return(P)
            break
        error = errornew
        if(iter%10 == 0):
            print(error)
    return(P)
    