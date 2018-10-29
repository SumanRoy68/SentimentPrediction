# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:24:31 2018

@author: m37
"""

import numpy as np
from numpy import linalg as LA
import pandas as pd
import os
from numpy.linalg import inv


base_dir = "C:/Users/m37/Topic Modeling/U_Matrix"


def readFeedback(filepath):
    if os.path.isfile(filepath):
        filename, file_extension = os.path.splitext(filepath)
        if file_extension == ".csv":
            return pd.read_csv(filepath)
        elif file_extension == ".xlsx": 
            return pd.read_excel(filepath)
    else:
        raise Exception('Invalid feedback path!. Hint: Use forward slashes e.g. use C:/Users instead C:\Users')

def split(ln, frac=0.7):
    msk = np.random.rand(ln) < frac
    #train = df[msk]
    #test =  df[~msk]
    return msk

#TriNMF(X_dtm.toarray().T, U_mat9, 9, df.Sentiment.values, 1000,0.7)
def TriNMF(X,U,K,W0, maxIter, f, fn):
    W0=W0.reshape(W0.shape[0],1)
    W=W0
    mu=0.8
    frac=f
    name=fn+".xlsx" 
    #mask = split(X.shape[0])
    W_lbl = W0[:int(X.shape[1]*frac)]    
    W_Ulbl = W0[int(X.shape[1]*frac):] 
    for i in range(maxIter):
        if i%50==0:
            print(i)
        # Update S
        out1 = inv(np.dot(U.T,U))
        out2 = np.dot(U.T,np.dot(X,W))
        out3 = 1.0/np.dot(W.T,W)
        S=np.dot(out1,out2*out3 )
        S=S.reshape(S.shape[0],1)

        # Update W
        alpha = np.dot(X.T,np.dot(U,S))
        neta = np.dot(S.T,np.dot(U.T,np.dot(U,S)))
        # Updated Labeled Feedback
        W_lbl = alpha[:int(X.shape[1]*frac)] + mu*W0[:int(X.shape[1]*frac)]
        W_lbl /= (neta + mu)
            
        # Update Unlabeled Feedback
        W_Ulbl = alpha[int(X.shape[1]*frac):]/neta
        W= np.concatenate((W_lbl,W_Ulbl),axis=0)
        W=W.reshape(W.shape[0],1)

    
    df = pd.DataFrame(W)
    df.to_excel(base_dir+name,index=False)
    
def Sindhwani_TriNMF(X,U,K,W0, maxIter, f, fn):
    W0=W0.reshape(W0.shape[0],1)
    W=W0
    mu=0.8
    frac=f
    U=np.nan_to_num(U)
    name="Sindh_"+fn+".xlsx" 
    doc_size = X.shape[1]
    train= doc_size*frac
    W_lbl = W0[:int(train)]    
    W_Ulbl = W0[int(train):]    
    for i in range(maxIter):
        if i%50==0:
            print( i, W.shape, np.sum(W))
        # Update S
        out1 = inv(np.dot(U.T,U))
        out2 = np.dot(U.T,np.dot(X,W))
        out3 = 1.0/np.dot(W.T,W)
        S=np.dot(out1,out2*out3) 
        S=S.reshape(S.shape[0],1)
        S=np.nan_to_num(S)
        # Update W
        alpha = np.dot(X.T,np.dot(U,S))
        neta = np.dot(S.T,np.dot(U.T,np.dot(U,S)))
        # Updated Labeled Feedback
        W_lbl = alpha[:int(train)] + mu*W0[:int(train)]
        W_lbl /= (neta + mu)
            
        # Update Unlabeled Feedback
        W_Ulbl = alpha[int(train):]/neta
        W= np.concatenate((W_lbl,W_Ulbl),axis=0)
        W=W.reshape(W.shape[0],1)
        W= np.nan_to_num(W)
        # Update U
        U = U*(np.dot(X,np.dot(W,S.T))/np.nan_to_num(np.dot(U,np.dot(S,np.dot(W.T,np.dot(W,S.T))))))
        U = np.nan_to_num(U)
     
    df = pd.DataFrame(W)
    df.to_excel(base_dir+name,index=False)
        
TriNMF(X_2000.T, U_2000, 9, df.Sentiment.values, 1000,0.7,"vocab_2000")
TriNMF(X_3000.T, U_3000, 9, df.Sentiment.values, 1000,0.7,"vocab_3000")
TriNMF(X_max.T, U_max, 9, df.Sentiment.values, 1000,0.7,"vocab_max")