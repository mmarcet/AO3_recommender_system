#!/user/bin/env python

#Build ALS_recommender based on implicit library

#Create matrices

import implicit
import pandas as pd
import scipy.sparse as sparse
import pickle
import random
import numpy as np
from sklearn import metrics
import argparse

def map_ids(row,mapper):
    """Returns the value of a dicctionary
    input 
    row -> value you're searching for
    mapper -> dictionary
    Output: value of the dictionary
    """
    return mapper[row]

def create_indices(df):
    """ Creates a list of dictionaries that will map users to indices, 
    items to indices and the other way around
    Input: df -> dataframe containing the user\titem\trating information
    Output: Returns four dictionaries: items to indices, users to indices,
    and indices to users and indices to items
    """
    ind2item, item2ind = {}, {} 
    for idx,item in enumerate(df["fic"].unique().tolist()): 
        ind2item[idx] = item 
        item2ind[item] = idx 
    user2ind, ind2user = {}, {} 
    for idx,uid in enumerate(df["user"].unique().tolist()): 
        user2ind[uid] = idx 
        ind2user[idx] = uid 
    return ind2item, item2ind, user2ind, ind2user

def create_sparse_matrix(df,user2ind,item2ind): 
    """ Builds a csr matrix.
    Input:
    df -> dataframe that contains user and item information
    user2ind -> dictionary that maps users to the indices
    item2ind -> dictionary that maps items to the indices
    Output -> Sparse matrix of users to items where all values
    of interaction between user and item are 1
    """
    U = df["user"].apply(map_ids, args=[user2ind]).values 
    I = df["fic"].apply(map_ids, args=[item2ind]).values
    V = np.ones(I.shape[0]) 
    sparse_user_item = sparse.coo_matrix((V, (U, I)), dtype=np.float64) 
    sparse_user_item = sparse_user_item.tocsr() 
    return sparse_user_item

parser = argparse.ArgumentParser(description="ALS recommender")
parser.add_argument("--inFile",dest="inFile",action="store",default=None,help="Provides the training file")
parser.add_argument("--outFile",dest="outFile",action="store",required=True,help="Prints all recommendations into a file")
parser.add_argument("-a",dest="alpha",action="store",type=int,default = 40, help="Alpha hyperparameter")
parser.add_argument("-f",dest="factors",action="store",type=int,default=100, help="Number of factors hyperparameter")
parser.add_argument("-r",dest="regularization",action="store",type=float,default=0.01, help="Regularization hyperparameter")
parser.add_argument("-i",dest="iterations",action="store",type=int,default=10,help="Number of iterations hyperparameter")
parser.add_argument("-l",dest="learning_rate",action="store",type=float,default=0.01,help="Learning rate")
parser.add_argument("-k",dest="k",action="store",type=int,default=50,help="Number of recommendations per user")
parser.add_argument("-m",dest="model",action="store",default="ALS",help="Model to use - choose between ALS, LMF, BPR")
args = parser.parse_args()

#Loads table into pandas and creates sparse matrix
print("Load datasets into memory:")
df_train = pd.read_csv(args.inFile,sep="\t",names=["user","fic","rating"])
print("Create_indices:")
ind2item, item2ind, user2ind, ind2user = create_indices(df_train)
print("Create sparse matrices:")
user_item_train = create_sparse_matrix(df_train,user2ind,item2ind)

#Build model
alpha = args.alpha
if args.model == "ALS":
    model = implicit.als.AlternatingLeastSquares(factors=args.factors,regularization=args.regularization,iterations=args.iterations)
    item_user_train = user_item_train.T
    model.fit(item_user_train*alpha)
elif args.model == "LMF":
    model = implicit.lmf.LogisticMatrixFactorization(factors=args.factors, learning_rate=args.learning_rate, regularization=args.regularization,iterations=args.iterations)
    item_user_train = user_item_train.T
    model.fit(item_user_train*alpha)
elif args.model == "BPR":
    model = implicit.bpr.BayesianPersonalizedRanking(factors=args.factors, learning_rate=args.learning_rate, regularization=args.regularization,iterations=args.iterations)
    item_user_train = user_item_train.T
    model.fit(item_user_train*alpha)

K=args.k

#Prints recommendations for all users
with open(args.outFile,"w") as outfileRecom:
    recommendations_train = model.recommend_all(user_item_train, K,filter_already_liked_items=False)
    for u in range(user_item_train.shape[0]):
        print(map_ids(u,ind2user)+"\t"+";".join([str(map_ids(x,ind2item)) for x in recommendations_train[u]]),file=outfileRecom)
    


