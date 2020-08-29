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
import itertools
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
    for idx,item in enumerate(df["item"].unique().tolist()): 
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
    I = df["item"].apply(map_ids, args=[item2ind]).values
    V = np.ones(I.shape[0]) 
    sparse_user_item = sparse.coo_matrix((V, (U, I)), dtype=np.float64) 
    sparse_user_item = sparse_user_item.tocsr() 
    return sparse_user_item

def sample_hyperparameters_ALS():

    while True:
        yield {
            "factors": np.random.randint(400,750),
            "regularization":np.random.exponential(0.01),
            "iterations":np.random.randint(5,100),
            "alpha":np.random.randint(40,100)
        }

def sample_hyperparameters_others():
    """ Randomly provides hyperparameters for exploration"""
    while True:
        yield {
            "factors": np.random.randint(400,750),
            "regularization":np.random.exponential(0.01),
            "iterations":np.random.randint(5,100),
            "learning_rate":np.random.exponential(0.01),
            "alpha":np.random.randint(40,100),
            "model":np.random.choice(["BPR","LMF"])
        }

def print_recommendations(model,user_item_train,K,outfileName,hyper):
    """ Prints recommendations for users into a file.
    Input:
    model -> Model obtained from implicit
    user_item_train -> user item table
    K -> Number of predictions to print
    """
    with open(outfileName,"w") as outfileRecom:
        if len(hyper) != 0:
            for h in hyper:
                print(h,hyper[h],file=outfileRecom)
        recommendations_train = model.recommend_all(user_item_train, K,filter_already_liked_items=False)
        for u in range(user_item_train.shape[0]):
            print(map_ids(u,ind2user)+"\t"+";".join([str(map_ids(x,ind2item)) for x in recommendations_train[u]]),file=outfileRecom)

parser = argparse.ArgumentParser(description="ALS recommender")
parser.add_argument("-i","--input",dest="inFile",action="store",default=None,help="Provides the training file")
parser.add_argument("-o","--output",dest="outFile",action="store",default="recom.txt",help="Prints all recommendations into a file")
parser.add_argument("-e","--explore",dest="explore",action="store_true",help="Explores hyperparameters")
parser.add_argument("-a","--alpha",dest="alpha",action="store",type=int,default = 40, help="Alpha hyperparameter")
parser.add_argument("-f","--factors",dest="factors",action="store",type=int,default=100, help="Number of factors hyperparameter")
parser.add_argument("-r","--regularization",dest="regularization",action="store",type=float,default=0.01, help="Regularization hyperparameter")
parser.add_argument("-t","--iterations",dest="iterations",action="store",type=int,default=10,help="Number of iterations hyperparameter")
parser.add_argument("-l","--learning_rate",dest="learning_rate",action="store",type=float,default=0.01,help="Learning rate")
parser.add_argument("-k","--num_recommendations",dest="k",action="store",type=int,default=50,help="Number of recommendations per user")
parser.add_argument("-m","--model",dest="model",action="store",default="ALS",help="Model to use - choose between ALS, LMF, BPR")
parser.add_argument("-n","--num_explorations",dest="numExp",action="store",type=int,default=30,help="Number of times the hyperparameter space will be explored")
args = parser.parse_args()

#Loads table into pandas and creates sparse matrix
print("Load datasets into memory:")
df_train = pd.read_csv(args.inFile,sep="\t")
print("Create_indices:")
ind2item, item2ind, user2ind, ind2user = create_indices(df_train)
print("Create sparse matrices:")
user_item_train = create_sparse_matrix(df_train,user2ind,item2ind)
item_user_train = user_item_train.T

K=args.k

if args.model == "ALS":
    if not args.explore:
        alpha = args.alpha
        model = implicit.als.AlternatingLeastSquares(factors=args.factors,regularization=args.regularization,iterations=args.iterations)
        model.fit(item_user_train*alpha)
        print_recommendations(model,user_item_train,K,args.outFile,[])
    else:
        num = 1
        for hyper in itertools.islice(sample_hyperparameters_ALS(), args.numExp):
            alpha = hyper.pop("alpha")
            model = implicit.als.AlternatingLeastSquares(**hyper)
            model.fit(item_user_train*alpha)
            hyper["alpha"] = alpha
            print_recommendations(model,user_item_train,K,"expl.ALS."+str(num)+".txt",hyper)
            num += 1
else:
    if args.explore:
        num = 1
        for hyper in itertools.islice(sample_hyperparameters_others(), args.numExp):
            m = hyper.pop("model")
            alpha = hyper.pop("alpha")
            if m == "LMF":
                model = implicit.lmf.LogisticMatrixFactorization(**hyper)
            else:
                model = implicit.bpr.BayesianPersonalizedRanking(**hyper)
            model.fit(item_user_train*alpha)
            hyper["model"] = m
            hyper["alpha"] = alpha
            print_recommendations(model,user_item_train,K,"expl.others."+str(num)+".txt",hyper)
            num += 1
    else:
        if args.model == "LMF":
            model = implicit.lmf.LogisticMatrixFactorization(factors=args.factors, learning_rate=args.learning_rate, regularization=args.regularization,iterations=args.iterations)
            model.fit(item_user_train*alpha)
            print_recommendations(model,user_item_train,K,args.outFile,[])
        else:
            model = implicit.bpr.BayesianPersonalizedRanking(factors=args.factors, learning_rate=args.learning_rate, regularization=args.regularization,iterations=args.iterations)
            model.fit(item_user_train*alpha)
            print_recommendations(model,user_item_train,K,args.outFile,[])
