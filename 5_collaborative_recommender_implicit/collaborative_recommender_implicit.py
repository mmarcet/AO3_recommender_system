#!/user/bin/env python

#Build ALS_recommender based on implicit library

#Create matrices

import implicit
import pandas as pd
import scipy.sparse as sparse
import random
import numpy as np
import itertools
import argparse

import sys
sys.path.append("../")
sys.path.append(".")
import common_functions as CF

def sample_hyperparameters():
    """ Randomly provides hyperparameters for exploration"""
    while True:
        yield {
            "factors": np.random.randint(50,1000),
            "regularization":np.random.exponential(0.01),
            "iterations":np.random.randint(5,100),
            "learning_rate":np.random.exponential(0.01),
            "alpha":np.random.randint(10,100),
            "model":np.random.choice(["ALS","BPR","LMF"])
        }

def print_recommendations(model,user_item,K,outfileName,hyper):
    """ Prints recommendations for users into a file.
    Input:
    model -> Model obtained from implicit
    user_item -> user item table
    K -> Number of predictions to print
    """
    with open(outfileName,"w") as outfileRecom:
        if len(hyper) != 0:
            print("#########################################",file=outfile)
            for h in hyper:
                print(h,hyper[h],file=outfileRecom)
        recommendations = model.recommend_all(user_item, K,\
                            filter_already_liked_items=False)
            print("#########################################",file=outfile)
        for u in range(user_item.shape[0]):
            print(CF.map_ids(u,ind2user)+"\t"+";".join([str(CF.map_ids(x,ind2item))\
                for x in recommendations[u]]),file=outfileRecom)

parser = argparse.ArgumentParser(description="ALS recommender")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("-e","--explore",dest="explore",action="store_true",\
    help="Explores hyperparameters")
parser.add_argument("-a","--alpha",dest="alpha",action="store",type=int,\
    default = 40, help="Alpha hyperparameter")
parser.add_argument("-f","--factors",dest="factors",action="store",type=int,\
    default=100, help="Number of factors hyperparameter")
parser.add_argument("-r","--regularization",dest="regularization",action="store",\
    type=float,default=0.01, help="Regularization hyperparameter")
parser.add_argument("-i","--iterations",dest="iterations",action="store",type=int,\
    default=10,help="Number of iterations hyperparameter")
parser.add_argument("-l","--learning_rate",dest="learning_rate",action="store",\
    type=float,default=0.01,help="Learning rate")
parser.add_argument("-k","--num_recommendations",dest="k",action="store",\
    type=int,default=50,help="Number of recommendations per user")
parser.add_argument("-m","--model",dest="model",action="store",default="ALS",\
    help="Model to use - choose between ALS, LMF, BPR")
parser.add_argument("-n","--num_explorations",dest="numExp",action="store",\
    type=int,default=30,help="Number of times the hyperparameter space will\
    be explored")
args = parser.parse_args()

#Loads table into pandas and creates sparse matrix
print("Building sparse matrices")
df_ui = pd.read_csv(args.user2item,sep="\t",\
            dtype={"user":str,"item":str,"rating":float})
ind2item, item2ind, user2ind, ind2user = CF.create_indices(df_ui)
user_item = CF.create_sparse_matrix(df_ui,user2ind,item2ind)
item_user = user_item.T

K=args.k

if not args.explore:
    #Provide recommendations with hyperparameters given by the user
    alpha = args.alpha
    hyper = {"factors":args.factors,\
                "regularization":args.regularization,\
                "iterations":args.iterations}

    if args.model == "ALS":            
        model = implicit.als.AlternatingLeastSquares(**hyper)
    elif args.model == "LMF":
        hyper["learning_rate"] = args.learning_rate
        model = implicit.lmf.LogisticMatrixFactorization(**hyper)
    elif args.model == "BPR":
        hyper["learning_rate"] = args.learning_rate
        model = implicit.bpr.BayesianPersonalizedRanking(**hyper)
    model.fit(item_user*alpha)
    print_recommendations(model,user_item,K,args.outfileName,[])
else:
    #Explore the hyperparameter space
    num = 1
    for hyper in itertools.islice(sample_hyperparameters_ALS(), \
                                    args.numExp):
        alpha = hyper.pop("alpha")
        model = hyper.pop("model")
        if model == "ALS":
            learning_rate = hyper.pop("learning_rate")
            model = implicit.als.AlternatingLeastSquares(**hyper)
        elif model == "LMF":
            model = implicit.lmf.LogisticMatrixFactorization(**hyper)
        elif model == "BPR":
            model = implicit.bpr.BayesianPersonalizedRanking(**hyper)
        model.fit(item_user*alpha)
        hyper["alpha"] = alpha
        hyper["model"] = model
        print_recommendations(model,user_item,K,\
                    args.outfileName+str(num)+".txt",hyper)
        num += 1

