#!/usr/bin/env python

#Recommender system based on Jaccard

import implicit
import pandas as pd
import scipy.sparse as sparse
import pickle
import random
import numpy as np
from sklearn import metrics
import itertools
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

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

def get_similar_users(similarity_vector,ind2item):
    """ Returns a list of recommendations for a given fic based on similarity.
    Input: 
    similarity_vector -> similarity vector between this fic and all others
    ind2item -> dictionary relating indices to fics
    
    Output:
    list of recommendations and similarities
    """
    similarity_scores = list(enumerate(similarity_vector[0]))
    similarity_scores_filtered = [x for x in similarity_scores if x[1] > args.thr]
    similarity_scores_filtered = sorted(similarity_scores_filtered,\
                                key=lambda x: x[1], reverse=True)
    if len(similarity_scores_filtered) < args.numR:
        K = len(similarity_scores_filtered)
    else:
        K = args.numR
    if K != 0:
        similarity_scores_filtered = similarity_scores_filtered[1:K+1]
        recom = [ind2item[i[0]] for i in similarity_scores_filtered]
        sim = [i[1] for i in similarity_scores_filtered]
    else:
        recom = []
        sim = []
    return recom,sim

parser = argparse.ArgumentParser(description="User similarity recommender")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("-k","--number_recommendations",dest="numR",action="store",\
    type=int,default=15,help="Number of recommendations")
parser.add_argument("--minSimilarity",dest="thr",action="store",type=float,\
    default=0.1,help="Minimum similarity score for a recommendation to be\
    considered")
args = parser.parse_args()


#Loads table into pandas and creates sparse matrix
print("Load datasets into memory:")
df_train = pd.read_csv(args.user2item,sep="\t",dtype={"user":str,"item":str,"rating":float})
print("Create_indices:")
ind2item, item2ind, user2ind, ind2user = create_indices(df_train)
print("Create sparse matrices:")
user_item_train = create_sparse_matrix(df_train,user2ind,item2ind)

with open(args.outfileName,"w") as outfile:
    for a in tqdm(range(user_item_train.shape[0])):
        user_vector = user_item_train[a]
        similarity_vector = cosine_similarity(user_vector,user_item_train)
        similarity_scores = list(enumerate(similarity_vector[0]))
        similarity_scores_filtered = [x for x in similarity_scores if x[1] > 0.01]
        similarity_scores_filtered = sorted(similarity_scores_filtered,\
                                        key=lambda x: x[1], reverse=True)
        close_users = [x[0] for x in similarity_scores_filtered[1:101]]
        similarity_users = [x[1] for x in similarity_scores_filtered[1:101]]
        items_scores = user_item_train[close_users]
        t = items_scores.toarray()
        users_mat2sim = (t.T*similarity_users).T
        item_similarity_vector = users_mat2sim.sum(axis=0)
        items_scores = list(enumerate(item_similarity_vector))
        read = np.where(user_item_train[a].toarray().reshape(-1) != 0)[0]
        item_scores_filtered = [x for x in items_scores if x[0] not in read]
        item_scores_filtered = sorted(items_scores,\
                                    key=lambda x: x[1], reverse=True)
        recom = [ind2item[x[0]] for x in item_scores_filtered[:args.numR]]
        print(ind2user[a]+"\t"+";".join(recom),file=outfile)
