#!/usr/bin/env python

"""
  AO3_recommender - a recommendation system for fanfiction
  Copyright (C) 2020 - Marina Marcet-Houben
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#Recommender system based cosine similarity between users

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

try:
    import sys
    sys.path.append(".")
    sys.path.append("../")
    import common_functions as CF
except:
    exit("The common_functions.py file needs to be in this folder or in the\
parent folder for it to be imported")


def get_similar_users(user_vector,user_item,sim_matrix,user_ind, sim_thr):
    """ Returns the list of users that are closest to the current user
    and their similarity.
    Input: 
    user_vector -> vector of items seen by the user
    user_item -> user to item sparse matrix
    sim_matrix -> pre-calculated similarity matrix
    user_ind -> index of the current user
    sim_thr -> Similarity threshold. users below this threshold will
        not be used
    
    Output:
    close_users -> a list of indices of the closest users
    similarity_users -> list of similarities for the selected users
    """
    if type(sim_matrix) == str:
        similarity_vector = 1-pairwise_distances(user_vector,user_item,\
                                metric="cosine",n_jobs=args.threads)
        similarity_vector = similarity_vector[0]
    else:
        similarity_vector = sim_matrix[user_ind]
    similarity_scores = list(enumerate(similarity_vector))
    similarity_scores_filtered = [x for x in similarity_scores if x[1] > sim_thr]
    similarity_scores_filtered = sorted(similarity_scores_filtered,\
                                        key=lambda x: x[1], reverse=True)
    close_users = [x[0] for x in similarity_scores_filtered[1:args.numU+1]]
    similarity_users = [x[1] for x in similarity_scores_filtered[1:args.numU+1]]

    return close_users, similarity_users

def get_recommendations(close_users,similarity_users,user_item,read_items):
    """ Obtains the recommendations for a user.
    Input:
    close_users -> list of users with the highest similarity to the user 
    of interest
    similarity_users -> Similarity between the close users and the user
    of interest
    user_item -> user to item matrix
    read_items -> list of items that have already been read by the user
    Output:
    List of recommendations
    """
    items_scores = user_item[close_users]
    t = items_scores.toarray()
    users_mat2sim = (t.T*similarity_users).T
    item_similarity_vector = users_mat2sim.sum(axis=0)
    items_scores = list(enumerate(item_similarity_vector))
    item_scores_filtered = [x for x in items_scores if x[0] not in read_items]
    item_scores_filtered = sorted(items_scores,\
                            key=lambda x: x[1], reverse=True)
    recom = [ind2item[x[0]] for x in item_scores_filtered[:args.numR]]
    return recom

parser = argparse.ArgumentParser(description="User similarity recommender")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("-k",dest="numR",action="store",\
    type=int,default=15,help="Number of recommendations")
parser.add_argument("--minSimilarity",dest="thr",action="store",type=float,\
    default=0.01,help="Minimum similarity to be considered")
parser.add_argument("--calcWholeMatrix",dest="calc_whole",action="store_true",\
    help="Calculates the whole similarity matrix at once instead of going\
    user by user, it's faster but needs more RAM.")
parser.add_argument("--threads",dest="threads",action="store",type=int,\
    default=6,help="Threads to use when calculating whole matrix.")
parser.add_argument("--num_users",dest="numU",action="store",type=int,\
    default=50,help="Number of most similar users to consider when making\
    recommendations")
args = parser.parse_args()

#Loads table into pandas and creates sparse matrix
print("Building sparse matrices")
df_ui = pd.read_csv(args.user2item,sep="\t",\
            dtype={"user":str,"item":str,"rating":float})
ind2item, item2ind, user2ind, ind2user = CF.create_indices(df_ui)
user_item = CF.create_sparse_matrix(df_ui,user2ind,item2ind)

if args.calc_whole:
    #Calculates similarity matrix of users against users
    sim_matrix = 1-pairwise_distances(user_item,\
                                    metric="cosine",n_jobs=args.threads)
else:
    sim_matrix = "None"

with open(args.outfileName,"w") as outfile:
    for a in tqdm(range(user_item.shape[0])):
        user_vector = user_item[a]
        read_items = np.where(user_item[a].toarray().reshape(-1) != 0)[0]     
        close_users, similarity_users = get_similar_users(user_vector,\
                                        user_item,sim_matrix,a,args.thr)
        recom = get_recommendations(close_users,similarity_users,\
                                    user_item, read_items)
        if len(recom) != 0:
            print(ind2user[a]+"\t"+";".join(recom),file=outfile)


