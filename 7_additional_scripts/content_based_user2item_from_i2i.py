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

#Content based recommender system - user 2 item via item 2 item recommendations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm

try:
    import sys
    sys.path.append(".")
    sys.path.append("../")
    import common_functions as CF
except:
    exit("The common_functions.py file needs to be in this folder or in the\
parent folder for it to be imported")


def get_recommendation(similarity_vector,ind2item):
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

def get_similarities(word_matrix,ind):
    """ Calculated the similarity between a fic and all the others based 
    on the word matrix calculated
    Input:
    word_matrix -> matrix that relates fics to tokens
    ind -> index of the fic of interest
    
    Output:
    Similarity vector
    """
    if args.word_method == "tfidf":
        similarity_vector = linear_kernel(word_matrix[ind],word_matrix)
    else:
        similarity_vector = cosine_similarity(word_matrix[ind],word_matrix)
    return similarity_vector

def recommend_to_single_item(ficName,item2ind,ind2item,word_matrix):
    """ Returns the recommendation for a single fic of interest.
    Input:
    ficName -> id of the fic of interest provided by the user
    item2ind -> dictionary that relates fic names to their index in the 
                dataframe and matrix
    ind2item -> dictionary that relates indexes to their fic names
    word_matrix -> matrix that relates fics to tokens
    Output:
    recom -> list of recommended items 
    similarity_scores -> list of similarity scores for recommended items
    """
    fic_index = item2ind[ficName]
    similarity_vector = get_similarities(word_matrix, fic_index)
    recom,similarity_scores = get_recommendation(similarity_vector,ind2item)
    return recom,similarity_scores

def recommend_all_users(df,word_matrix,user2items,outfileName,K):
    """ Calculates recommendations for all the users provided.
    Input:
    df -> Datafram where the list of fics can be found
    word_matrix -> matrix that relates fics to tokens
    user2items -> dictionary that lists all fics for a given user
    outfileName -> file where all recommendations will be printed
    """
    list_fics = set([])
    for user, fics in user2items.items():
        for f in fics:
            list_fics.add(f)
    all_recommendations = {}
    for ficName in tqdm(list_fics):
        recom,similarity_scores = recommend_to_single_item(ficName,\
                                    item2ind,ind2item,word_matrix)
        all_recommendations[ficName] = [recom,similarity_scores]
    with open(outfileName,"w") as outfile:
        for user, list_fics in user2items.items():
            recoms = {}
            for ficName in list_fics:
                recom = all_recommendations[ficName]
                r = {}
                for a in range(len(recom[0])):
                    if recom[0][a] not in list_fics:
                        r[recom[0][a]] = recom[1][a]
                recoms[ficName] = r
            recoms = join_recommendations(recoms, K)
            print(user+"\t"+";".join(recoms),file=outfile)

def parse_date(x): 
    """ Parses a date to a datetime format.
    Input: string containing the date
    Output: datetime object
    """
    if "-" in x: 
        f = lambda x: pd.datetime.strptime(x, "%Y-%m-%d") 
    else: 
        f = lambda x: pd.datetime.strptime(x, "%d %b %Y") 
    return f(x)

def join_recommendations(recommendations,K):
    counter = {}
    for parent_fic,fics in recommendations.items():
        for f, s in fics.items():
            if f not in counter:
                counter[f] = 0
            counter[f] += s
    all_fics = list(counter.keys())
    all_fics = sorted(all_fics,key=lambda x: counter[x],reverse=True)
    return all_fics[:K]

def recommend_to_single_user(list_fics,item2ind,ind2item,word_matrix,K):
    """Make recommendations for a single user.
    Input 
    list_fics -> List of fics the user has read
    item2ind -> dictionary that maps the fics to their indices
    ind2item -> dictionary that maps the indices to the fics
    word_matrix -> matrix that relates fics to tokens
    K -> Number of recommendations
    Output -> List of recommendations
    """
    recommendations = {}
    for ficName in tqdm(list_fics):
        recom,similarity_scores = recommend_to_single_item(ficName,item2ind,\
                                ind2item,word_matrix)
        r = {}
        for a in range(len(recom)):
            if recom[a] not in list_fics:
                r[recom[a]] = similarity_scores[a]
        recommendations[ficName] = r
    recom = join_recommendations(recommendations, K)
    return recom

def load_recommendations(recommFile):
    """ Loads a list of recommendations into memory.
    Input: File name of the recommendations to upload
    Output: Dictionary where the keys are the users and the values
            are a list of recommendations for said users
    """
    recomm = {}
    with open(recommFile,"r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split("\t")
            if len(dades) == 2:
                recomm[dades[0]] = dades[1].split(";")
    return recomm

def recommend_with_precomputed(infileName,outfileName,user2items,K):
    """Makes recommendations for a user based on pre-computed item2item
    recommendations.
    Input:
    infileName -> File containing the pre-computed recommendations
    outfileName -> File where user item recommendations will be printed
    user2items -> List of fics read by a user
    K -> Number of recommendations
    """
    recommendationsItems = load_recommendations(infileName)
    with open(outfileName,"w") as outfile:
        for user, list_fics in tqdm(user2items.items()):
            counter = {}
            for ficName in list_fics:
                if ficName in recommendationsItems:
                    recom = recommendationsItems[ficName]
                    for r in recom:
                        if r not in list_fics:
                            if r not in counter:
                                counter[r] = 0
                            counter[r] += 1
            recoms = list(counter.keys())
            recoms = sorted(recoms,key=lambda x: counter[x],reverse=True)
            print(user+"\t"+";".join(recoms[:K]),file=outfile)

parser = argparse.ArgumentParser(description="Content based recommender")
parser.add_argument("-i",dest="metadataFile",action="store",required=True,\
    help="File containing the fics metadata")
parser.add_argument("-u",dest="userInterest",action="store",default=None,\
    help="Id of the user you want to search similarities to")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-p",dest="preComp",action="store",default=None,\
    help="Contains a set of pre-computed predictions for each fic")
parser.add_argument("-w",dest="word_method",action="store",\
    choices=["tfidf","counts"],default="tfidf",help="Method used to \
    create the word matrix")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("--predict_all",dest="pred_all",action="store_true",\
    help="Makes recommendations for all users")
parser.add_argument("-k",dest="numR",action="store",\
    type=int,default=15,help="Number of recommendations")
parser.add_argument("--number_words",dest="numW",action="store",type=int,\
    default=10000,help="Number of words in Tfid analysis")
parser.add_argument("--number_recommendations",dest="numR",action="store",\
    type=int,default=15,help="Number of recommendations")
parser.add_argument("--add_tags",dest="addT",action="store_true",\
    help="Adds additional tags to build the word matrix")
parser.add_argument("--add_characters",dest="addC",action="store_true",\
    help="Adds character information to build the word matrix")
parser.add_argument("--add_relationships",dest="addR",action="store_true",\
    help="Adds relationship information to build the word matrix")
parser.add_argument("--add_authors",dest="addA",action="store_true",\
    help="Adds author name to build the word matrix")
parser.add_argument("--add_fandoms",dest="addF",action="store_true",\
    help="Adds fandom information to build the word matrix")
parser.add_argument("--minSimilarity",dest="thr",action="store",type=float,\
    default=0.1,help="Minimum similarity score for a recommendation to be\
    considered")
parser.add_argument("--print_vocabulary",dest="printVoc",action="store",\
    default=None,help="Name of a file where the vocabulary obtained in the\
     bag of words will be printed. If empty it will not be printed")
args = parser.parse_args()
 
#Load metadata into memory
df = pd.read_csv(args.metadataFile,sep="\t",na_values="-",\
     usecols=["idName","title","author","additional_tags","characters",\
     "relationships","numWords","published_date","date_update","fandoms"],\
     parse_dates=["published_date","date_update"],date_parser=parse_date)
df['idName'] = df['idName'].astype("str")
df = df.fillna("")

#Load user data into memory
user2items = CF.load_dataset(args.user2item)

#Decide which information will be used for the bag of words
addT,addC,addA,addR,addF = args.addT,args.addC,args.addA,args.addR,args.addF
if not addT and not addC and not addA and not addR and not addF:
    addT, addC, addR, addA, addF = True, True, True, True, True

#Create indices for each fic
ind2item, item2ind = {}, {}
for idx,item in enumerate(df["idName"].tolist()): 
    ind2item[idx] = item 
    item2ind[item] = idx 

list_fics = list(item2ind.keys())

word_matrix, main_vocab = CF.create_bag_of_words(df,list_fics,addT, addC, addR, addA,\
                                    addF, args.word_method, args.numW)

word_matrix = word_matrix.tocsr()
    
userName = args.userInterest

if userName:
    list_fics = user2items[userName]
    recom = recommend_to_single_user(list_fics,item2ind,ind2item,word_matrix,args.numR)
    print("Fic id\tTitle and author\tSimilarity")
    for a in range(len(recom)):
        print(recom[a]+"\t"+df[df["idName"] == recom[a]]["title"].to_string(index=False)\
        +" by "+df[df["idName"] == recom[a]]["author"].to_string(index=False))
elif args.pred_all:
    print("Warning: All recommendations will be computed")
    if not args.preComp:
        recommend_all_users(df,word_matrix,user2items,args.outfileName,args.numR)
    else:
        recommend_with_precomputed(args.preComp,args.outfileName,user2items,args.numR)

else:
    exit("You need to supply a fic id or activate the option predict_all")
