#!/usr/bin/env python

#Content based recommender system - user to item

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse
import argparse
from tqdm import tqdm

import sys
sys.path.append("../")
import common_functions as CF

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

parser = argparse.ArgumentParser(description="Content based recommender")
parser.add_argument("-i",dest="metadataFile",action="store",required=True,\
    help="File containing the fics metadata")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("-w",dest="word_method",action="store",\
    choices=["tfidf","counts"],default="tfidf",help="Method used to \
    create the word matrix")
parser.add_argument("--number_words",dest="numW",action="store",type=int,\
    default=10000,help="Number of words in analysis")
parser.add_argument("-k","--number_recommendations",dest="numR",action="store",\
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


#Loads table into pandas and creates sparse matrix
print("Building sparse matrices")
df_ui = pd.read_csv(args.user2item,sep="\t",\
            dtype={"user":str,"item":str,"rating":float})
ind2item, item2ind, user2ind, ind2user = CF.create_indices(df_ui)
user_item = CF.create_sparse_matrix(df_ui,user2ind,item2ind)

#Load metadata into memory
df = pd.read_csv(args.metadataFile,sep="\t",na_values="-",\
     usecols=["idName","title","author","fandoms","additional_tags","characters",\
     "relationships"],dtype={"idName":str})
df = df.fillna("")

#Decide which information will be used for the bag of words
addT,addC,addA,addR,addF = args.addT,args.addC,args.addA,args.addR,args.addF
if not addT and not addC and not addA and not addR and not addF:
    addT, addC, addR, addA, addF = True, True, True, True, True
#Create a list with fics under consideration
list_fics = list(item2ind.keys())

#Create bag of words
word_matrix, main_vocab = CF.create_bag_of_words(df,list_fics,addT, addC, addR, addA,\
                                    addF, args.word_method, args.numW)

if args.printVoc:
    with open(args.printVoc,"w") as outfile:
        for word,freq in main_vocab.items():
            print(word+"\t"+str(freq),file=outfile)

with open(args.outfileName,"w") as outfile:
    for userName, userInd in tqdm(user2ind.items()):
        #Extract user to words vector
        u2w = np.asarray(user_item[userInd].dot(word_matrix).sum(axis=0))[0]

        #Calculate similarity between user profile and all fic profiles
        similarity_vector = cosine_similarity(u2w.reshape(1,-1),word_matrix)
        recom, sim = get_recommendation(similarity_vector,ind2item)
        print(userName+"\t"+";".join(recom),file=outfile)

    
