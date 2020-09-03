#!/usr/bin/env python

#Content based recommender system - item 2 item

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
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

def recommend_all_items(df,word_matrix,outfileName):
    """ Calculates recommendations for all the items in a dataframe.
    Input:
    df -> Datafram where the list of fics can be found
    word_matrix -> matrix that relates fics to tokens
    outfileName -> file where all recommendations will be printed
    """
    list_fics = df["idName"].to_list()
    with open(outfileName,"w") as outfile:
        for ficName in tqdm(list_fics):
            recom,similarity_scores = recommend_to_single_item(ficName,\
                                        item2ind,ind2item,word_matrix)
            print(ficName+"\t"+";".join(recom),file=outfile)



parser = argparse.ArgumentParser(description="Content based recommender")
parser.add_argument("-i",dest="metadataFile",action="store",required=True,\
    help="File containing the fics metadata")
parser.add_argument("-f",dest="ficInterest",action="store",default=None,\
    help="Id of the fic you want to search similarities to")
parser.add_argument("-w",dest="word_method",action="store",\
    choices=["tfidf","counts"],default="tfidf",help="Method used to \
    create the word matrix")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("--predict_all",dest="pred_all",action="store_true",\
    help="Makes recommendations for all fanfictions")
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
     usecols=["idName","author","fandoms","additional_tags","characters",\
     "relationships"],dtype={"idName":str})
df = df.fillna("")


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

    
ficName = args.ficInterest

if ficName:
    recom,similarity_scores = recommend_to_single_item(ficName,item2ind,\
                            ind2item,word_matrix)
    print("Fic id\tTitle and author\tSimilarity")
    for a in range(len(recom)):
        print(recom[a]+"\t"+df[df["idName"] == recom[a]]["title"].to_string(index=False)\
        +" by "+df[df["idName"] == recom[a]]["author"].to_string(index=False)\
        +"\t"+str(similarity_scores[a]))
elif args.pred_all:
    print("Warning: All recommendations will be computed")
    recommend_all_items(df,word_matrix,args.outfileName)
else:
    exit("You need to supply a fic id or activate the option predict_all")
