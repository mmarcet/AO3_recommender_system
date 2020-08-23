#!/usr/bin/env python

#Content based recommender system - item 2 item

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm

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

def get_word_matrix(metadata,word_method,number_words):
    """ Returns a matrix with the tfidf or counter for each 
    word in each fic
    Input:
    metadata -> list of keywords for each fic that will be used to 
                create the bag of words
    word_method -> tag indicating the methodology that will be used to 
                   obtain the bag of words
    number_words -> number of tokens that will be provided
    
    Output: Matrix that related fics to tokens
    
    """
    if word_method == "tfidf":    
        tfidf = TfidfVectorizer(stop_words="english",\
                    max_features=number_words,max_df=0.95) 
        word_matrix = tfidf.fit_transform(metadata)
    else:
        count_vec = CountVectorizer(stop_words="english",\
                        max_features=number_words,max_df=0.95)
        word_matrix = count_vec.fit_transform(metadata)
    return word_matrix

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

def create_bag_of_words(df):
    """ Creates the word matrix.
    Input: df -> dataframe containing all the information considered
    Output: 
        word_matrix -> matrix that relates fics to tokens
        df -> dataframe that now contains the "metadata" column
    """
    df["metadata"] = df["additional_tags"]
    if args.addC:
        df["metadata"] = df["metadata"] + df["characters"]
    if args.addR:
        df["metadata"] = df["metadata"] + df["relationships"]
    word_matrix = get_word_matrix(df["metadata"],args.word_method,args.numW)   
    return word_matrix, df

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
parser.add_argument("--add_characters",dest="addC",action="store_true",\
    help="Adds character information to the metadata")
parser.add_argument("--add_relationships",dest="addR",action="store_true",\
    help="Adds relationship information to the metadata")
parser.add_argument("--minSimilarity",dest="thr",action="store",type=float,\
    default=0.1,help="Minimum similarity score for a recommendation to be\
    considered")
args = parser.parse_args()
 
#Load metadata into memory
df = pd.read_csv(args.metadataFile,sep="\t",na_values="-",\
     usecols=["idName","title","author","additional_tags","characters",\
     "relationships"])
df['idName'] = df['idName'].astype("str")
df = df.fillna("")

#Create bag of words
word_matrix, df = create_bag_of_words(df)

#Create indices for each fic
ind2item, item2ind = {}, {}
for idx,item in enumerate(df["idName"].tolist()): 
    ind2item[idx] = item 
    item2ind[item] = idx 
    
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
