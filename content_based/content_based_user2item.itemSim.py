#!/usr/bin/env python

#Content based recommender system - user to item

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse
import argparse
from tqdm import tqdm

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
                    max_features=number_words) 
        word_matrix = tfidf.fit_transform(metadata)
    else:
        count_vec = CountVectorizer(stop_words="english",\
                        max_features=number_words)
        word_matrix = count_vec.fit_transform(metadata)
    return word_matrix

def create_bag_of_words(df, list_fics):
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
    df_red = df[df["idName"].isin(list_fics)]
    word_matrix = get_word_matrix(df_red["metadata"],args.word_method,args.numW)   
    return word_matrix

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
parser.add_argument("-w",dest="word_method",action="store",\
    choices=["tfidf","counts"],default="tfidf",help="Method used to \
    create the word matrix")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("--number_words",dest="numW",action="store",type=int,\
    default=10000,help="Number of words in Tfid analysis")
parser.add_argument("-k","--number_recommendations",dest="numR",action="store",\
    type=int,default=15,help="Number of recommendations")
parser.add_argument("--add_characters",dest="addC",action="store_true",\
    help="Adds character information to the metadata")
parser.add_argument("--add_relationships",dest="addR",action="store_true",\
    help="Adds relationship information to the metadata")
parser.add_argument("--minSimilarity",dest="thr",action="store",type=float,\
    default=0.1,help="Minimum similarity score for a recommendation to be\
    considered")
args = parser.parse_args()


#Loads table into pandas and creates sparse matrix
print("Load datasets into memory:")
df_train = pd.read_csv(args.user2item,sep="\t",names=["user","fic","rating"])
df_train['fic'] = df_train['fic'].astype("str")
print("Create_indices:")
ind2item, item2ind, user2ind, ind2user = create_indices(df_train)
print("Create sparse matrices:")
user_item_train = create_sparse_matrix(df_train,user2ind,item2ind)

#Load metadata into memory
df = pd.read_csv(args.metadataFile,sep="\t",na_values="-",\
     usecols=["idName","title","author","additional_tags","characters",\
     "relationships"])
df['idName'] = df['idName'].astype("str")
df = df.fillna("")

#Create a list with fics under consideration
list_fics = list(item2ind.keys())

#Create bag of words
word_matrix = create_bag_of_words(df,list_fics)

with open(args.outfileName,"w") as outfile:
    for userName, userInd in tqdm(user2ind.items()):
        #Extract user to words vector
        u2w = np.asarray(user_item_train[userInd].dot(word_matrix).sum(axis=0))[0]

        #Calculate similarity between user profile and all fic profiles
        similarity_vector = cosine_similarity(u2w.reshape(1,-1),word_matrix)
        recom, sim = get_recommendation(similarity_vector,ind2item)
        print(userName+"\t"+";".join(recom),file=outfile)

    
