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

def get_recommendation(fic,mapping_reduced,similarity_matrix,df_reduced):
    if fic in mapping_reduced:
        fic_index = mapping_reduced[fic]
    else:
        exit("This fic does not appear in the database")
    sc = similarity_matrix[fic_index].toarray().reshape(-1)
    identical = np.where(sc==1.0)
    similarity_score = list(enumerate(similarity_matrix[fic_index].toarray().reshape(-1)))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:args.numR]
    fic_indices = [i[0] for i in similarity_score]
    return (df_reduced["idName"].iloc[fic_indices].tolist(),similarity_score)

def find_optimal_clusters(data, max_k):
    iters = range(6, max_k+1, 2)
    for a in range(100):
        sse = []
        for k in iters:
            sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=a).fit(data).inertia_)
            print('Fit {} clusters'.format(k))
        print(a)
        f, ax = plt.subplots(1, 1)
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot - random_seed = '+str(a))
        plt.show()

def reduce_through_clustering(word_matrix,df,ficName):
    mapping = pd.Series(df.index,index = df["idName"])
    fic_index = mapping[ficName]
    cl = df.iloc[fic_index]["cluster"]
    cluster_idx = df[df["cluster"] == cl].index
    df_reduced = df[df["cluster"] == cl]
    df_reduced.reset_index(drop=True,inplace=True)
    word_matrix_reduced = word_matrix[cluster_idx]
    return df,df_reduced, word_matrix_reduced,cluster_idx

def reduce_through_wordNumber(word_matrix,df):
    df_reduced = df.sort_values(by=["numWords"],ascending=False)[:50000]
    numWords_idx = df_reduced.index
    df_reduced.reset_index(drop=True,inplace=True)
    word_matrix_reduced = word_matrix[numWords_idx]
    return df_reduced, word_matrix_reduced

def get_word_matrix(metadata,word_method,number_words):
    if word_method == "tfidf":    
        tfidf = TfidfVectorizer(stop_words="english",max_features=number_words,max_df=0.95) 
        word_matrix = tfidf.fit_transform(metadata)
    else:
        count_vec = CountVectorizer(stop_words="english",max_features=number_words,max_df=0.95)
        word_matrix = count_vec.fit_transform(metadata)
    return word_matrix

def get_similarity_matrix(word_matrix):
    if args.word_method == "tfidf":
        similarity_matrix = linear_kernel(word_matrix,word_matrix,dense_output=False)
    else:
        similarity_matrix = cosine_similarity(word_matrix,word_matrix,dense_output=False)
    return similarity_matrix

def get_clusters(word_matrix,df):
    if args.optimize_clusters:
        find_optimal_clusters(word_matrix, 20)
        exit()
    clusters = MiniBatchKMeans(n_clusters=args.numCl, init_size=1024, batch_size=2048, random_state=args.numR).fit_predict(word_matrix)
    df["cluster"] = clusters
    return df
    
parser = argparse.ArgumentParser(description="Content based recommender")
parser.add_argument("-i",dest="metadataFile",action="store",required=True,help="File containing the fics metadata")
parser.add_argument("-m",dest="filtering_mode",action="store",choices=["clustering","number_words","clustering_by_characters","clustering_by_relationships"],default="number_words",help="How data will be filtered before prediction")
parser.add_argument("-f",dest="ficInterest",action="store",type=int,default=None,help="Id of the fic you want to search similarities to, if none is provided the script will compute recommendations for all users")
parser.add_argument("-w",dest="word_method",action="store",choices=["tfidf","counts"],default="tfidf",help="Method used to create the word matrix")
parser.add_argument("-o",dest="outfileName",action="store",default="results.item2item.contentbased.txt",help="Output for all the recommendations")
parser.add_argument("--number_clusters",dest="numCl",action="store",type=int,default=12,help="Number of clusters used in kmeans when clustering filter is selected")
parser.add_argument("--random_seed",dest="randS",action="store",type=int,default=33,help="Random seed to run kmeans")
parser.add_argument("--number_words",dest="numW",action="store",type=int,default=10000,help="Number of words in Tfid analysis")
parser.add_argument("--number_recommendations",dest="numR",action="store",type=int,default=15,help="Number of recommendations")
parser.add_argument("--add_characters",dest="addC",action="store_true",help="Adds character information to the metadata")
parser.add_argument("--add_relationships",dest="addR",action="store_true",help="Adds relationship information to the metadata")
parser.add_argument("--find_optimal_clusters",dest="optimize_clusters",action="store_true",help="Provides graphs to explore the optimal number of clusters, iterated over number of cl and random_state")
args = parser.parse_args()
    
df = pd.read_csv(args.metadataFile,sep="\t",na_values="-")

df = df.fillna("")
df["metadata"] = df["additional_tags"]

if args.addC:
    df["metadata"] = df["metadata"] + df["characters"]
if args.addR:
    df["metadata"] = df["metadata"] + df["relationships"]

word_matrix = get_word_matrix(df["metadata"],args.word_method,args.numW)

ficName = args.ficInterest

if args.filtering_mode == "clustering":
    df = get_clusters(word_matrix,df)
    if ficName:
        df, df_reduced, word_matrix_reduced = reduce_through_clustering(word_matrix,df,ficName)
elif args.filtering_mode == "clustering_by_characters":
    word_matrix1 = get_word_matrix(df["characters"],args.word_method,args.numW)
    df = get_clusters(word_matrix1,df)
    if ficName:
        df, df_reduced, word_matrix1_reduced,cluster_idx = reduce_through_clustering(word_matrix1,df,ficName)
        word_matrix_reduced = word_matrix[cluster_idx]
elif args.filtering_mode == "clustering_by_relationships":
    word_matrix1 = get_word_matrix(df["relationships"],args.word_method,args.numW)
    df = get_clusters(word_matrix1,df)
    if ficName:
        df, df_reduced, word_matrix1_reduced,cluster_idx = reduce_through_clustering(word_matrix1,df,ficName)
        word_matrix_reduced = word_matrix[cluster_idx]
elif args.filtering_mode == "number_words":
    df_reduced, word_matrix_reduced = reduce_through_wordNumber(word_matrix,df)

if ficName:
    mapping_reduced = pd.Series(df_reduced.index,index = df_reduced["idName"])
    fic_index = mapping_reduced[ficName]
    similarity_matrix = get_similarity_matrix(word_matrix_reduced)

    recom,similarity_score = get_recommendation(ficName,mapping_reduced,similarity_matrix,df_reduced)

    print(df_reduced[df_reduced["idName"] == ficName][["idName","title","metadata"]])
    print(recom)
    print(similarity_score)
else:
    print("Warning: All recommendations will be computed")
    if args.filtering_mode == "number_words":
        mapping_reduced = pd.Series(df_reduced.index,index = df_reduced["idName"])
        similarity_matrix = get_similarity_matrix(word_matrix_reduced)
        with open(args.outfileName,"w") as outfile:
            for a in range(50000):
                ficName = df_reduced.iloc[a]["idName"]
                recom, similarity_score = get_recommendation(ficName,mapping_reduced,similarity_matrix,df_reduced)
                recom = [str(x) for x in recom]
                print(str(ficName)+"\t"+";".join(recom),file=outfile)
    else:
        with open(args.outfileName,"w") as outfile:
            for cl in range(args.numCl):
                cluster_idx = df[df["cluster"] == cl].index
                df_reduced = df[df["cluster"] == cl]
                df_reduced.reset_index(drop=True,inplace=True)
                word_matrix_reduced = word_matrix[cluster_idx]
                mapping_reduced = pd.Series(df_reduced.index,index = df_reduced["idName"])
                similarity_matrix = get_similarity_matrix(word_matrix_reduced)
                for a in range(df_reduced.shape[0]):
                    ficName = df_reduced.iloc[a]["idName"]
                    recom, similarity_score = get_recommendation(ficName,mapping_reduced,similarity_matrix,df_reduced)
                    recom = [str(x) for x in recom]
                    print(str(cl)+"\t"+str(ficName)+"\t"+";".join(recom),file=outfile)
            
