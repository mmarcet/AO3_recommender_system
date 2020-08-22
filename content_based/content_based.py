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

def get_recommendation(fic,mapping_reduced,similarity_matrix,df_reduced):
    fic_index = mapping_reduced[fic]
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

def reduce_matrix(df,word_matrix,ficName,mode):
    if mode == "clustering":
        df = get_clusters(word_matrix,df)
        print(df)
        if ficName:
            df, df_reduced, word_matrix_reduced = reduce_through_clustering(word_matrix,df,ficName)
    elif mode == "clustering_by_characters":
        word_matrix1 = get_word_matrix(df["characters"],args.word_method,args.numW)
        df = get_clusters(word_matrix1,df)
        if ficName:
            df, df_reduced, word_matrix1_reduced,cluster_idx = reduce_through_clustering(word_matrix1,df,ficName)
            word_matrix_reduced = word_matrix[cluster_idx]
    elif mode == "clustering_by_relationships":
        word_matrix1 = get_word_matrix(df["relationships"],args.word_method,args.numW)
        df = get_clusters(word_matrix1,df)
        if ficName:
            df, df_reduced, word_matrix1_reduced,cluster_idx = reduce_through_clustering(word_matrix1,df,ficName)
            word_matrix_reduced = word_matrix[cluster_idx]
    elif mode == "number_words":
        df_reduced, word_matrix_reduced = reduce_through_wordNumber(word_matrix,df)
    return df_reduced, word_matrix_reduced, df

def recommend_to_single_item(ficName,df_reduced,word_matrix_reduced):
    mapping_reduced = pd.Series(df_reduced.index,index = df_reduced["idName"])
    fic_index = mapping_reduced[ficName]
    similarity_matrix = get_similarity_matrix(word_matrix_reduced)
    recom,similarity_scores = get_recommendation(ficName,mapping_reduced,similarity_matrix,df_reduced)
    return recom,similarity_scores

def recommend_all_items(df,df_reduced,word_matrix,word_matrix_reduced,mode,ficList):
    recommendations = {}
    if len(ficList) == 0:
        ficList = set(df["idName"].to_list())
    if mode == "number_words":
        mapping_reduced = pd.Series(df_reduced.index,index = df_reduced["idName"])
        similarity_matrix = get_similarity_matrix(word_matrix_reduced)
        with open(args.outfileName,"w") as outfile:
            for a in tqdm(range(50000)):
                ficName = df_reduced.iloc[a]["idName"]
                if str(ficName) in ficList:
                    if ficName in mapping_reduced:
                        recom, similarity_score = get_recommendation(ficName,mapping_reduced,similarity_matrix,df_reduced)
                        recommendations[ficName] = [recom, similarity_score]
                        recom = [str(x) for x in recom]
                        print(str(ficName)+"\t"+";".join(recom),file=outfile)
    else:
        with open(args.outfileName,"w") as outfile:
            for cl in tqdm(range(args.numCl)):
                cluster_idx = df[df["cluster"] == cl].index
                df_reduced = df[df["cluster"] == cl]
                df_reduced.reset_index(drop=True,inplace=True)
                word_matrix_reduced = word_matrix[cluster_idx]
                mapping_reduced = pd.Series(df_reduced.index,index = df_reduced["idName"])
                similarity_matrix = get_similarity_matrix(word_matrix_reduced)
                for a in range(df_reduced.shape[0]):
                    ficName = df_reduced.iloc[a]["idName"]
                    if str(ficName) in ficList:
                        recom, similarity_score = get_recommendation(ficName,mapping_reduced,similarity_matrix,df_reduced)
                        recommendations[ficName] = [recom, similarity_score]
                        recom = [str(x) for x in recom]
                        print(str(cl)+"\t"+str(ficName)+"\t"+";".join(recom),file=outfile)
    return recommendations

def recommend_to_single_user(recommendations, list_fics):
    counter = {}
    for fic_parent in list_fics:
        for a in range(len(recommendations[int(fic_parent)][0])):
            fic = recommendations[int(fic_parent)][0][a]
            if fic not in list_fics:
                if fic not in counter:
                    counter[fic] = 0
                counter[fic] += recommendations[fic_parent][1][a][1]
    fics = list(counter.keys())
    fics = sorted(fics, key=lambda x: counter[x], reverse=True)
    return fics[:args.numR]

def recommend_to_user_with_limits(recommendations, list_fics, limited_fics):
    counter = {}
    for fic_parent in list_fics:
        if fic_parent in recommendations:
            for a in range(len(recommendations[fic_parent][0])):
                fic = str(recommendations[fic_parent][0][a])
                if fic not in list_fics:
                    if fic not in counter:
                        counter[fic] = 0
                    counter[fic] += recommendations[fic_parent][1][a][1]
    fics = list(counter.keys())
    fics = [x for x in fics if x not in limited_fics]
    fics = sorted(fics, key=lambda x: counter[x], reverse=True)
    return fics[:args.numR]

def load_dataset(infile):
    data = {}
    with open(infile,"r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split("\t")
            if dades[0] not in data:
                data[dades[0]] = set([])
            data[dades[0]].add(dades[1])
    return data

    
parser = argparse.ArgumentParser(description="Content based recommender")
parser.add_argument("-i",dest="metadataFile",action="store",required=True,help="File containing the fics metadata")
parser.add_argument("-m",dest="filtering_mode",action="store",choices=["clustering","number_words","clustering_by_characters","clustering_by_relationships"],default="number_words",help="How data will be filtered before prediction")
parser.add_argument("-f",dest="ficInterest",action="store",type=int,default=None,help="Id of the fic you want to search similarities to, if none is provided the script will compute recommendations for all users")
parser.add_argument("-u",dest="userInterest",action="store",default=None,help="User of interest, recommends fics according to what the user has read")
parser.add_argument("-w",dest="word_method",action="store",choices=["tfidf","counts"],default="tfidf",help="Method used to create the word matrix")
parser.add_argument("-o",dest="outfileName",action="store",default="results.item2item.contentbased.txt",help="Output for all the recommendations")
parser.add_argument("-r",dest="user_item_table",action="store",default=None,help="User to item csv file")
parser.add_argument("--recommendations_list",dest="recomList",action="store",default=None,help="List of pre-calculated recommendations for the fics obtained with the option predict_all_fics")
parser.add_argument("--predict_all_fics",dest="pred_all_fics",action="store_true",help="Makes recommendations for all fanfictions")
parser.add_argument("--predict_all_users",dest="pred_all_users",action="store_true",help="Makes recommendations for all users")
parser.add_argument("--number_clusters",dest="numCl",action="store",type=int,default=12,help="Number of clusters used in kmeans when clustering filter is selected")
parser.add_argument("--random_seed",dest="randS",action="store",type=int,default=33,help="Random seed to run kmeans")
parser.add_argument("--number_words",dest="numW",action="store",type=int,default=10000,help="Number of words in Tfid analysis")
parser.add_argument("--number_recommendations",dest="numR",action="store",type=int,default=15,help="Number of recommendations")
parser.add_argument("--add_characters",dest="addC",action="store_true",help="Adds character information to the metadata")
parser.add_argument("--add_relationships",dest="addR",action="store_true",help="Adds relationship information to the metadata")
parser.add_argument("--find_optimal_clusters",dest="optimize_clusters",action="store_true",help="Provides graphs to explore the optimal number of clusters, iterated over number of cl and random_state")
args = parser.parse_args()
    
df = pd.read_csv(args.metadataFile,sep="\t",na_values="-")
df['idName'] = df['idName'].astype("str")
df = df.fillna("")
df["metadata"] = df["additional_tags"]

if args.addC:
    df["metadata"] = df["metadata"] + df["characters"]
if args.addR:
    df["metadata"] = df["metadata"] + df["relationships"]

word_matrix = get_word_matrix(df["metadata"],args.word_method,args.numW)

ficName = args.ficInterest
userName = args.userInterest

df_reduced, word_matrix_reduced, df = reduce_matrix(df,word_matrix,ficName,args.filtering_mode)

if ficName:
    recom,similarity_scores = recommend_to_single_item(ficName,df_reduced,word_matrix_reduced)
elif userName:
    train = load_dataset(args.user_item_table)
    if userName in train:
        list_fics = train[userName]
    else:
        exit("The chosen user is not in the training file")
    recommendations = recommend_all_items(df,df_reduced,word_matrix,word_matrix_reduced,args.filtering_mode,list_fics)
    recommend = recommend_to_single_user(recommendations,list_fics)
    print(df_reduced[df_reduced["idName"].isin(recommend)][["idName","title","author","metadata"]])
elif args.pred_all_fics:
    print("Warning: All recommendations will be computed")
    recommendations = recommend_all_items(df,df_reduced,word_matrix,word_matrix_reduced,args.filtering_mode,[])
elif args.pred_all_users:
    train = load_dataset(args.user_item_table)
    all_fics = set([])
    for user in train:
        for f in train[user]:
            all_fics.add(f)
    recommendations = recommend_all_items(df,df_reduced,word_matrix,word_matrix_reduced,args.filtering_mode,all_fics)
    for user in train:
        recommend = recommend_to_user_with_limits(recommendations,train[user],all_fics)
        print(user+"\t"+";".join([str(x) for x in recommend]))
