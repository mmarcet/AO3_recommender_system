#!/usr/bin/env python

#Calculates recommendations for most popular fics and for random fics for
#each user

import pandas as pd
import random
import argparse
from tqdm import tqdm

import sys
sys.path.append("../")
import common_functions as CF


parser = argparse.ArgumentParser(description="Non-personalized recommender")
parser.add_argument("-i",dest="metadataFile",action="store",default=None,\
    help="File containing the fics metadata")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("-m",dest="model",action="store",choices=["Random",\
    "Longest","MostRecent","NumComments","NumHits","NumBookmarks","NumLikes"],\
    default="NumHits",help="Base for a recommendation")
parser.add_argument("-k",dest="numR",action="store",type=int,default=20,\
    help="Number of recommendations")
args = parser.parse_args()

user2read = CF.load_dataset(args.user2item)
df = pd.read_csv(args.metadataFile,sep="\t",na_values="-", \
usecols=["idName","published_date","date_update","numWords","numHits", \
    "numKudos","numComments","numBookmarks"], \
    parse_dates=["published_date","date_update"],date_parser=CF.parse_date)
df["idName"] = df["idName"].astype("str")

K = args.numR

with open(args.outfileName,"w") as outfile:
    for user in tqdm(user2read):
        listFics = user2read[user]
        df_user = df[~df["idName"].isin(listFics)]
        if args.model == "numHits":
            recom = df_user.sort_values(by=["numHits"],ascending=False)["idName"].to_list()[:K]
        elif args.model == "NumComments":
            recom = df_user.sort_values(by=["numComments"],ascending=False)["idName"].to_list()[:K]
        elif args.model == "NumLikes":
            recom = df_user.sort_values(by=["numKudos"],ascending=False)["idName"].to_list()[:K]
        elif args.model == "NumBookmarks":
            recom = df_user.sort_values(by=["numBookmarks"],ascending=False)["idName"].to_list()[:K]
        elif args.model == "Longest":
            recom = df_user.sort_values(by=["numWords"],ascending=False)["idName"].to_list()[:K]
        elif args.model == "MostRecent":
            recom = df_user.sort_values(by=["date_update","published_date"],ascending=False)["idName"].to_list()[:K]
        elif args.model == "Random":
            all_fics = df_user["idName"].to_list()
            random.shuffle(all_fics)
            recom = all_fics[:K]
        print(user+"\t"+";".join(recom),file=outfile)
