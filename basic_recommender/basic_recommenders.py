#!/usr/bin/env python

#Calculates recommendations for most popular fics and for random fics for
#each user

import pandas as pd
import random
import argparse
from tqdm import tqdm

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


parser = argparse.ArgumentParser(description="ALS recommender")
parser.add_argument("-i","--inFile",dest="inFile",action="store",required=True,help="File containing the users to item file")
parser.add_argument("-o","--outFile",dest="outFile",action="store",required=True,help="Prints all recommendations into a file")
parser.add_argument("-d","--metaFile",dest="metaFile",action="store",required=True,help="File containing the metadata")
parser.add_argument("-m",dest="model",action="store",choices=["Random","Longest","MostRecent","NumComments","NumHits","NumBookmarks","NumLikes"],default="NumHits",help="Base for a recommendation")
parser.add_argument("-k",dest="numR",action="store",type=int,default=20,help="Number of recommendations")
args = parser.parse_args()

user2read = load_dataset(args.inFile)
df = pd.read_csv(args.metaFile,sep="\t",na_values="-", \
usecols=["idName","published_date","date_update","numWords","numHits", \
"numKudos","numComments","numBookmarks"], \
parse_dates=["published_date","date_update"],date_parser=parse_date)
df["idName"] = df["idName"].astype("str")

K = args.numR

with open(args.outFile,"w") as outfile:
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
