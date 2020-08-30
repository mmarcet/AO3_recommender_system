#!/usr/bin/env python

#Recommendations using lightFM

from lightfm import LightFM
from lightfm.data import Dataset
import csv
import numpy as np
from tqdm import tqdm
import itertools
import scipy.sparse as sparse
import argparse
import pandas as pd

import sys
sys.path.append("../")
import common_functions as CF

def get_iterator(fileName):
    """ Creates an iterator for csv input data.
    Input
    fileName: name of the csv file
    Output
    Iterator
    """
    
    return csv.DictReader((x for x in open(fileName)), delimiter="\t")

def get_all_mappings(mapping):
    """ Gets the mappings between numeric ids and string ids for users or
    items.
    Input:
    mapping -> mappings provided by the lightFM dataset
    Output:
    items -> list of items or users
    item2ind -> dictionary relating items / users to their indices
    ind2item -> dictionary relating indices to items / users
    """
    item2ind = {}
    ind2item = {}
    for item,ind in mapping.items():
        item2ind[item] = ind
        ind2item[ind] = item
    items = list(mapping.values())
    return items,item2ind,ind2item

def get_read_fics(user,interactions_csr):
    """ Gets the list of items already read by the user
    Input:
    user -> index of the user of interest
    interactions_csr -> csr matrix containing the user to item table
    Output:
    read -> collection of items read by the user"""
    user_list = interactions_csr[user].toarray().reshape(-1)
    read = np.where(user_list != 0.0)
    read = set(read[0])
    return read

def sample_hyperparameters():
    """ Randomly provides hyperparameters for exploration"""
    while True:
        yield {
            "no_components": np.random.randint(45, 100),
            "learning_schedule": np.random.choice(["adagrad"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 20),
            "num_epochs": np.random.randint(5, 50),
        }

parser = argparse.ArgumentParser(description="Content based recommender")
parser.add_argument("-i",dest="metadataFile",action="store",default=None,\
    help="File containing the fics metadata")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-o",dest="outfileName",action="store",\
    default="results.item2item.contentbased.txt",\
    help="Output for all the recommendations")
parser.add_argument("-e",dest="explore",action="store_true",\
    help="Explores the space to search for hyperparameters")
parser.add_argument("-n","--num_explorations",dest="numExp",\
    action="store",type=int,default=30,help="Number of times \
    the hyperparameter space will be explored")
parser.add_argument("-c",dest="no_components",type=int,\
    default=50, help="Number of components")
parser.add_argument("-l",dest="learning_schedule",choices=["adagrad","adadelta"],\
    default="adagrad", help="Learning schedule")
parser.add_argument("-loss",dest="loss",choices=["warp","bpr","warp-kos"],\
    default="warp", help="Loss function")
parser.add_argument("-lr",dest="learning_rate",type=float,\
    default=0.005, help="Learning rate")
parser.add_argument("-ia",dest="item_alpha",type=float,\
    default=1e-8,help="Item alpha")
parser.add_argument("-ua",dest="user_alpha",type=float,\
    default=1e-8,help="User alpha")
parser.add_argument("-max_sampled",dest="max_sampled",type=int,\
    default=5,help="Maximum sampled")
parser.add_argument("-num_epochs",dest="num_epochs",type=int,\
    default=15,help="Number of iterations")
parser.add_argument("-w",dest="word_method",action="store",\
    choices=["tfidf","counts"],default="tfidf",help="Method used to \
    create the word matrix")
parser.add_argument("--number_words",dest="numW",action="store",type=int,\
    default=10000,help="Number of words in analysis")
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
parser.add_argument("--print_vocabulary",dest="printVoc",action="store",\
    default=None,help="Name of a file where the vocabulary obtained in the\
    bag of words will be printed. If empty it will not be printed")
parser.add_argument("--threads",dest="threads",action="store",type=int,\
    default=6,help="Number of threads used by lightFM")



args = parser.parse_args()

print("Creating dataset")
dataset = Dataset()
dataset.fit((x["user"] for x in get_iterator(args.user2item)),\
    (x["item"] for x in get_iterator(args.user2item)))
users, user2ind, ind2user = get_all_mappings(dataset.mapping()[0])
items, item2ind, ind2item = get_all_mappings(dataset.mapping()[-1])
(interactions, weights) = dataset.build_interactions(((x["user"], x["item"]) \
    for x in get_iterator(args.user2item)))
interactions_csr = interactions.tocsr()

if args.metadataFile:
    #Load metadata into memory
    df = pd.read_csv(args.metadataFile,sep="\t",na_values="-",\
     usecols=["idName","title","author","additional_tags","characters",\
     "relationships","fandoms"])
    df['idName'] = df['idName'].astype("str")
    df = df.fillna("")

    #Create a list with fics under consideration
    list_fics = [ind2item[x] for x in items]
    
    #Decide which information will be used for the bag of words
    addT,addC,addA,addR,addF = args.addT,args.addC,args.addA,args.addR,args.addF
    if not addT and not addC and not addA and not addR and not addF:
        addT, addC, addR, addA, addF = True, True, True, True, True

    #Create bag of words
    item_features, main_vocab = CF.create_bag_of_words(df,list_fics,addT,\
                    addC, addR, addA,addF, args.word_method, args.numW)
    
    #Format features to add them in the model
    eye = sparse.eye(item_features.shape[0], item_features.shape[0]).tocsr()
    item_features_concat = sparse.hstack((eye, item_features))
    item_features_concat = item_features_concat.tocsr().astype(np.float32)

if args.explore:
    num = 1
    for hyper in itertools.islice(sample_hyperparameters(), args.numExp):
        print("Training model ",num)
        num_epochs = hyper.pop("num_epochs")

        model = LightFM(**hyper)
        if args.metadataFile:
            model.fit(interactions,epochs=num_epochs, num_threads=args.threads,\
                item_features=item_features)
        else:
            model.fit(interactions,epochs=num_epochs, num_threads=args.threads)

        print("Getting predictions...")
        with open(args.outfileName+"."+str(num)+".txt","w") as outfile:
            for m in hyper:
                print(m,hyper[m],file=outfile)
            print("num_epochs",num_epochs,file=outfile)
            for user in tqdm(users[:1000]):
                #Get predictions
                items_user = items.copy()
                p = model.predict(users[0],items)
                results = [x for x in zip(items_user,p)]
                items_user = sorted(items_user,key=lambda x:results[x][1],reverse=True)
                #Get fics already read by the user
                read = get_read_fics(user,interactions_csr)
                items_user = [x for x in items_user if x not in read]
                recom = [ind2item[x] for x in items_user][:50]
                print(ind2user[user]+"\t"+";".join(recom),file=outfile)
        num += 1
else:
    hyper = {
            "no_components": args.no_components,
            "learning_schedule": args.learning_schedule,
            "loss": args.loss,
            "learning_rate": args.learning_rate,
            "item_alpha": args.item_alpha,
            "user_alpha": args.user_alpha,
            "max_sampled": args.max_sampled,
        }
    num_epochs = args.num_epochs
    model = LightFM(**hyper)
    if args.metadataFile:
        model.fit(interactions,epochs=num_epochs, num_threads=8,\
            item_features=item_features)
    else:
        model.fit(interactions,epochs=num_epochs,\
            num_threads=8)
    print("Getting predictions...")
    with open(args.outfileName,"w") as outfile:
        for m in hyper:
            print(m,hyper[m],file=outfile)
        print("num_epochs",num_epochs,file=outfile)
        for user in tqdm(users[:1000]):
            #Get predictions
            items_user = items.copy()
            p = model.predict(users[0],items)
            results = [x for x in zip(items_user,p)]
            items_user = sorted(items_user,key=lambda x:results[x][1],reverse=True)
            #Get fics already read by the user
            read = get_read_fics(user,interactions_csr)
            items_user = [x for x in items_user if x not in read]
            recom = [ind2item[x] for x in items_user][:50]
            print(ind2user[user]+"\t"+";".join(recom),file=outfile)
