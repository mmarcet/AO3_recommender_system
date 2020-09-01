#!/usr/bin/env python

import pandas as pd
import random
from tqdm import tqdm
import argparse
import numpy as np

import sys
sys.path.append("../")
sys.path.append(".")
import common_functions as CF

#Clean metadata obtained from the AO3 scrapper and generate the input
#files for the different programs

def curate_dataset(training, validation, test, users):
    """ Given a set of users it tries to find a set of users and items
    that are represented in all three sets.
    
    Input:
    training -> user item dictionary for training
    validation -> user item dictionary for validation
    test -> user item dictionary for test
    users -> initial selected set of users
    
    Output:
    toKeepUsers -> List of users that should be kept
    toKeepFics -> list of items that should be kept
    """
    toKeepUsers = set([])
    toKeepFics = set([])
    rounds = 1
    while len(toKeepUsers) != len(users) and rounds < 10:
        if len(toKeepUsers) != 0:
            users = toKeepUsers
        toKeepFics = map_fic_presence(training, validation, test, users)
        toKeepUsers = map_user_presence(training, validation, test, users, toKeepFics)
        print("Round:",rounds,len(users),len(toKeepUsers),len(toKeepFics))
    rounds += 1
    return toKeepUsers, toKeepFics
    
def map_fic_presence(training, validation, test, users):
    """ Checks which items are present in the three datasets
    Input:
    training -> user item dictionary for training
    validation -> user item dictionary for validation
    test -> user item dictionary for test
    users -> initial selected set of users
    Output:
    toKeepFics -> returns a list of items that are present in the three 
        groups
    """
    fics_taken = {}
    for user in users:
        for f in training[user]:
            if f not in fics_taken:
                fics_taken[f] = set([])
            fics_taken[f].add("T")
        for f in validation[user]:
            if f not in fics_taken:
                fics_taken[f] = set([])
            fics_taken[f].add("V")
        for f in test[user]:
            if f not in fics_taken:
                fics_taken[f] = set([])
            fics_taken[f].add("Te")
    toKeepFics = set([])
    for f in fics_taken:
        if len(fics_taken[f]) == 3:
            toKeepFics.add(f)
    return toKeepFics

def map_user_presence(training, validation, test, users, fics):
    """ Checks that, after deleting items that were not in all three groups,
    all users are still in the three groups.
        Input:
    training -> user item dictionary for training
    validation -> user item dictionary for validation
    test -> user item dictionary for test
    users -> initial selected set of users
    fics -> List of fics that should be kept
    Output:
    toKeepUsers -> returns a list of usersthat are present in the three 
        groups
    """ 
    
    users_taken = {}
    for user in users:
        if user not in users_taken:
            users_taken[user] = set([])
        present = set(training[user]).intersection(fics)
        if len(present) != 0:
            users_taken[user].add("T")
        present = set(validation[user]).intersection(fics)
        if len(present) != 0:
            users_taken[user].add("V")
        present = set(test[user]).intersection(fics)
        if len(present) != 0:
            users_taken[user].add("Te")
    toKeepUsers = set([])
    for u in users_taken:
        if len(users_taken[u]) == 3:
            toKeepUsers.add(u)
    return toKeepUsers

def split_train_test_random(infileName,numUsers,outfileTrain,outfileVal,\
    outfileTest,tag):
    """Create a single training, validation and test sets. The dataset 
    will delete fics that are not present at least once in each one
    of the sets.
    Input:
    infileName -> table containing the user item rating trios
    numUsers -> Number of users to start the dataset (the number can 
            decrease due to the consistency between training, validation
            and test.
    outfileTrain, outfileVal, outfileTest -> Names for the training, 
            validation and test tables
    tag -> Indicator as to the users should be taken randomly or rather
        the ones that have read the most of the less fics.
    """
    info = {}
    with open(infileName,"r") as infile:
        for line in infile:
            line = line.strip()
            if "user" not in line and "item" not in line:
                dades = line.split("\t")
                if dades[0] not in info:
                    info[dades[0]] = {}
                info[dades[0]][dades[1]] = dades[2]

    training, validation, test = {}, {}, {}
    all_users = set([])
    for userId, fics in tqdm(info.items()):
        all_users.add(userId)
        fics = list(fics.keys())
        numFics = len(fics)
        trainingSplit = int(numFics*0.7)
        validationSplit = trainingSplit + int(numFics*0.15)
        random.shuffle(fics)
        training[userId] = fics[:trainingSplit]
        validation[userId] = fics[trainingSplit:validationSplit]
        test[userId] = fics[validationSplit:]

    all_users = list(all_users)
    if tag == "Best":
        all_users = sorted(all_users,key=lambda x: len(info[x]),reverse=True)
        users_subset = set(all_users[:numUsers])
    else:
        random.shuffle(all_users)
        users_subset = set(all_users[:numUsers])

    all_users, all_fics = curate_dataset(training,validation,test,users_subset)

    with open(outfileTrain,"w") as outfileT, open(outfileVal,"w") as outfileV, open(outfileTest,"w") as outfileTe:
        print("user\titem\trating",file=outfileT)
        print("user\titem\trating",file=outfileV)
        print("user\titem\trating",file=outfileTe)
        for userId in tqdm(all_users):
            for f in training[userId]:
                if f in all_fics:
                    print(userId+"\t"+f+"\t"+info[userId][f],file=outfileT)
            for f in validation[userId]:
                if f in all_fics:
                    print(userId+"\t"+f+"\t"+info[userId][f],file=outfileV)
            for f in test[userId]:
                if f in all_fics:
                    print(userId+"\t"+f+"\t"+info[userId][f],file=outfileTe)

def clean_metadata(infileName,outfileName):
    """ Cleans the metadata obtained from the archive
    Input -> Name of the file where the raw metadata has been collected
    Output -> Name of the resulting cleaned metadata
    """
    header = ["idName","author","title","published_date","date_update",\
    "series","numWords","numChapters","warnings","fandoms","required_tags",\
    "relationships","characters","additional_tags","numHits","numKudos",\
    "numBookmarks","numComments","readers_kudos"]
    idNames = set([])
    with open(outfileName,"w") as outfile:
        print("\t".join(header),file=outfile)
        with open(infileName,"r") as infile:
            for line in infile:
                line = line.strip()
                dades = line.split("\t")
                ficId = dades[0]
                #Remove duplicates if present
                if ficId not in idNames:
                    idNames.add(ficId)
                    #Fill empty fields with - it will be our NA value
                    num_unknown_fields = dades.count("-")
                    if num_unknown_fields > 3:
                        pass
                    else:
                        #Replace , in numbers
                        dades[6] = dades[6].replace(",","")
                        #Add - to empty fields
                        for a in range(len(dades)):
                            if dades[a] == "":
                                dades[a] = "-"
                        #Pandas does not like " in title apparently
                        dades[2] = dades[2].replace('"',"'")
                        #Remove spaces so that each tag is considered as one 
                        #entity then substitute | in tags, characters and 
                        #relationships for spaces so that they can be interpreted 
                        #as different words
                        dades[9] = dades[9].replace(" ","").replace("/","").replace("|"," ")
                        dades[11] = dades[11].replace(" ","").replace("/","").replace("|"," ")
                        dades[12] = dades[12].replace(" ","").replace("|"," ")
                        dades[13] = dades[13].replace(" ","").replace("|"," ")
                        print("\t".join(dades),file=outfile)

def get_user2item_table(inputFile,outFile,minNumReads,minNumLikes):
    """ Creates the user to item table which is formated as a three
    column table that includes the username the id of the fic and a 1
    because we do not have more traditional ratings
    Input:
    inputFile -> Name of the file where the data can be found, usually
                the result of the clean_metadata option
    outFile -> Name where the user to item table will be printed
    minNumReads -> Minimum number of likes an item needs to have to be 
                    included in the dataset
    minNumLikes -> Minimum number of likes a user needs to have given
                to be included in the dataset
    """
    relations = {}
    item_likes = {}
    with open(inputFile,"r") as infile:
        for line in infile:
            line = line.strip()
            if "idName" in line:
                pass
            else:
                dades = line.split("\t")
                idName = dades[0]
                author = dades[1]
                users = dades[-1].split("|")
                users.append(author)
                item_likes[idName] = len(users)
                for u in users:
                    if u != "nan":
                        if u not in relations:
                            relations[u] = set([])
                        relations[u].add(idName)
    
    items2keep = set([x for x in item_likes if item_likes[x] > minNumReads])
    with open(outFile,"w") as outfile:
        print("user\titem\trating",file=outfile)
        for user in relations:
            items = relations[user]
            items = items.intersection(items2keep)
            if len(items) > minNumLikes:
                for i in items:
                    print(user+"\t"+i+"\t1.0",file=outfile)

def get_user2author_table(inputFile,outFile):
    """ Builds a user to author table with rankings according the percentage
    of fics written by the author that the user has read.
    Input
    inputFile -> Name of the file where the data can be found, usually
                the result of the clean_metadata option
    outFile -> Name where the user to item table will be printed
    """
    
    relations = {}
    authors = {}
    with open(inputFile,"r") as infile:
        for line in infile:
            line = line.strip()
            if "idName" in line:
                pass
            else:
                dades = line.split("\t")
                idName = dades[0]
                author = dades[1]
                users = dades[-1].split("|")
                if author not in authors:
                    authors[author] = 0
                authors[author] += 1
                for u in users:
                    if u != author:
                        if u not in relations:
                            relations[u] = {}
                        if author not in relations[u]:
                            relations[u][author] = set([])
                        relations[u][author].add(idName)
    #Get ratings from 1 to 5
    with open(outFile, "w") as outfile:
        print("user\titem\trating",file=outfile)
        for u in relations:
            for author in relations[u]:
                fics = relations[u][author]
                fics_total = authors[author]
                if fics_total >= 5:
                    rating = int(len(fics) / fics_total * 5)
                    if rating == 0:
                        if len(fics) != 0:
                            rating = 1
                    print(u+"\t"+author+"\t"+str(rating),file=outfile)

def get_author_metadata(inputFile,outputFile):
    """Summaryzes data from different works to create the author metadata.
    Input:
    inputFile -> contains the cleaned metadata file
    outputFile -> contains the name of the file where the metadata will be
        printed
    """    
    metadata = {}
    with open(inputFile,"r") as infile:
        for line in infile:
            line = line.strip()
            dades = line.split("\t")
            if "numHits" and "numKudos" in line:
                pass
            else:
                author = dades[1]
                if author not in metadata:
                    metadata[author] = {}
                if "date1" not in metadata[author]:
                    metadata[author]["date1"] = CF.parse_date(dades[3])
                elif metadata[author]["date1"] < CF.parse_date(dades[3]):
                    metadata[author]["date1"] = CF.parse_date(dades[3])
                if "date2" not in metadata[author]:
                    metadata[author]["date2"] = CF.parse_date(dades[4])
                elif metadata[author]["date2"] < CF.parse_date(dades[4]):
                    metadata[author]["date2"] = CF.parse_date(dades[4])
                if "numWords" not in metadata[author]:
                    metadata[author]["numWords"] = []
                    metadata[author]["fandoms"] = {}
                    metadata[author]["relationships"] = {}
                    metadata[author]["characters"] = {}
                    metadata[author]["additional_tags"] = {}
                    metadata[author]["numHits"] = []
                    metadata[author]["numKudos"] = []
                    metadata[author]["numBookmarks"] = []
                    metadata[author]["numComments"] = []
                metadata[author]["numWords"].append(int(dades[6]))
                metadata[author]["numHits"].append(int(dades[14]))
                metadata[author]["numKudos"].append(int(dades[15]))
                metadata[author]["numBookmarks"].append(int(dades[16]))
                metadata[author]["numComments"].append(int(dades[17]))
                fandoms = dades[9].split()
                for f in fandoms:
                    if f not in metadata[author]["fandoms"]:
                        metadata[author]["fandoms"][f] = 0
                    metadata[author]["fandoms"][f] += 1
                relationships = dades[11].split()
                for f in relationships:
                    if f not in metadata[author]["relationships"]:
                        metadata[author]["relationships"][f] = 0
                    metadata[author]["relationships"][f] += 1
                characters = dades[12].split()
                for f in characters:
                    if f not in metadata[author]["characters"]:
                        metadata[author]["characters"][f] = 0
                    metadata[author]["characters"][f] += 1
                additional_tags = dades[13].split()
                for f in additional_tags:
                    if f not in metadata[author]["additional_tags"]:
                        metadata[author]["additional_tags"][f] = 0
                    metadata[author]["additional_tags"][f] += 1
    
    with open(outputFile,"w") as outfile:
        print("idName\tauthor\tpublished_date\tdate_update\tnumWords\t\
fandoms\trelationships\tcharacters\tadditional_tags\tnumHits\t\
numKudos\tnumBookmarks\tnumComments",file=outfile)
        for author, info in metadata.items():
            string = str(author)+"\t"+author
            string += "\t"+str(info["date1"]).split()[0]
            string += "\t"+str(info["date2"]).split()[0]
            string += "\t"+str(int(np.average(info["numWords"])))
            fandoms = list(info["fandoms"].keys())
            fandoms = sorted(fandoms,key=lambda x: info["fandoms"][x],reverse=True)
            string += "\t"+" ".join(fandoms[:10])
            relationships = list(info["relationships"].keys())
            relationships = sorted(relationships,key=lambda x: info["relationships"][x],reverse=True)
            string += "\t"+" ".join(relationships[:10])
            characters = list(info["characters"].keys())
            characters = sorted(characters,key=lambda x: info["characters"][x],reverse=True)
            string += "\t"+" ".join(characters[:10])
            additional_tags = list(info["additional_tags"].keys())
            additional_tags = sorted(additional_tags,key=lambda x: info["additional_tags"][x],reverse=True)
            string += "\t"+" ".join(additional_tags[:20])
            string += "\t"+str(int(np.average(info["numHits"])))
            string += "\t"+str(int(np.average(info["numKudos"])))
            string += "\t"+str(int(np.average(info["numBookmarks"])))
            string += "\t"+str(int(np.average(info["numComments"])))
            print(string,file=outfile)
    
    
parser = argparse.ArgumentParser(description="Content based recommender")
parser.add_argument("-i",dest="inputFile",action="store",required=True,\
    help="File containing raw or clean metadata")
parser.add_argument("-o",dest="outFile",action="store",\
    default=None,help="Output file where the data will be printed")
parser.add_argument("--clean_metadata",dest="cleanMeta",action="store_true",\
    help="Cleans the raw metadata and outputs the cleaned metadata")
parser.add_argument("--obtain_user_item_file",dest="user2item",\
    action="store_true",help="Creates a user to item file")
parser.add_argument("--obtain_user_authors_file",dest="user2author",\
    action="store_true",help="Creates a user to author file")
parser.add_argument("--obtain_author_metadata",dest="author_meta",\
    action="store_true",help="Creates an author metadata file")
parser.add_argument("--split_data",dest="split_data",action="store_true",\
    help="Splits data into training, validation and test randomly masking items")
parser.add_argument("--min_num_reads",dest="minNumReads",type=int,
    action="store",default=0,help="Number of times a fic has to have \
    been read to keep it")
parser.add_argument("--min_num_liked",dest="minNumLikes",type=int,
    action="store",default=0,help="Number of Likes a user needs to have \
    given in order to be included")
parser.add_argument("--tag",dest="tag",action="store",\
    choices=["Best","Random"],default="Best",\
    help="Whether the users taken have the most information or are\
    picked randomly")
parser.add_argument("--num_users",dest="numUsers",action="store",\
    type=int,default=10000,\
    help="Initial number of users picked for the training dataset")
parser.add_argument("--outTrain",dest="outfileTrain",action="store",\
    default="train.txt", help="Training outfile")
parser.add_argument("--outVal",dest="outfileVal",action="store",\
    default="validation.txt", help="Validation outfile")
parser.add_argument("--outTest",dest="outfileTest",action="store",\
    default="test.txt", help="Test outfile")
args = parser.parse_args()


if args.cleanMeta:
    if not args.outFile:
        exit("An outfile name needs to be provided (-o)")
    clean_metadata(args.inputFile,args.outFile)
elif args.user2item:
    if not args.outFile:
        exit("An outfile name needs to be provided (-o)")
    get_user2item_table(args.inputFile,args.outFile,args.minNumReads,\
                        args.minNumLikes)
elif args.split_data:
    split_train_test_random(args.inputFile,args.numUsers,args.outfileTrain, \
                    args.outfileVal,args.outfileTest,args.tag)
elif args.user2author:
    if not args.outFile:
        exit("An outfile name needs to be provided (-o)")
    get_user2author_table(args.inputFile,args.outFile)
elif args.author_meta:
    get_author_metadata(args.inputFile,args.outFile)
    

