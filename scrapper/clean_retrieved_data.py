#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse
import random
from tqdm import tqdm

#Clean metadata obtained from the AO3 scrapper and generate the input
#files for the different programs

def curate_dataset(training, validation, test, users):
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

inputFile = "metadata_fics.txt"
outfileCleaned = "metadata_fics.cleaned.txt"
user_to_item_table_content = "user_to_item.content.txt"
user_to_item_table_collab = "user_to_item.collab.txt"
headerFile = "header.txt"

info = {}
authors = {}
authors_liked = {}
with open(outfileCleaned,"w") as outfile:
    #Print header
    header = [x.strip() for x in open(headerFile)][0]
    idNames = set([])
    print(header,file=outfile)
    with open(inputFile,"r") as infile:
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
                    #Save information for the user to item matrix
                    author = dades[1]
                    if author not in authors:
                        authors[author] = set([])
                    authors[author].add(ficId)
                    name = dades[2]
                    if dades[-1] != "-":
                        readers = dades[-1].split("|")
                    else:
                        readers = None
                    if author not in info:
                        info[author] = {}
                    info[author][ficId] = "2.0"
                    if readers:
                        for r in readers:
                            if r not in info:
                                info[r] = {}
                            if r not in authors_liked:
                                authors_liked[r] = {}
                            if author not in authors_liked[r]:
                                authors_liked[r][author] = set([])
                            authors_liked[r][author].add(ficId)
                            info[r][ficId] = "1.0"
                    #Replace , in numbers
                    dades[6] = dades[6].replace(",","")
                    try:
                        numWords = int(dades[6])
                    except:
                        print("check:",dades[0],num_unknown_fields)
                    #Add - to empty fields
                    for a in range(len(dades)):
                        if dades[a] == "":
                            dades[a] = "-"
                    #Remove spaces so that each tag is considered as one 
                    #entity then substitute | in tags, characters and 
                    #relationships for spaces so that they can be interpreted 
                    #as different words
                    dades[11] = dades[11].replace(" ","").replace("/","").replace("|"," ")
                    dades[12] = dades[12].replace(" ","").replace("|"," ")
                    dades[13] = dades[13].replace(" ","").replace("|"," ")
                    print("\t".join(dades),file=outfile)

all_users = list(info.keys())
#Marks users that have liked less than 10 fics
users_cold_start = set([x for x in all_users if len(info[x]) < 10])

#Marks fics that have been liked less than 10 times
ficCounter = {}
for userId, fics in info.items():
    for fic in fics:
        if fic not in ficCounter:
            ficCounter[fic] = set([])
        ficCounter[fic].add(userId)
fics_cold_start = set([x for x in ficCounter if len(ficCounter[x]) < 10])

#Print the user to item relation that will be used for collaborative analyses
with open(user_to_item_table_collab,"w") as outTableCollab:
    for userId,fics in info.items():
        if userId not in users_cold_start:
            for fic in fics:
                if fic not in fics_cold_start:
                    print(userId+"\t"+fic+"\t"+info[userId][fic],file=outTableCollab)


#Print user to item relations that will be used for content based analyses

with open(user_to_item_table_content, "w") as outTable:
    for userId,fics in info.items():
        for fic in fics:
            print(userId+"\t"+fic+"\t"+info[userId][fic],file=outTable)

with open("user_to_author_table.txt","w") as outfile:
    for user in authors_liked:
        for author in authors_liked[user]:
            if len(authors[author]) > 5:
                ranking = int(len(authors_liked[user][author]) / len(authors[author])*5)
                print(user+"\t"+author+"\t"+str(ranking),file=outfile)

#Create a single training, validation and test sets to evaluate all methods
#The dataset will delete fics that are not present at least once in each one
#of the sets. To make things go faster I'll take a subset of those

training, validation, test = {}, {}, {}
all_users = set([])
for userId, fics in tqdm(info.items()):
    if userId == "nan":
        pass
    else:
        fics = list(fics.keys())
        if len(fics) >= 20:
            all_users.add(userId)
            numFics = len(fics)
            trainingSplit = int(numFics*0.7)
            validationSplit = trainingSplit + int(numFics*0.15)
            random.shuffle(fics)
            training[userId] = fics[:trainingSplit]
            validation[userId] = fics[trainingSplit:validationSplit]
            test[userId] = fics[validationSplit:]

all_users = list(all_users)
all_users = sorted(all_users,key=lambda x: len(info[x]),reverse=True)
users_subset = set(all_users[:100000])

#I artificially add dls because it's the user I've been using for checking that things are going well
users_subset.add("dls")
all_users, all_fics = curate_dataset(training,validation,test,users_subset)

with open("training_user_item.txt","w") as outfileT, open("validation_user_item.txt","w") as outfileV, open("test_user_item.txt","w") as outfileTe:
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
