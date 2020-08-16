#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse

#Clean metadata obtained from the AO3 scrapper

inputFile = "metadata_fics.txt"
outfileCleaned = "metadata_fics.cleaned.txt"
user_to_item_table_content = "user_to_item.content.txt"
user_to_item_table_collab = "user_to_item.collab.txt"

info = {}
ficInfo = {}
with open(outfileCleaned,"w") as outfile:
    #Print header
    header = [x.strip() for x in open("header.txt")][0]
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
                    name = dades[2]
                    ficInfo[ficId] = [name,author,dades[3],dades[6],dades[11]]
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


mappingItems = {}
mappingUsers = {}

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
            mappingUsers[userId] = str(len(mappingUsers))
            for fic in fics:
                if fic not in fics_cold_start:
                    if fic not in mappingItems:
                        mappingItems[fic] = str(len(mappingItems))
                    print(mappingUsers[userId]+"\t"+mappingItems[fic]+"\t"+info[userId][fic],file=outTableCollab)

#Assign indexes to cold start users and fics
for userId in users_cold_start:
    mappingUsers[userId] = str(len(mappingUsers))

for ficId in fics_cold_start:
    mappingItems[ficId] = str(len(mappingItems))

#Print user to item relations that will be used for content based analyses

only_read_by_cold_users = set([])
with open(user_to_item_table_content, "w") as outTable:
    for userId,fics in info.items():
        for fic in fics:
            if fic not in mappingItems:
                mappingItems[fic] = str(len(mappingItems))
                only_read_by_cold_users.add(fic)
            # ~ print(userId,fic,info[userId][fic])
            print(mappingUsers[userId]+"\t"+mappingItems[fic]+"\t"+info[userId][fic],file=outTable)

#Print to file the code conversions for users and fics so that was can get them back
with open("mappingUsers.txt","w") as outUsers:
    for code in mappingUsers:
        if code in users_cold_start:
            tag = "cold"
        else:
            tag = "fine"
        print(code+"\t"+mappingUsers[code]+"\t"+tag,file=outUsers)

with open("mappingItems.txt", "w") as outFics:
    for code in mappingItems:
        if code in fics_cold_start:
            tag = "cold"
        elif tag in only_read_by_cold_users:
            tag = "read_by_col_users"
        else:
            tag = "fine"
        print(code+"\t"+mappingItems[code]+"\t"+tag,file=outFics)
