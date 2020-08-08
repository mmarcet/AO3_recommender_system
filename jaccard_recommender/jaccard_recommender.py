#!/usr/bin/env python

#Recommender system based on Jaccard

import argparse

#Loads list of items for each user
users2items = {}
with open("users_kudos_50.txt","r") as infile: 
    for line in infile:
        line = line.strip() 
        dades = line.split("\t") 
        users2items[dades[0]] = set(dades[1:])

users = list(users2items.keys())
points = {}

u1 = "anthonyedwardstark"
jaccard = lambda a, b: len(a.intersection(b)) / len(a.union(b)) 
for a in range(len(users)): 
    u2 = users[a] 
    j = jaccard(users2items[u1],users2items[u2])
    if u1 != u2:
        points[u2] = j
    else:
        points[u2] = 0

users = sorted(users,key=lambda x:points[x],reverse=True)

for u in users[:30]:
    print(u,points[u])
most_similar_users = users[:30]

recommendations = {}

for u2 in most_similar_users:
    for item in users2items[u2]:
        if item not in users2items[u1]:
            if item not in recommendations:
                recommendations[item] = 0
            recommendations[item] += points[u2]

items = list(recommendations.keys())
items = sorted(items,key= lambda x: recommendations[x],reverse=True)

for i in items[:10]:
    print(i,recommendations[i])



