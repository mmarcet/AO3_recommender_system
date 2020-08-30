#!/uusr/bin/env python

#Create bases for graphs

import pandas as pd

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

df_met = pd.read_csv("../data/metadata_fics.cleaned.txt",sep="\t",na_values="-",parse_dates=["published_date","date_update"],date_parser=parse_date,usecols=["idName","author","title","published_date","date_update","numWords","numChapters","numHits","numKudos","numBookmarks","numComments"])

df = pd.read_csv("../data/less_read_dataset/training_user_item.txt",sep="\t")
df_cnt = pd.DataFrame(df.groupby('item').size(), columns=['count'])
df_cnt = df_cnt.sort_values('count', ascending=False).reset_index()
df_cnt = df_cnt.reset_index()

df_red = df_met[df_met["idName"].isin(df_cnt["item"].to_list())]
df_red = df_cnt.merge(df_red, left_on='item', right_on='idName')

df_red.to_csv("info_fics.csv", index=False) 

df_cnt = pd.DataFrame(df.groupby('user').size(), columns=['count'])
df_cnt = df_cnt.sort_values('count', ascending=False).reset_index()
df_cnt = df_cnt.reset_index()

df_cnt.to_csv("info_user.csv",index=False)

outfile = open("recom.ALS.csv","w")
for line in open("recom.a55.r0.01.f500.i50.k50.txt"):
    line = line.strip()
    line = line.replace("\t",",").replace(";",",")
    print(line,file=outfile)
outfile.close()

outfile = open("recom.Cont.csv","w")
for line in open("item2item.recommender.txt"):
    line = line.strip()
    line = line.replace(";","\t")
    dades = line.split("\t")
    if len(dades) == 1:
        line=None
    elif len(dades) < 50:
        add2dades = ["-" for x in range(len(dades),50)]
        dades = dades + add2dades
        line = ",".join(dades)
    else:
        line = ",".join(dades)
    if line:
        print(line,file=outfile)
outfile.close()
