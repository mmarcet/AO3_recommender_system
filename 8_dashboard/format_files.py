#!/uusr/bin/env python

"""
  AO3_recommender - a recommendation system for fanfiction
  Copyright (C) 2020 - Marina Marcet-Houben
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#Create basic files for dash

import pandas as pd
import argparse

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

parser = argparse.ArgumentParser(description="Formatting for dash")
parser.add_argument("-t",dest="user2item",action="store",required=True,\
    help="Table that contains the user to item information")
parser.add_argument("-i",dest="metadataFile",action="store",required=True,\
    help="File containing the fics metadata")
parser.add_argument("-b",dest="bestRecom",action="store",required=True,\
    help="List of recommendations user to item that will be shown")
parser.add_argument("-r",dest="bestRecomItem",action="store",required=True,\
    help="List of recommendations item to item that will be shown")
args = parser.parse_args()

df_met = pd.read_csv(args.metadataFile,sep="\t",na_values="-",\
    parse_dates=["published_date","date_update"],date_parser=parse_date,\
    usecols=["idName","author","title","published_date","date_update","numWords",\
    "numChapters","numHits","numKudos","numBookmarks","numComments"])

df = pd.read_csv(args.user2item,sep="\t")
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

outfile = open("recom.u2i.csv","w")
for line in open(args.bestRecom):
    line = line.strip()
    if ";" in line:
        line = line.replace("\t",",").replace(";",",")
        print(line,file=outfile)
outfile.close()

outfile = open("recom.i2i.csv","w")
for line in open(args.bestRecomItem):
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
