#!/usr/bin/env python

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

import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

##Common modules
def map_ids(row,mapper):
    """Returns the value of a dicctionary
    input 
    row -> value you're searching for
    mapper -> dictionary
    Output: value of the dictionary
    """
    return mapper[row]

def create_indices(df):
    """ Creates a list of dictionaries that will map users to indices, 
    items to indices and the other way around
    Input: df -> dataframe containing the user\titem\trating information
    Output: Returns four dictionaries: items to indices, users to indices,
    and indices to users and indices to items
    """
    ind2item, item2ind = {}, {} 
    for idx,item in enumerate(df["item"].unique().tolist()): 
        ind2item[idx] = item 
        item2ind[item] = idx 
    user2ind, ind2user = {}, {} 
    for idx,uid in enumerate(df["user"].unique().tolist()): 
        user2ind[uid] = idx 
        ind2user[idx] = uid 
    return ind2item, item2ind, user2ind, ind2user

def create_sparse_matrix(df,user2ind,item2ind): 
    """ Builds a csr matrix.
    Input:
    df -> dataframe that contains user and item information
    user2ind -> dictionary that maps users to the indices
    item2ind -> dictionary that maps items to the indices
    Output -> Sparse matrix of users to items where all values
    of interaction between user and item are 1
    """
    U = df["user"].apply(map_ids, args=[user2ind]).values 
    I = df["item"].apply(map_ids, args=[item2ind]).values
    V = np.ones(I.shape[0]) 
    sparse_user_item = sparse.coo_matrix((V, (U, I)), dtype=np.float64) 
    sparse_user_item = sparse_user_item.tocsr() 
    return sparse_user_item

def get_word_matrix(metadata,word_method,number_words):
    if word_method == "tfidf":    
        model = TfidfVectorizer(stop_words="english",\
            max_features=number_words,max_df=0.98) 
        word_matrix = model.fit_transform(metadata)
        vocabulary = model.vocabulary_
    else:
        model = CountVectorizer(stop_words="english",\
            max_features=number_words,max_df=0.98)
        model.fit(metadata)
        word_matrix = model.transform(metadata)
        vocabulary = model.vocabulary_
    return word_matrix,vocabulary

def create_bag_of_words(df, list_fics, addT, addC, addA, addR, addF, word_method, num_words):
    """ Creates the word matrix.
    Input: df -> dataframe containing all the information considered
    Output: 
        word_matrix -> matrix that relates fics to tokens
        df -> dataframe that now contains the "metadata" column
    """
    df_train = df[df["idName"].isin(list_fics)]
    list_words = []
    main_vocab = {}
    if addT:
        w_tags, vocab_tags = get_word_matrix(df_train["additional_tags"],word_method,num_words)
        list_words.append(w_tags)
        main_vocab.update(vocab_tags)
    if addC:
        w_char, vocab_char = get_word_matrix(df_train["characters"],word_method,num_words)
        list_words.append(w_char)
        main_vocab.update(vocab_char)
    if addR:
        w_rel, vocab_rel = get_word_matrix(df_train["relationships"],word_method,num_words)
        list_words.append(w_rel)
        main_vocab.update(vocab_rel)
    if addA:
        w_aut, vocab_aut = get_word_matrix(df_train["author"],word_method,num_words)
        list_words.append(w_aut)
        main_vocab.update(vocab_aut)
    if addF:
        w_fan, vocab_fan = get_word_matrix(df_train["fandoms"],word_method,num_words)
        list_words.append(w_fan)
        main_vocab.update(vocab_fan)

    if len(list_words) == 1:
        word_matrix = list_words[0]
    elif len(list_words) == 0:
        exit("Please select at least one item with which to build the word matrix")
    else:
        word_matrix = sparse.hstack(list_words)
    return word_matrix, main_vocab


def load_dataset(infile):
    """ Imports list of fics that a user has read. When testing this
    should be the training file.
    Input: Name of the file that contains three columns: user\titem\trating
    Output: A dictionary where the user is the key and the value is a set
    of items the user has read.
    """
    data = {}
    with open(infile,"r") as infile:
        for line in infile:
            line = line.strip()
            if "user" in line and "item" in line:
                pass
            else:
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
