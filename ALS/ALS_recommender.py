#!/user/bin/env python

#Build ALS_recommender based on implicit library

#Create matrices

import implicit
import pandas as pd
import scipy.sparse as sparse
import pickle
import random
import numpy as np
from sklearn import metrics
import argparse

def obtain_popular_recommendations(sparse_user_item, fics, user, K, sorted_items):
    """ Recommends K most popular items """
    user_row = sparse_user_item[user].toarray().reshape(-1)
    ind1 = np.where(user_row == 1.0)[0]
    recommend = [x[0] for x in sorted_items if x[0] not in ind1][:K]
    return recommend,user_row

def obtain_random_recommendations(fics,user,K):
    """ Recommends K random items """
    #Take out fics that have already been read by the user
    user_row = sparse_user_item[user].toarray().reshape(-1)
    ind1 = np.where(user_row == 1.0)[0]
    rFics = [x for x in fics if x not in ind1]
    #Shuffle the fics
    random.shuffle(rFics)
    #Recommend K random fics
    recommend = rFics[:K]
    return recommend

def calculate_metrics(recommendM,read_by_user,K):
    """Calculates metrics for a given recommendation """
    TP = len(set(recommendM).intersection(set(read_by_user)))
    recall = TP / min(len(read_by_user),K)
    precision = TP / K
    if precision != 0.0 or recall != 0.0:
        F1 = 2 / ((1/precision) + (1/recall))
    else:
        F1 = 0.0
    return recall,precision,F1

def caculate_accuracies(model,train,test):
    av_mae_train = []
    av_mae_test = []
    av_mse_train = []
    av_mse_test = []
    for a in range(0,sparse_user_item.shape[0],3000): 
        pred = model.user_factors[a:a+1000].dot(model.item_factors.T) 
        y = train.T[a:a+1000].toarray()
        mae = metrics.mean_absolute_error(y,pred) 
        mse = metrics.mean_squared_error(y,pred)
        av_mae_train.append(mae)
        av_mse_train.append(mse)
        y = test.T[a:a+1000].toarray()
        mae = metrics.mean_absolute_error(y,pred) 
        mse = metrics.mean_squared_error(y,pred)
        av_mae_test.append(mae)
        av_mse_test.append(mse)
    return np.average(av_mae_train),np.average(av_mse_train),np.average(av_mae_test),np.average(av_mse_train)

def calc_popular_metrics(sparse_user_item, test, train, users, fics, K):
    popularity_auc = []
    random_auc = []
    precisionP = []
    recallP = []
    precisionR = []
    recallR = []

    #Compare for each user of test
    for user in users[:1000]:
        te = test[user].toarray().reshape(-1)
        tr = train[user].toarray().reshape(-1)
        complete = te + tr
        complete[complete==2.0] = 1.0
        aucP = metrics.roc_auc_score(complete,pop)
        aucR = metrics.roc_auc_score(complete,rand)
        # ~ print(user,auc)
        popularity_auc.append(aucP)
        random_auc.append(aucR)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(complete,pop)
        precisionP.append(precision)
        recallP.append(recall)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(complete,rand)
        precisionR.append(precision)
        recallR.append(recall)
    return float('%.3f'%np.mean(popularity_auc)),float('%.3f'%np.mean(random_auc)),float('%.3f'%np.mean(precisionP)),float('%.3f'%np.mean(precisionR)),float('%.3f'%np.mean(recallP)),float('%.3f'%np.mean(recallR))

def mapk(actual, predicted, k):
    """
    Computes the average precision at k.
    
    :param actual : A list of elements that are to be predicted (order doesn't matter)
    :param predicted : A list of predicted elements (order does matter)
    :param k: The maximum number of predicted elements
    
    :return The average precision at k over the input lists
    """
    
    score = 0.0    # This will store the numerator
    num_hits = 0.0 # This will store the sum of rel(i)

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def calculate_all_metrics(users,recommendations,user_item,model,numUsers,K):
    """ Calculates different metrics for a model and its recommendations
    Inputs:
    -> users: list of users
    -> recommendations: set of recommendations @ K as provided by implicit recommend_all
    -> user_item: csr matrix of users vs items
    -> model: model obtained with implicit
    -> numUsers: number of users that will be scanned to provide the metrics
    -> K: number of recommendations asked for
    
    returns:
    Averages of the following metrics: recall, precision, F1, map@k, MAE and RMSE
    
    """
    mainRecall, mainPrec, mainF1, mainMap, mainMAE, mainRMSE = [],[],[],[],[],[]
    for a in range(numUsers):
        user = users[a]
        recommend = recommendations[user]
        user_row = user_item[user].toarray().reshape(-1)
        factors = model.user_factors[user].dot(model.item_factors.T)
        read_by_user = np.where(user_row >= 1.0)[0]
        if len(read_by_user) != 0:
            recall,precision,F1 = calculate_metrics(recommend,set(read_by_user),K)
            score = mapk(set(read_by_user), recommend, K)
            rmse = metrics.mean_squared_error(user_row[read_by_user],factors[read_by_user])
            mae = metrics.mean_absolute_error(user_row[read_by_user],factors[read_by_user])
            mainRecall.append(recall)
            mainPrec.append(precision)
            mainF1.append(F1)
            mainMap.append(score)
            mainMAE.append(mae)
            mainRMSE.append(rmse)
            if a % 200 == 0:
                m = np.average(mainRecall)
                n = np.average(mainPrec)
                f = np.average(mainF1)
                b = np.average(mainMap)
                c = np.average(mainMAE)
                d = np.average(mainRMSE)
                print(a,m,n,f,b,c,d)
    return np.average(mainRecall),np.average(mainPrec),np.average(mainF1),np.average(mainMap),np.average(mainMAE),np.average(mainRMSE)


parser = argparse.ArgumentParser(description="ALS recommender")
parser.add_argument("-t",dest="user_item_table",action="store",required=True,help="Table containing a user - item - ratings table")
parser.add_argument("-a",dest="alpha",action="store",type=int,default = 40, help="Alpha hyperparameter")
parser.add_argument("-f",dest="factors",action="store",type=int,default=100, help="Number of factors hyperparameter")
parser.add_argument("-r",dest="regularization",action="store",type=float,default=0.01, help="Regularization hyperparameter")
parser.add_argument("-i",dest="iterations",action="store",type=int,default=10,help="Number of iterations hyperparameter")
parser.add_argument("-l",dest="learning_rate",action="store",type=float,default=0.01,help="Learning rate")
parser.add_argument("-k",dest="k",action="store",type=int,default=50,help="Number of recommendations per user")
parser.add_argument("-m",dest="model",action="store",default="ALS",help="Model to use - choose between ALS, LMF, BPR")
parser.add_argument("--calc_sparsity",dest="calc_spars",action="store_true",help="Prints matrix sparsity on screen")
parser.add_argument("--trainFile",dest="trainFile",action="store",default=None,help="Provides as saved spare matrix that represents the training set")
parser.add_argument("--valFile",dest="valFile",action="store",default=None,help="Provides as saved spare matrix that represents the validation set")
parser.add_argument("--testFile",dest="testFile",action="store",default=None,help="Provides as saved spare matrix that represents the test set")
parser.add_argument("--numUsers",dest="numUsers",action="store",type=int,default=10000,help="Number of users that will be used to calculate metrics")
parser.add_argument("--calc_train_stats",dest="calc_train_stats",action="store_true",help="Calculates the statistics for the training set too")
args = parser.parse_args()

tableFileName = args.user_item_table
#Loads table into pandas
df_ratings = pd.read_csv(tableFileName,sep="\t",names=["userId","ficId","rating"])
#Creates user and item lists
users = df_ratings['userId'].unique()
fics = df_ratings['ficId'].unique()
#Creates sparse matrices
sparse_user_item = sparse.csr_matrix((df_ratings['rating'].astype(float), (df_ratings['userId'].astype(int), df_ratings['ficId'].astype(int))), shape=(len(users),len(fics)))
sparse_item_user = sparse.csr_matrix((df_ratings["rating"].astype(float), (df_ratings["ficId"].astype(int), df_ratings["userId"].astype(int))), shape=(len(fics),len(users)))
#Prints shape to double check the matrices are named ok
print("IU",sparse_item_user.shape)
print("UI",sparse_user_item.shape)


if args.calc_spars:
    #Calculate sparsity
    matrix_size = sparse_user_item.shape[0]*sparse_user_item.shape[1] # Number of possible interactions in the matrix
    num_reads = len(sparse_user_item.nonzero()[0]) # Number of items interacted with
    sparsity = 100*(1 - (num_reads/matrix_size))
    print(sparsity)

if not args.trainFile:
    validation, test = implicit.evaluation.train_test_split(sparse_item_user,train_percentage=0.85)
    train, validation = implicit.evaluation.train_test_split(validation,train_percentage=0.85)
    sparse.save_npz("train.npz",train)
    sparse.save_npz("validation.npz",validation)
    sparse.save_npz("test.npz",test)
else:
    if args.testFile:
        test = sparse.load_npz(args.testFile)
    if args.valFile:
        validation = sparse.load_npz(args.valFile)
    if args.trainFile:
        train = sparse.load_npz(args.trainFile)

alpha = args.alpha
if args.model == "ALS" or args.model == "BPR":
    model = implicit.als.AlternatingLeastSquares(factors=args.factors,regularization=args.regularization,iterations=args.iterations)
    model.fit(train*alpha)
elif args.model == "LMF":
    model = implicit.lmf.LogisticMatrixFactorization(factors=args.factors, learning_rate=args.learning_rate, regularization=args.regularization,iterations=args.iterations)
    model.fit(train*alpha)
elif args.model == "BPR":
    model = implicit.bpr.BayesianPersonalizedRanking(factors=args.factors, learning_rate=args.learning_rate, regularization=args.regularization,iterations=args.iterations)
    model.fit(train*alpha)



K=args.k
numUsers = args.numUsers

if args.calc_train_stats:
    recomendations_train = model.recommend_all(train.T, K,filter_already_liked_items=False)
    recall, precision, F1, map_at_k, mae, mse = calculate_all_metrics(users,recomendations_train,train.T,model,numUsers,K)
    print("Model train:",recall, precision, F1, map_at_k, mae, mse)

#Recalculates recommendations for validation set
if args.model == "ALS":
    recomendations_val = model.recommend_all(validation.T,K,recalculate_user=True,filter_already_liked_items=False)
else:
    recomendations_val = model.recommend_all(validation.T,K,filter_already_liked_items=False)
recall, precision, F1, map_at_k, mae, mse = calculate_all_metrics(users,recomendations_val,validation.T,model,numUsers,K)
print("Model validation:",recall, precision, F1, map_at_k, mae, mse)

exit()
#Popular item list
pop_items = np.array(sparse_user_item.sum(axis = 0)).reshape(-1)
item_pairs = list(zip(fics,pop_items))
sorted_items = sorted(item_pairs, key=lambda x: x[1],reverse=True)
mainF1P = []
mainF1R = []
for a in range(10000):
    user = users[a]
    recommendP,user_row = obtain_popular_recommendations(sparse_user_item, fics, user, K, sorted_items)
    recommendR = obtain_random_recommendations(fics,user,K)
    read_by_user = set(np.where(user_row >= 1.0)[0])
    if len(read_by_user) != 0:
        recallP,precisionP,F1P = calculate_metrics(recommendP,read_by_user,K)
        recallR,precisionR,F1R = calculate_metrics(recommendR,read_by_user,K)
        mainRecallP.append(recallP)
        mainRecallR.append(recallR)
        mainPrecP.append(precisionP)
        mainPrecR.append(precisionR)
        mainF1P.append(F1P)
        mainF1R.append(F1R)
    if a % 200 == 0:
        fP = np.average(mainF1P)
        fR = np.average(mainF1R)
        print(a,fP,fR)
print("Popular:",np.average(mainRecallP),np.average(mainPrecP),np.average(mainF1P))
print("Random:",np.average(mainRecallR),np.average(mainPrecR),np.average(mainF1R))
