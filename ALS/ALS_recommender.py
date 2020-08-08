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
    
    
tableFileName = "table_user_item_rating.long.txt"
inputFile = "../dataset/metadata_long.txt"

#I'll give better rating to something the author has writen him/herself 
#as they are more likely to like it than anything they have read. 
#If I manage to get bookmarks I'll also rate them higher.

# ~ info = {}
# ~ with open(inputFile,"r") as infile:
    # ~ ficInfo = {}
    # ~ for line in infile:
        # ~ line = line.strip()
        # ~ dades = line.split("\t")
        # ~ author = dades[1]
        # ~ name = dades[2]
        # ~ ficId = dades[0]
        # ~ ficInfo[ficId] = [name,author,dades[3],dades[6],dades[11]]
        # ~ readers = dades[-1].split("|")
        # ~ if len(readers) > 0:
            # ~ if author not in info:
                # ~ info[author] = {}
            # ~ info[author][ficId] = ["2.0",0]
            # ~ for r in readers:
                # ~ if r not in info:
                    # ~ info[r] = {}
                # ~ info[r][ficId] = ["1.0",0]

# ~ #Separate "NEW" users from the rest and change IDs
# ~ conversion = {"users":{},"fics":{}}
# ~ with open("new_users.txt","w") as outNew, open(tableFileName, "w") as outTable:
    # ~ for userId,fics in info.items():
        # ~ if len(fics) < 10:
            # ~ for fic in fics:
                # ~ print(str(userId)+"\t"+fic+"\t"+info[userId][fic][0],file=outNew)
        # ~ else:
            # ~ conversion["users"][userId] = str(len(conversion["users"]))
            # ~ for fic in fics:
                # ~ if fic not in conversion["fics"]:
                    # ~ conversion["fics"][fic] = str(len(conversion["fics"]))
                # ~ print(conversion["users"][userId]+"\t"+conversion["fics"][fic]+"\t"+info[userId][fic][0],file=outTable)

# ~ #Print to file the code conversions for users and fics
# ~ with open("conversionUsers","w") as outUsers:
    # ~ for code in conversion["users"]:
        # ~ print(code+"\t"+conversion["users"][code],file=outUsers)

# ~ with open("conversionFics", "w") as outFics:
    # ~ for code in conversion["fics"]:
        # ~ print(code+"\t"+conversion["fics"][code],file=outFics)

#Define a sparse matrix from users2items and one for items2users
df_ratings = pd.read_csv(tableFileName,sep="\t",names=["userId","ficId","rating"])
users = df_ratings['userId'].unique()
fics = df_ratings['ficId'].unique()
sparse_user_item = sparse.csr_matrix((df_ratings['rating'].astype(float), (df_ratings['userId'].astype(int), df_ratings['ficId'].astype(int))), shape=(len(users),len(fics)))

sparse_item_user = sparse.csr_matrix((df_ratings["rating"].astype(float), (df_ratings["ficId"].astype(int), df_ratings["userId"].astype(int))), shape=(len(fics),len(users)))
print("IU",sparse_item_user.shape)
print("UI",sparse_user_item.shape)


#Calculate sparsity
# ~ matrix_size = sparse_user_item.shape[0]*sparse_user_item.shape[1] # Number of possible interactions in the matrix
# ~ num_reads = len(sparse_user_item.nonzero()[0]) # Number of items interacted with
# ~ sparsity = 100*(1 - (num_reads/matrix_size))
# ~ print(sparsity)


validation, test = implicit.evaluation.train_test_split(sparse_item_user,train_percentage=0.85)
train, validation = implicit.evaluation.train_test_split(validation,train_percentage=0.85)


# ~ alphas = [x for x in range(10,100,10)]
# ~ factors = [x for x in range(10,500,20)]
# ~ regularization = [0.001,0.01,0.1,1.0,10.0]
# ~ iterations = [x for x in range(10, 100,10)]
# ~ Ks = [x for x in range(10,100,5)]

# ~ with open("metrics.txt","w") as outfile:
    # ~ for alpha in alphas:
        # ~ for factor in factors:
            # ~ for reg in regularization:
                # ~ for ite in iterations:
                    # ~ print("Running:",alpha, factor,reg,ite)
                    # ~ model = implicit.als.AlternatingLeastSquares(factors=factor,regularization=reg,iterations=ite)
                    # ~ model.fit(train*alpha)
                    # ~ for K in Ks:
                        # ~ info = implicit.evaluation.ranking_metrics_at_k(model,train.T.tocsr(),test.T.tocsr(),K=K,num_threads = 7)
                        # ~ print(alpha,factor,reg,ite,K,info["precision"],info["map"],info["ndcg"],info["auc"])
                        # ~ print(alpha,factor,reg,ite,K,info["precision"],info["map"],info["ndcg"],info["auc"],file=outfile)

# ~ #Fit the model
alpha = 40

model = implicit.als.AlternatingLeastSquares(factors=100,regularization=0.01,iterations=10)
model.fit(train*alpha)


K=500

mainRecallM = []
mainRecallP = []
mainRecallR = []
mainPrecM = []
mainPrecP = []
mainPrecR = []
mainF1M = []


samplesR = []
samplesP = []
samplesF = []
recomendations = model.recommend_all(train.T, K)
for a in range(10000):
    user = users[a]
    recommendM = recomendations[user]
    user_row = validation.T[user].toarray().reshape(-1)
    read_by_user = set(np.where(user_row >= 1.0)[0])
    if len(read_by_user) != 0:
        recallM,precisionM,F1M = calculate_metrics(recommendM,read_by_user,K)
        mainRecallM.append(recallM)
        mainPrecM.append(precisionM)
        mainF1M.append(F1M)
    if a % 200 == 0:
        m = np.average(mainRecallM)
        n = np.average(mainPrecM)
        f = np.average(mainF1M)
        samplesR.append(m)
        samplesP.append(n)
        samplesF.append(f)
        print(a,m,n,f)

print("Model:",np.average(mainRecallM),np.average(mainPrecM),np.average(mainF1M))

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

#Make recommendations
# ~ user_id = 21964
# ~ recommended = model.recommend(user_id, sparse_user_item)
# ~ for rec,score in recommended:
    # ~ print(rec,"\t".join(ficInfo[str(rec)]),score)

# ~ #Save model and matrices into pickle files
# ~ filename = 'modelALS_implicit.50.sav'
# ~ pickle.dump(model, open(filename, 'wb'))

# ~ filename = 'sparse_user_item_matrix.pickle'
# ~ pickle.dump(sparse_user_item, open(filename, 'wb'))

# ~ filename = 'sparse_item_user_matrix.pickle'
# ~ pickle.dump(sparse_item_user, open(filename, 'wb'))



# ~ filename = 'modelALS_implicit.10.sav'
# ~ pickle.dump(model, open(filename, 'wb'))
# ~ model = pickle.load(open(filename, 'rb'))
