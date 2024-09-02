# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import sklearn
import scipy
import pyspark
from pandas.api.types import CategoricalDtype
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

########################################################## MAIN RECOMMENDER
    
# main runner 
def main_recommender_als(df):
    # statistical and format preparation 
    df = main_preparation(df)
    # make recommender
    df, X, Y = main_recommender(df)
    # testing 
    df = main_testing(df)
    return df, X, Y 


########################################################### DATA PREPARATION


def main_preparation(df):
    # make list of products available, no repetition
    prods_available = products_available(df)
    # drop useless values 
    df = useful_info(df)
    # group by user_id and keep useful columns only 
    df = group_info(df)
    # create estimated ranking for each product x bought by user_id y
    df = ranking_creator(df)
    # expand matrix by user_id once ranking is calculated
    df = purchase_separator(df)

    return df
    

# merge dataframes
def data_frame_merge(df_new, df_old):
    frame = [df_new, df_old]
    df = pd.concat(frame)
    return df


# create list of unique products
def products_available(df):
    products = df['Products']
    products_single = list(set(products))
    return products_single
    

# eliminate data not containing userId, products
def useful_info(df):
    # df drops na values for user_id and billing city
    df.dropna(subset=['UserId', 'Products'], inplace=True)
    return df


# group information by UserId, place Products into list 
def group_info(df):
    df = df.groupby('UserId')['Products'].apply(list).reset_index().sort_values('UserId')
    return df


# rank products by frequency in total purchases bought, add column to df
def ranking_creator(df):
    # create column list for items and respective rankings  
    id_items = []
    id_ranks = []
    
    # loop through df, rank by # of appearences in total purchases 
    #   find ranking for items using dictionary, append to new list WITH 
    #   duplication in order to allow explode in function below 
    for i in range(0, df.shape[0]): 
                
        items = []
        item_ranks = []
        
        count_dict = Counter(df.iloc[i]['Products'])
        
        for item, count in count_dict.items():
            
            ranking = 1.0
            
            items.extend([item for i in range(count)])
            item_ranks.extend([ranking for i in range(count)])
            
        id_items.append(items)
        id_ranks.append(item_ranks)
    
    df['IdItems'] = id_items
    df['IdRanks '] = id_ranks

    return df


# drop initial Product list, expand df into individual rows that include each purchase and respective ranking
#   prepares data to make utility matrix
def purchase_separator(df):
    df = df[['UserId', 'IdItems', 'IdRanks']]
    df = df.explode(['IdItems', 'IdRanks'])
    
    df = df.reset_index(drop=True)
    
    return df


###################################################### RECOMMENDER PREPARATION

# main recommender function 
def main_recommender(df):
    # ALS data organizing  
    df, user_index, product_index, csr = data_processing(df)
    # implement cost minimization function
    X, Y = cost_min(df, csr, 25.0, iterations=10, lambda_val=0.1, features=10)
    # make function to find user recommendations 
    recommended = als_recommender(df, user_index, product_index, X, Y)

    return df, X, Y 


# randomly split dataframe for testing and training
def data_processing(df):
    # divide function into two 
    
    df_train = df.iloc[:2*df.shape[0]//3][:]
    df_test = df.iloc[2*df.shape[0]//3:][:].reset_index()
    
    # change index of df_train into numeric 
    user_id = list(df['UserId'].unique())
    product_id = list(df['IdItems'].unique())
    
    # create dimensions of new matrix 
    shape = (len(user_id), len(product_id))
    
    # make categorical variable for users and products (eliminates repetitions )
    user_cat = CategoricalDtype(categories=sorted(user_id), ordered=True)
    product_cat = CategoricalDtype(categories=sorted(product_id), ordered=True)
        
    # add values to categories 
    user_index = df['UserId'].astype(user_cat).cat.codes
    product_index = df['IdItems'].astype(product_cat).cat.codes
    
    # make coo and csr matrix (coo for use, csr for better memory usage)
    coo = sparse.coo_matrix((df['IdRanks'], (user_index, product_index)), shape=shape, dtype=float)
    csr = coo.tocsr()
        
    return df, user_index, product_index, csr


# implement ALS cost minimization function 
def cost_min(df, csr, alpha_val, iterations=10, lambda_val=0.1, features=10):
              
    # confidence
    confidence = csr * alpha_val
    
    # get size of user / items 
    user_size, product_size = csr.shape
    
    # Find X, Y Matrices
    X = sparse.csr_matrix(np.random.normal(size = (user_size, features)).astype(np.float64))
    Y = sparse.csr_matrix(np.random.normal(size = (product_size, features)).astype(np.float64))
    
    
    # Find Identity Matrices
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(product_size)
    
    I = sparse.eye(features)
    lI = lambda_val * I
        
    for i in range(iterations):
        
        # Transposes
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
        
        # Loop through users 
        for u in range(user_size):
            
            # Get the user row.
            u_row = confidence[u,:].toarray()

            # Calculate the preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0
            
            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I
        
            # Combine previous terms 
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)
            
        for i in range(product_size):
            
            # Get the user row.
            i_row = confidence[:,i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row[:].copy()
            p_i[p_i != 0] = 1.0
            

            # Calculate Ci and Ci - I
            CiI = sparse.diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)

            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y
        

# begin making user recommendation
def als_recommender(df, user_index, product_index, X, Y):
    recommended = []

    # select item and call item recommendation helper 
    item = "sample_product_id"
    item, top_10_products = item_recommendation(item, product_index, Y, df)
    
    print('For product:' + ' ' + item)
    print(top_10_products)
    
    
    # select user and call item recommendation helper 
    user = "sample_user_id"
    user, top_users = user_recommendation(user, product_index, user_index, X, Y, df)
    
    print('For user:' + ' ' + user)
    print(top_users)
    
    return recommended

# helper function that consumes item and produces recommendation by similarity measures
def item_recommendation(item, product_index, Y, df):
    
    # find desired item index by inputting user id 
    item_id = df[df['IdItems'] == item].index[0]
    item_id = product_index[item_id]
        
    # use item_id to query our X and Y matrices 
    item_pref = Y[item_id].T
    
    # calculate similarity between item queried and item matrix to calculate top 10 similar products
    scores = Y.dot(item_pref).toarray().reshape(1,-1)[0]
    top_10 = np.argsort(scores)[::-1][:10]
    
    # loop through top 10 products, query csr matrix and find their ids 
    top_10_products = []
    
    for prod in top_10:
        rec = df['IdItems'].iloc[product_index[product_index.values == prod].index[0]]
        top_10_products.append(rec)
        
    return item, top_10_products
        
        
# helper function that consumers user and produces recommendation by similarity measures
def user_recommendation(user, product_index, user_index, X, Y, df):
    
    # find desired user index by inputting user_id and querying x matrix 
    user_id = df[df['UserId'] == user].index[0]
    user_id = user_index[user_id]
    
    # use user_id to query our X and Y matrices
    user_pref = X[user_id]
    
    # calculate similarity scores    
    user_scores = user_pref.dot(Y.T).toarray()
    top_user_scores = np.argsort(user_scores)[::-1][:10][0]

    items = product_index.index[top_user_scores][:10]
    
    # retrieve actual item id from df using item indices
    top_users = []
    for user_id in items:

        rec = df['IdItems'].iloc[product_index[product_index.values == user_id].index[0]]
        top_users.append(rec)
    
    return user, top_users
    

    
########################################################### TODO: TESTING


if __name__ == '__main__':
    print('Please run from main file')


