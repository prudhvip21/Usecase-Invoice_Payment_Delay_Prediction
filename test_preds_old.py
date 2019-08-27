import os
import io
import pandas as pd
import pydot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from collections import Counter
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from pandas import ExcelWriter
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy.stats
from scipy.stats import spearmanr, kendalltau, pearsonr  # for chi-square
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, hstack
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from prettytable import PrettyTable
#from rpy2.robjects import r, pandas2ri
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pickle

from itertools import izip

import operator

os.chdir('/home/prudhvi/Documents/DSM')

scaler = preprocessing.MinMaxScaler()

cols = [0,1,3,5,8,9,10]
cols_all = [0,1,3,5,6,7,8]

columns = ['System', 'Region', 'Companycode',
           'Vendor Name', 'Critical vendor',
           'ChannelId', 'Block Status']


#data = pd.read_csv('TrainTest_DNP.csv')
data = pd.read_csv('PPD Template V_3.1.csv')
data = data[data['BG Name'] == 'DNP']

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def getEncoded(colName):
    lbl_enc = preprocessing.LabelEncoder()
    return lbl_enc.fit_transform(colName)


def OneHot_Encoder(df, cols):
    enc = OneHotEncoder(sparse=True)
    encoded_data = enc.fit_transform(df.iloc[:, cols].as_matrix())

    """ Feature names code
    features = []
    for i, name in enumerate(columns):
        l = len(Counter(df[name]))
        for k in range(l):
            f_nm = str(name) + '_' + str(k + 1)
            features.append(f_nm)

    '''for i, name in enumerate(features):
        encoded_data.rename(columns={i:name},inplace=True)'''
    encoded_data.col = features
    
     """
    return encoded_data






#data.drop_duplicates('DpDocumentNumber', keep='first', inplace=True)

for column in data:
    if (data[column].isnull().sum()) > 0:
        if column in ['Vendor Name', 'ChannelId','POInvoiceType', 'Block Status','Critical vendor', 'Region','Companycode', 'System' , 'PO Item number']:
            data[column].fillna('Missing', inplace=True)
        if column in ['AmountLC', 'DueDtToInvDt', 'InvDtToCrtnDt', 'FiDueDate','ProcessingTime','BlockTime','ClearTime']:
            data[column].fillna(0, inplace=True)
            data[column] = data[column].replace(np.NaN, 0)

# data.isnull().any() data.columns data.head(5)

train = data
train = train.drop(['BG Name', 'FiDueDate', 'CurrencyLC','InvoiceDate', 'DpCreationDate'], 1)

# train.shape, train.columns
del data

quant_df = train.quantile([0.25, 0.5, 0.75, 0.99])
quant_df.loc[4, :] = quant_df.apply(lambda x: x[0.75] - x[0.25], axis=0)
quant_df.loc[5, :] = quant_df.apply(lambda x: x[0.5] + 3* x[4], axis=0)
quant_df.loc[6, :] = quant_df.apply(lambda x: x[0.5] - 3 * x[4], axis=0)
train = train.loc[train['DueDtToInvDt'] < quant_df.iloc[5]['DueDtToInvDt']]
train = train.loc[train['DueDtToInvDt'] > quant_df.iloc[6]['DueDtToInvDt']]
train = train.loc[train['InvDtToCrtnDt'] < quant_df.iloc[5]['InvDtToCrtnDt']]
train = train.loc[train['InvDtToCrtnDt'] > quant_df.iloc[6]['InvDtToCrtnDt']]

#df = preprocessing.scale(train.iloc[:, [4,12,13,14,15,16]])
train.iloc[:, [4,11,12,13,14,15]] = preprocessing.scale(train.iloc[:, [4,11,12,13,14,15]])

#train.iloc[:, [4,9,10]] = preprocessing.scale(train.iloc[:, [4,9,10]])

del quant_df
# change data type as throwing error train.shape

train["Companycode"] = train["Companycode"].astype('category')
train["Vendor Name"] = train["Vendor Name"].astype('category')
# train["POInvoiceType"] = train["POInvoiceType"].astype('category')
# train["POItemNo"] = train["POItemNo"].astype('category')
train["ChannelId"] = train["ChannelId"].astype('category')
train['Companycode'] = pd.to_numeric(train['Companycode'], errors='ignore')
train['Companycode'].fillna(0, inplace=True)
# train.dtypes.head(20)  train.head(10)
lbl_enc = preprocessing.LabelEncoder()
train['Companycode'] = lbl_enc.fit_transform(train['Companycode'].astype(str))
train['Critical vendor'] = lbl_enc.fit_transform(train['Critical vendor'].astype(str))
train['InvStatus'] = lbl_enc.fit_transform(train['InvStatus'].astype(str))
# perform label encoding
train['System'] = getEncoded(train['System'])
train['Region'] = getEncoded(train['Region'])
# train['Companycode'] = getEncoded(train['Companycode'])#*
train['Vendor Name'] = getEncoded(train['Vendor Name'])
# train['Critical vendor'] = getEncoded(train['Critical vendor'])#*
train['ChannelId'] = getEncoded(train['ChannelId'])
train['Block Status'] = getEncoded(train['Block Status'])
# train['InvStatus'] = getEncoded(train['InvStatus'])#*

train['Block Reason'].replace(to_replace = ['Manual block','Blocked Header Workflow','Blocked Header '], value = ['Manual Block','Blocked Header','Blocked Header'] , inplace = True)

# train_scaled = pd.DataFrame(scaler.fit_transform(train))
train_scaled = train
#train_scaled[train_scaled['DpDocumentNumber'] == 556588]
ind = list(train_scaled.index)
k = ind.index(359852)
index = range(0, len(train_scaled))
ind_train = index[:k-2]
ind_test = ind[k:]

train_final = train_scaled.ix[ind_train]
test_final = train_scaled.ix[ind_test]

delays = train_final[train_final['InvStatus']==0]

delays_late = delays[delays['Arrival']=='LATE']

kk = train_final.groupby(['InvStatus','Arrival'])['Arrival'].agg(['count'])

sample_minus = sample[sample['Arrival'] == 'LATE']

sample['InvDtToCrtnDt'].describe()


sample_minus_test = test_final[test_final['InvDtToCrtnDt'] < 0]


np.mean(sample.InvDtToCrtnDt)

np.mean(train_final.InvDtToCrtnDt)

np.mean(test_final.InvDtToCrtnDt)





cl = [5]
# where test starts
for item in cl :
    test_final = test_final[test_final.iloc[:,item].isin(list(train_final.iloc[:,item].unique()))]

len(test_final)
train_scaled = pd.concat([train_final, test_final])
cols_all = [0,1,3,5,8,9,11]
encoded_data = OneHot_Encoder(train_scaled, cols_all)
enc_cols = encoded_data.col
enc_cols += ['AmountLC', 'DueDtToInvDt', 'InvDtToCrtnDt']
# encoded_data.shape
# encoded_data['AmountLC'] = train_scaled['AmountLC']
encoded_data = hstack((encoded_data, np.array(train_scaled['AmountLC'])[:, None]))
encoded_data = hstack((encoded_data, np.array(train_scaled['DueDtToInvDt'])[:, None]))
encoded_data = hstack((encoded_data, np.array(train_scaled['InvDtToCrtnDt'])[:, None]))

#index = list(train_scaled.index)
enc_index = range(0, len(train_scaled))
#ind = list(train_scaled.index)
#ind_train = ind[:543297]
enc_ind_test = enc_index[-len(test_final):]
#train_index = index[0:-8676]
#test_index = index[-7247:]
X_test = encoded_data.tocsr()[enc_ind_test,:]

encoded_data = OneHot_Encoder(train_final, cols_all)
enc_cols = encoded_data.col
enc_cols += ['AmountLC', 'DueDtToInvDt', 'InvDtToCrtnDt']
encoded_data = hstack((encoded_data, np.array(train_final['AmountLC'])[:, None]))
encoded_data = hstack((encoded_data, np.array(train_final['DueDtToInvDt'])[:, None]))
encoded_data = hstack((encoded_data, np.array(train_final['InvDtToCrtnDt'])[:, None]))

X_train = encoded_data
y_train = train_scaled.iloc[:, 20]
#y_test = train_scaled_all.iloc[:, 11][test_index]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

y_train = getEncoded(y_train)


""" Classification Model """
# Spot Check Algorithms
# Define Support Vector Machine
model_svc = svm.SVC(kernel='linear', gamma=1, max_iter=10000)
model_svc.fit(X_train, y_train)
pred_svc = model_svc.predict(X_test)
test_final['preds'] = pred_svc
test_final.to_csv('preds.csv')
print(confusion_matrix(y_test, pred_svc))
print(accuracy_score(y_test, pred_svc))

feature_imps = list(model_svc.coef_.toarray()[0, :])
features = {}
for item in enc_cols:
    f = str(item).split('_')[0]
    if f in features.keys():
        features[f] = round((features[f] + feature_imps[enc_cols.index(item)]), 4)
    else:
        features[f] = round(feature_imps[enc_cols.index(item)], 4)

print(sorted(features.items(), key=operator.itemgetter(1), reverse=True))




