import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from collections import Counter
from scipy.sparse import coo_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn import svm
import operator
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from flask import Flask, render_template, json, request
from sklearn.externals import joblib
import re


#os.chdir("D:\PA\DSM\POC24Nov\predictions\SIM1")

#INPUT_PATH = "D_Train.csv"
os.chdir('/home/prudhvi/Downloads/PickleMOdel')
cols = [1,2,3,4,5,10]

columns = ['ArrivalStatus', 'BlockReason', 'BlockStatus', 'ChannelId','CriticalVendor','VendorName']

train_data = read_data('D_Train.csv')
test_data = read_data('D_Test_final.csv')
    
def read_data(path):
    """
    Read the data into dataframe
    :param path:
    :return:
    """
    col = ['System', 'Region', 'DpDocumentNumber', 'Companycode', 'CurrencyLC',
           'AmountLC', 'VendorNumber','VendorName','CriticalVendor', 'ChannelId',
           'BlockReason', 'BlockStatus','PTDays', 'InvtoCrtnDt','OverduePayment',
           'InvoiceDate','DpCreationDate', 'FiDueDate', 'ClearingDate', 'ArrivalStatus',
           'InvoiceStatus']
    data = pd.read_csv(path, usecols=col, index_col='DpDocumentNumber',
                       parse_dates=True, dayfirst=False)
    data['DpCreationDate'] = pd.to_datetime(data.DpCreationDate, format='%m/%d/%Y')
    data['ClearingDate'] = pd.to_datetime(data.ClearingDate, format='%m/%d/%Y')
    data['FiDueDate'] = pd.to_datetime(data.FiDueDate, format='%m/%d/%Y')
    return data

def getEncoded(colName):
    """
    encode label
    """
    lbl_enc = preprocessing.LabelEncoder()
    return lbl_enc.fit_transform(colName)

def OneHot_Encoder(df, cols):
    enc = OneHotEncoder(sparse=True)
    encoded_data = enc.fit_transform(df.iloc[:, cols].as_matrix())
    # print encoded_data.shape
    # print type(encoded_data)
    # encoded_date = pd.DataFrame(encoded_data.todense())
    """ Feature names code """
    features = []
    for i, name in enumerate(columns):
        l = len(Counter(df[name]))
        for k in range(l):
            f_nm = str(name) + '_' + str(k + 1)
            features.append(f_nm)
    return encoded_data, features

def preprocess_data(data) :
    for column in data:
        if (data[column].isnull().sum()) > 0:
            if column in ['VendorName', 'ChannelId', 'BlockReason', 'InvoiceStatus', 'Arrival Status',
                          'CriticalVendor']:
                data[column].fillna('Missing', inplace=True)
            if column in ['AmountLC']:
                data[column].fillna(0, inplace=True)
                data[column] = data[column].replace(np.NaN, 0)

    data.drop(['System', 'Region', 'Companycode', 'CurrencyLC', 'VendorNumber', 'InvoiceDate', 'DpCreationDate', 'FiDueDate',
         'ClearingDate'], axis=1, inplace=True)

    data["VendorName"] = data["VendorName"].astype('category')
    data["CriticalVendor"] = data["CriticalVendor"].astype('category')

    data["ChannelId"] = data["ChannelId"].astype('category')
    data["ArrivalStatus"] = data["ArrivalStatus"].astype('category')

    data["InvoiceStatus"] = data["InvoiceStatus"].astype('category')

    lbl_enc = preprocessing.LabelEncoder()
    data['ChannelId'] = lbl_enc.fit_transform(data['ChannelId'].astype(str))
    data['CriticalVendor'] = lbl_enc.fit_transform(data['CriticalVendor'].astype(str))
    data['BlockReason'] = lbl_enc.fit_transform(data['BlockReason'].astype(str))
    data['BlockStatus'] = lbl_enc.fit_transform(data['BlockStatus'].astype(str))
    data['VendorName'] = lbl_enc.fit_transform(data['VendorName'].astype(str))
    data['ArrivalStatus'] = lbl_enc.fit_transform(data['ArrivalStatus'].astype(str))
    data['InvoiceStatus'] = lbl_enc.fit_transform(data['InvoiceStatus'].astype(str))
    return data

def return_originalpreds(data) :
    data.iloc[:,[0,7,8,9]] = preprocessing.scale(data.iloc[:,[0,7,8,9]])
    cols = [1,2,3,4,5,10]
    encoded_data, feature_names = OneHot_Encoder(data, cols)
    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))

    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data, data.OverduePayment, shuffle=False,stratify=None, train_size=359851, test_size=4985)
    # load Linear & log pickle
    model_linear = joblib.load('LinearModel.pkl')
    model_log = joblib.load('LogModel.pkl')

    overdue_test = model_linear.predict(X_test)
    data['OverduePayment'][-4985:] = overdue_test
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))
    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data, data.OverduePayment, shuffle=False,stratify=None, train_size=359851, test_size=4985)
    #preds = model_rfc.predict(X_test)
    original_preds = model_log.predict(X_test)
    
    return original_preds

def return_preds(data, original_preds) :
    data.iloc[:,[0,7,8,9]] = preprocessing.scale(data.iloc[:,[0,7,8,9]])
    cols = [1,2,3,4,5,10]
    encoded_data, feature_names = OneHot_Encoder(data, cols)
    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))

    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data, data.OverduePayment, shuffle=False,stratify=None, train_size=359038, test_size=4985)
    # load Linear & log pickle
    model_linear = joblib.load('LinearModel.pkl')
    model_log = joblib.load('LogModel.pkl')

    overdue_test = model_linear.predict(X_test)
    data['OverduePayment'][-4985:] = overdue_test
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))
    X_train, X_test, y_train, y_test = train_test_split(encoded_data, data.InvoiceStatus, shuffle=False,stratify=None, train_size=359038, test_size=4985)
    #preds = model_rfc.predict(X_test)
    new_preds = model_log.predict(X_test)
    mat= confusion_matrix(original_preds,new_preds)
    op1 =  sum(mat[0])
    op2 =   mat[0][0]

    return op1,op2

def update_inv_crt(row,low_limit,upper_limit,change) :
    if row['InvtoCrtnDt'] > low_limit and row['InvtoCrtnDt'] < upper_limit :
        row['InvtoCrtnDt'] = row['InvtoCrtnDt'] - change
        if row['InvtoCrtnDt'] < row['PTDays'] :
            row['ArrivalStatus'] = 'Before'
    return row

def update_pt(row,low_limit,upper_limit,change) :
    if row['PTDays'] > low_limit and row['PTDays'] < upper_limit and row['VendorNumber'] in top_short_vendors :
        row['PTDays'] =row['PTDays'] + change
        if row['InvtoCrtnDt'] < row['PTDays'] :
            row['ArrivalStatus'] = 'Before'
    return row


def run_sim(pt_change,inv_change) :
    data = pd.concat([train_data,test_data])
    data = preprocess_data(data)
    df = pd.read_csv('log_reg_preds.csv')
    originalpreds = df['log_reg_preds']
    #originalpreds = return_originalpreds(data)
    new_test = test_data.apply(lambda x: update_pt(x, -259, 15, pt_change), axis=1)
    new_test = new_test.apply(lambda x: update_inv_crt(x,4,95,inv_change), axis=1)    
    data = pd.concat([train_data,new_test])
    data = preprocess_data(data)
    o1 , o2  = return_preds(data, originalpreds)
    return o1,o2



def main():

    input_first = 25
    input_second = 0
    first,second = run_sim(input_first, input_second)
    print "Old delays count is ",first," and percentage is " , round((first/4985)*100,2)
    print "New delays count is" , second , "percentage is " , round(((second/4985)*100),2)

    
counts = [ ]
ven = [10,50,100,200,300]
for i in ven :
    input_second = 0
    top_short_vendors = list(vendor_df.VendorNumber)[:i]
    first, second = run_sim(30, input_second)
    counts.append(second)
    print second ,"New delays "
    print i , 'Done'


pt = [20,25,30]
inv = [10,15,20]
counts = [ ]

for p in pt :
    for i in inv :
        first, second = run_sim(p, i)
        counts.append(second)
        print "For ",p ,"days increase in PT", i ,'Days decrease in Inv , new delays are', second


top_short_vendors = [0,2559681,2572367,2582155,2601916,2580994,2581897,2567232,2591814,'VCH0118230']

top_short_vendors = [str(i) for i in top_short_vendors]