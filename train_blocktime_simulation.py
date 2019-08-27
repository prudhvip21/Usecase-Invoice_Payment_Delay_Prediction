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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy import stats

# returns current working directory
# os.getcwd()
# changes working directory
# os.chdir("D:\PA\DSM\POC24Nov\predictions\SIM1")

os.chdir("/home/prudhvi/Documents/DSM/data")

# INPUT_PATH = "D_Train.csv"

#cols = [1, 2, 3, 5, 6, 11]

columns = ['ArrivalStatus', 'BlockReason', 'BlockStatus', 'ChannelId', 'CriticalVendor', 'VendorName']


# 11, 'PaymentTerm',
def read_data(path):
    """
    Read the data into dataframe
    :param path:
    :return:
    """
    col = ['System', 'Region', 'DpDocumentNumber', 'Companycode', 'CurrencyLC',
           'AmountLC', 'VendorNumber', 'VendorName', 'CriticalVendor', 'ChannelId',
           'BlockReason', 'BlockStatus', 'PTDays', 'InvtoCrtnDt', 'OverduePayment',
           'InvoiceDate', 'DpCreationDate', 'FiDueDate', 'ClearingDate',
           'ArrivalStatus', 'InvoiceStatus', 'BlockTime']
    data = pd.read_csv(path, usecols=col, index_col='DpDocumentNumber',
                       parse_dates=True, dayfirst=False)
    data['InvoiceDate'] = pd.to_datetime(data.InvoiceDate, format='%m/%d/%Y')
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



def preprocess_data(dataset):

    # impute missing vaues
    for column in dataset:
        if (dataset[column].isnull().sum()) > 0:
            if column in ['VendorName', 'ChannelId', 'BlockReason', 'InvoiceStatus', 'Arrival Status',
                          'CriticalVendor']:
                dataset[column].fillna('Missing', inplace=True)
            if column in ['AmountLC', 'BlockTime']:
                dataset[column].fillna(0, inplace=True)
                dataset[column] = dataset[column].replace(np.NaN, 0)

    dataset.drop(
        ['System', 'Region', 'Companycode', 'CurrencyLC', 'VendorNumber', 'InvoiceDate', 'DpCreationDate', 'FiDueDate',
         'ClearingDate'], axis=1, inplace=True)

    dataset["VendorName"] = dataset["VendorName"].astype('category')
    dataset["CriticalVendor"] = dataset["CriticalVendor"].astype('category')
    # dataset["Block Reason"] = dataset["Block Reason"].astype('category')
    dataset["ChannelId"] = dataset["ChannelId"].astype('category')
    dataset["ArrivalStatus"] = dataset["ArrivalStatus"].astype('category')
    dataset["InvoiceStatus"] = dataset["InvoiceStatus"].astype('category')

    lbl_enc = preprocessing.LabelEncoder()
    dataset['ChannelId'] = lbl_enc.fit_transform(dataset['ChannelId'].astype(str))
    dataset['CriticalVendor'] = lbl_enc.fit_transform(dataset['CriticalVendor'].astype(str))
    dataset['BlockReason'] = lbl_enc.fit_transform(dataset['BlockReason'].astype(str))
    dataset['BlockStatus'] = lbl_enc.fit_transform(dataset['BlockStatus'].astype(str))
    dataset['VendorName'] = lbl_enc.fit_transform(dataset['VendorName'].astype(str))
    dataset['ArrivalStatus'] = lbl_enc.fit_transform(dataset['ArrivalStatus'].astype(str))
    dataset['InvoiceStatus'] = lbl_enc.fit_transform(dataset['InvoiceStatus'].astype(str))

    return dataset


def return_preds(pred_data):
    pred_data.iloc[:, [0, 8, 11]] = preprocessing.scale(pred_data.iloc[:, [0, 8, 11]])
    cols = [1, 2, 3, 4, 5, 9]
    #cols = [1, 2, 3,9]

    encoded_data, feature_names = OneHot_Encoder(pred_data, cols)
    encoded_data = hstack((encoded_data, np.array(pred_data['AmountLC'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(pred_data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(pred_data['InvtoCrtnDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(pred_data['OverduePayment'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(pred_data['BlockTime'])[:, None]))

    X_train, X_test, y_train, y_test = train_test_split(encoded_data, pred_data.InvoiceStatus, shuffle=False,stratify=None, train_size=351998,test_size=7867)

    #model_logreg2 = LogisticRegression(penalty='l1')
    #model_logreg2.fit(X_train,y_train)
    preds = model_logreg2.predict(X_test)
    #print  accuracy_score(y_test,preds)
    return preds


def update_inv_crt(row, low_limit, upper_limit, change):
    if row['InvtoCrtnDt'] > low_limit and row['InvtoCrtnDt'] < upper_limit :
        row['InvtoCrtnDt'] = row['InvtoCrtnDt'] - change
        if row['InvtoCrtnDt'] < row['PTDays']:
            row['ArrivalStatus'] = 'Before'
    return row


def update_pt(row, low_limit, upper_limit, change):
    if row['PTDays'] > low_limit and row['PTDays'] < upper_limit :
        row['PTDays'] = row['PTDays'] + change
        if row['InvtoCrtnDt'] < row['PTDays']:
            row['ArrivalStatus'] = 'Before'
    return row

def update_bt(row, low_limit, upper_limit, change):
    if row['BlockTime'] > low_limit and row['BlockTime'] < upper_limit :
        if row['BlockTime'] - change < 0 :
            row['BlockTime'] = 0
        else :
            row['BlockTime'] = row['BlockTime'] - change
    return row


def run_sim(pt_change,inv_change,bt_change):
    #train_data = read_data('D_Train_BT.csv')
    #start = pd.Timestamp('2017-08-31')
    #end = pd.Timestamp('2017-11-30')
    #train_data = train_data.loc[(train_data.InvoiceDate <= start) | (train_data.InvoiceDate >= end)]
    #test_data = read_data('D_test_BT_train.csv')
    new_test = test_data.apply(lambda x: update_pt(x,-294,15, pt_change), axis=1)
    new_test = new_test.apply(lambda x: update_inv_crt(x, -296, 95, inv_change), axis=1)
    #print np.mean(new_test.InvtoCrtnDt)
    new_test = new_test.apply(lambda x: update_bt(x, 0, 50, bt_change), axis=1)
    dt = pd.concat([train_data,new_test])
    processed_data = preprocess_data(dt)
    preds = return_preds(processed_data)
    print Counter(preds)
    mat = confusion_matrix(originalpreds,preds)
    op1 = sum(mat[0])
    op2 = mat[0][0]
    d = dict()
    change = [ ]
    if pt_change != 0 :
        s = 'Payment Terms (Short) by ' + str(pt_change) + 'days'
        change.append(s)

    if inv_change !=0 :
        s='Late Arrivals changed by ' + str(inv_change) + ' days'
        change.append(s)
    if bt_change != 0 :
        s = 'Block Time changed by ' + str(bt_change) + ' days'
        change.append(s)

    final[str(change)] = preds
    d['Change applied on'] = change
    d['New_Delays'] = op2
    tt = test_data
    tt['new_preds'] = preds
    tt['orig_preds'] = originalpreds
    avg_df = tt[tt['orig_preds']==0]
    tt = tt[(tt['orig_preds']==0) & (tt['new_preds']==0)]
    d['Short_Payment_terms'] = len(tt[tt['PTDays']<15])
    d['Late_Arrivals'] = len(tt[tt['InvtoCrtnDt'] >= tt['PTDays']])
    d['Blocks'] = len(tt[tt['BlockStatus']=='Block'])

    #avgs = {}
    #avgs['changes'] = change
    #avgs['Inv'] = np.mean(avg_df.InvtoCrtnDt)
    #avgs['PT'] = np.mean(avg_df.PTDays)
    del test_data['new_preds']
    del test_data['orig_preds']
    return d

counts = [ ]
bt = range(0,60,5)
for i in bt :
    new_test = test_data.apply(lambda x: update_bt(x, 0, 90,i), axis=1)
    new_test['orig_preds'] = originalpreds
    new_test = new_test[new_test['orig_preds']==0]
    counts.append(np.mean(new_test.BlockTime))
    print np.mean(new_test.BlockTime)
    del new_test

pt_change = 0
inv_change = 0
bt_change = 0



pt_ch = range(0,30,5)
la_ch = range(0,35,5)

run_sim(pt_change,inv_change,bt_change)

for i in pt_ch :
    for j in la_ch :
        d = run_sim(i,j,0)

#result = pd.DataFrame()


ch = range(5,35,10)
for i in ch :
    c = run_sim(i,0,0)
    result = result.append(c,ignore_index = True)
    print i, 'Done'

ch = range(5,50,5)
for i in ch :
    c = run_sim(i,0,0)
    result = result.append(c,ignore_index = True)
    print i, 'Done'

ch = range(5,60,5)
for i in ch :
    c = run_sim(0,0,i)
    result = result.append(c,ignore_index = True)
    print i, 'Done'


result.to_csv('Block_time_simulation_train.csv')






final.to_csv('ss.csv')











"""   

def main():

    # impute missing vaues
    for column in data:
        if (data[column].isnull().sum()) > 0:
            if column in ['VendorName', 'ChannelId', 'BlockReason', 'InvoiceStatus', 'Arrival Status',
                          'CriticalVendor']:
                data[column].fillna('Missing', inplace=True)
            if column in ['AmountLC', 'BlockTime']:
                data[column].fillna(0, inplace=True)
                data[column] = data[column].replace(np.NaN, 0)

    data.drop(
        ['System', 'Region', 'Companycode', 'CurrencyLC', 'VendorNumber', 'InvoiceDate', 'DpCreationDate', 'FiDueDate',
         'ClearingDate'], axis=1, inplace=True)

    # data.iloc[:, [0,8,10]] = preprocessing.scale(data.iloc[:, [0,8,10]])
    data.iloc[:, [0, 8, 11]] = preprocessing.scale(data.iloc[:, [0, 8, 11]])

    data["VendorName"] = data["VendorName"].astype('category')
    data["CriticalVendor"] = data["CriticalVendor"].astype('category')
    # data["Block Reason"] = data["Block Reason"].astype('category')
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
    cols = [1, 2, 3, 4, 5, 9]

    encoded_data, feature_names = OneHot_Encoder(data, cols)
    encoded_data.shape


    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['BlockTime'])[:, None]))

    X_train, X_test, y_train, y_test = train_test_split(encoded_data, data.InvoiceStatus, shuffle=False,stratify=None, train_size=35981,test_size=7867)

    model_logreg = LogisticRegression(penalty='l1')
    model_logreg.fit(X_train, y_train)
    train_pred = model_logreg.predict(X_test)
    print(accuracy_score(y_train, train_pred))

    data = read_data('D_test_BT_train.csv')



if __name__ == "__main__":
    main()

"""