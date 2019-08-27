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
import re
from sklearn.neural_network import MLPRegressor


cols = [1,2,3,4,5,9]

columns = ['VendorName', 'CriticalVendor', 'ChannelId','BlockReason','BlockStatus','ArrivalStatus']

train_data = read_data('D_Train.csv')
test_data = read_data('DN_Test_final.csv')


def read_data(path):
    """
    Read the data into dataframe
    :param path:
    :return:
    """
    col = ['System', 'Region', 'DpDocumentNumber', 'Companycode', 'CurrencyLC','AmountLC', 'VendorNumber','VendorName','CriticalVendor', 'ChannelId', 'BlockReason', 'BlockStatus','PTDays', 'InvtoCrtnDt','OverduePayment', 'InvoiceDate','DpCreationDate', 'FiDueDate', 'ClearingDate', 'ArrivalStatus','InvoiceStatus']
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

def return_preds(data) :
    data.iloc[:, [0,7,9]] = preprocessing.scale(data.iloc[:, [0,7,9]])
    cols = [1,2,3,4,5,10]
    encoded_data, feature_names = OneHot_Encoder(data, cols)
    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))

    """ 
    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data, data.OverduePayment, shuffle=False,stratify=None, train_size=359851, test_size=5538)
    overdue_test = regr.predict(X_test)
    data['OverduePayment'][-5538:] = overdue_test
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))
    """

    X_train, X_test, y_train, y_test = train_test_split(encoded_data, data.InvoiceStatus, shuffle=False,stratify=None, train_size=359851, test_size=5538)

    preds = model_logreg.predict(X_test)

    """ 
    mat= confusion_matrix(logreg_preds,preds)
    op1 =  "Old delays count is",sum(mat[0]),"and percentage is " , round((sum(mat[0])/4985)*100,2)
    op2 =  mat[0][0]
    """

    return preds

def update_inv_crt(row,low_limit,upper_limit,change) :
    if row['InvtoCrtnDt'] > low_limit and row['InvtoCrtnDt'] < upper_limit :
        row['InvtoCrtnDt'] = row['InvtoCrtnDt'] - change
        if row['InvtoCrtnDt'] < row['PTDays'] :
            row['ArrivalStatus'] = 'Before'
    return row

def update_pt(row,low_limit,upper_limit,change) :
    if row['PTDays'] > low_limit and row['PTDays'] < upper_limit :
        row['PTDays'] =row['PTDays'] + change
        if row['InvtoCrtnDt'] < row['PTDays'] :
            row['ArrivalStatus'] = 'Before'
    return row


def run_sim(pt_change,inv_change) :
    new_test = test_data.apply(lambda x: update_pt(x, -259, 30, pt_change), axis=1)
    new_test = new_test.apply(lambda x: update_inv_crt(x,4,95,inv_change), axis=1)
    data = pd.concat([train_data,new_test])
    data = preprocess_data(data)
    ds  = return_preds(data)
    mat = confusion_matrix(originalpreds, ds)
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

    final[str(change)] = ds
    d['Change applied on'] = change
    d['New_Delays'] = op2
    tt = test_data
    tt['new_preds'] = ds
    tt['orig_preds'] = originalpreds
    avg_df = tt[tt['orig_preds']==0]
    tt = tt[(tt['orig_preds']==0) & (tt['new_preds']==0)]
    d['Short_Payment_terms'] = len(tt[tt['PTDays']<15])
    d['Late_Arrivals'] = len(tt[tt['InvtoCrtnDt'] >= tt['PTDays']])
    d['Blocks'] = len(tt[tt['BlockStatus']=='Block'])

    del test_data['new_preds']
    del test_data['orig_preds']
    return d

pt_change = 0
inv_change = 0


final = copy.deepcopy(test_data)

pt_ch = range(0,30,5)
la_ch = range(0,35,5)

for i in pt_ch :
    for j in la_ch :
        d = run_sim(i,j)
        print  j , 'done in' , i


result = pd.DataFrame()

ch = range(1,31)
ch = range(5,50,5)
for i in ch :
    c = run_sim(i,0)
    result = result.append(c,ignore_index = True)
    print i, 'Done'



result.to_csv('tes_sim_feb_11.csv')





app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/delay_check', methods=['POST'])
def delay_check():
    input_first = request.form.get('input_1') #Input from ui
    input_second = request.form.get('input_2') #Input from ui
    first_msg,second_msg = run_sim(input_first, input_second)
    first_msg_result = re.findall('\d+', first_msg) #the first_msg contains like 'old delay count is 34 and percentage is 45.06'
    second_msg_result = re.findall('\d+', second_msg) #the second_msg contains like 'old delay count is 20 and percentage is 15.06'
    old_delay_count = int(first_msg_result[0])
    old_percentage = int(first_msg_result[1]) + float(first_msg_result[2])/100
    new_delay_count = int(second_msg_result[0])
    new_percentage = int(second_msg_result[1]) + float(second_msg_result[2])/100
    print old_delay_count, old_percentage, new_delay_count, new_percentage
    return json.dumps({'old_delay_count':old_delay_count,'old_percentage':old_percentage,'new_delay_count':new_delay_count,'new_percentage':new_percentage})

if __name__ == "__main__":
    app.run(port=5002)


""" 
final_preds = [ ]
for i in range(len(rf_preds)):
    if rf_preds[i] == 0 :
        final_preds.append(preds[i])
    else :
        final_preds.append(rf_preds[i])

test_data['new_preds'] = final_preds

test_data.to_csv('pred_P60_I10.csv')
del test_data['new_preds']




"""


