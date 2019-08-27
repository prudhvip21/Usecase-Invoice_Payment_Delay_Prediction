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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import  Ridge
from sklearn.ensemble import RandomForestClassifier,\
VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve,auc
#returns current working directory
#os.getcwd()
#changes working directory
#os.chdir("C:\DSM")
#INPUT_PATH = "D_Train.csv"

cols = [1,2,3,4,5,9]

columns = ['VendorName', 'CriticalVendor', 'ChannelId','BlockReason','BlockStatus','ArrivalStatus']
    
# 11, 'PaymentTerm',
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

def main():
    #data = pd.read_csv('D_Test.csv')
    train_data = read_data('D_Train.csv')
    test_data = read_data('D_Test_final.csv')
    data = pd.concat([train_data,test_data])
    #data.columns #data.shape

    #impute missing vaues
    for column in data:
        if (data[column].isnull().sum()) >0:
            if column in ['VendorName', 'ChannelId', 'BlockReason','InvoiceStatus','Arrival Status', 'CriticalVendor']:
                data[column].fillna('Missing' ,inplace = True)
            if column in ['AmountLC']:
                data[column].fillna(0 ,inplace = True)
                data[column] = data[column].replace(np.NaN, 0)    
       
    data.drop(['System', 'Region', 'Companycode', 'CurrencyLC','VendorNumber','InvoiceDate','DpCreationDate', 'FiDueDate', 'ClearingDate'], axis =1, inplace=True)
     
    
    data.iloc[:,[0,6,7]] = preprocessing.scale(data.iloc[:,[0,6,7]])
    #change data type as throwing error train.shape


    data["VendorName"] = data["VendorName"].astype('category')
    data["CriticalVendor"] = data["CriticalVendor"].astype('category')
    #data["Block Reason"] = data["Block Reason"].astype('category')
    data["ChannelId"] = data["ChannelId"].astype('category')
    data["ArrivalStatus"] = data["ArrivalStatus"].astype('category')
    data["InvoiceStatus"] = data["InvoiceStatus"].astype('category')
   
    #perform label encoding    
    #perform label encoding 
    lbl_enc = preprocessing.LabelEncoder()
    data['ChannelId'] = lbl_enc.fit_transform(data['ChannelId'].astype(str))
    data['CriticalVendor'] = lbl_enc.fit_transform(data['CriticalVendor'].astype(str))
    data['BlockReason'] = lbl_enc.fit_transform(data['BlockReason'].astype(str))
    data['BlockStatus'] = lbl_enc.fit_transform(data['BlockStatus'].astype(str))
    data['VendorName'] = lbl_enc.fit_transform(data['VendorName'].astype(str))
    data['ArrivalStatus'] = lbl_enc.fit_transform(data['ArrivalStatus'].astype(str))
    data['InvoiceStatus'] = lbl_enc.fit_transform(data['InvoiceStatus'].astype(str))

   
    encoded_data,feature_names = OneHot_Encoder(data, cols)
    encoded_data.shape
    #enc_cols = encoded_data.col 
    #feature_names += ['AmountLC','PTDays','InvtoCrtnDt','OverduePayment']
    feature_names += ['AmountLC', 'PTDays', 'InvtoCrtnDt']
    feature_names += ['AmountLC', 'PTDays']
    #encoded_data.shape
    #encoded_data['AmountLC'] = train_scaled['AmountLC']
    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))
   
    
    X_train, X_test, y_train, y_test = train_test_split(encoded_data, data.InvoiceStatus,shuffle = False, stratify= None, train_size= 359851, test_size= 4985 )

    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data, data.OverduePayment, shuffle=False, stratify=None,train_size=359851, test_size=4985)
   

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
         
    
    """ Classification Model """  
   
    # Define Support Vector Machine
    model_svc = svm.SVC(kernel='rbf', gamma=1, max_iter=2000)
    model_svc.fit(X_train,y_train)
    train_pred = model_svc.predict(X_train)
    print(accuracy_score(y_train, train_pred))
    pred_svc = model_svc.predict(X_test)
    Counter(pred_svc)
    test_data['preds'] = pred_svc
    test_data.to_csv('test_predictions.csv')
    del test_data['preds']
    print(confusion_matrix(y_test,pred_svc))
    print(accuracy_score(y_test,pred_svc))
    print(confusion_matrix(y_test,pred_svc))


    """ Random Forest """

    model_rfc = RandomForestClassifier(random_state=1)
    model_rfc.fit(X_train, y_train)
    train_pred = model_rfc.predict(X_train)
    print(accuracy_score(y_train, train_pred))

    rf_preds = model_rfc.predict(X_test)
    Counter(rf_preds)
    test_data['rf_preds'] = rf_preds
    test_data.to_csv('rf_preds_overdue.csv')
    del test_data['rf_preds']



    """ regression prediction for Over time """

    regr2 = LinearSVR(random_state=0 ,max_iter= 10000 ,fit_intercept= False )
    model_lr = LinearRegression(fit_intercept=False)
    model_ridge = Ridge()

    regr2.fit(X_train,y_train_regr)
    regr2.score(X_train,y_train_regr)

    model_lr.fit(X_train,y_train_regr)
    model_lr.score(X_train, y_train_regr)
    overdue = model_lr.predict(X_test)
    test_data['better_overdue'] = overdue
    test_data.to_csv('improved_overdue.csv')

    model_ridge.fit(X_train,y_train_regr)
    model_ridge.score(X_train, y_train_regr)

    kk = regr.predict(X_test)
    kk= kk - 6
    delays = [ ]
    for i in kk :
        if i > 0 :
            delays.append(0)
        else :
            delays.append(2)
    Counter(delays)
    print(accuracy_score(y_train, delays))
    test_data['pred_overdue'] = kk
    test_data['regr_preds'] = delays
    test_data['rf_preds'] = rf_preds
    test_data.to_csv('regr_rf_predictions.csv')

    # putting predicted over due payment in data
    data['OverduePayment'][-4985:] = kk



    encoded_data,feature_names = OneHot_Encoder(data, cols)
    feature_names += ['AmountLC','PTDays','InvtoCrtnDt','OverduePayment']

    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))

    X_train, X_test, y_train, y_test = train_test_split(encoded_data, data.InvoiceStatus, shuffle=False, stratify=None,train_size=359851, test_size=4985)


    """ logistic regression """

    model_logreg = LogisticRegression(penalty= 'l1')
    model_logreg.fit(X_train, y_train)

    filename = 'finalized_logreg_model.sav'
    pickle.dump(model_logreg, open(filename, 'wb'))

    model_logreg = pickle.load(open(filename, 'rb'))
    train_pred = model_logreg.predict(X_train)
    train_probs = model_logreg.predict_proba(X_train)
    test_pred = model_logreg.predict(X_test)
    test_pred = model_logreg.predict_proba(X_test)

    predicted = cross_val_predict(model_logreg, X_train, y_train, cv=10)
    print accuracy_score(y_train, predicted)
    print metrics.classification_report(y_train, predicted)

    results = cross_val_score(model_logreg, X_train, y_train, cv=5, scoring='accuracy')
    print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
    delay_probs = []
    for item in train_probs :
        delay_probs.append(item[0])

    test_data['log_reg_probs'] = delay_probs
    test_data.to_csv('log_reg_probs.csv')

    print(accuracy_score(y_train, train_pred))

    fpr , tpr ,threshold = roc_curve(y_train,delay_probs ,pos_label= 0)
    print auc(fpr,tpr)



    logreg_preds = model_logreg.predict(X_test)
    Counter(logreg_preds)
    test_data['log_reg_preds'] = logreg_preds
    test_data.to_csv('log_reg_preds_overdue.csv')



    test_data.to_csv('log_reg_preds_overdue.csv')
    del test_data['log_reg_preds']




    #feature_imps = list(model_svc.coef_.toarray()[0,:])
    #feature_imps = list(model_rfc.feature_importances_)
    """ 
    feature_imps = list(model_logreg.coef_)
    features = {}
    for item in feature_names :
        f = str(item).split('_')[0]
        if f in features.keys() :
            features[f] = round((features[f] + feature_imps[feature_names.index(item)]),4)
        else :
            features[f] = round(feature_imps[feature_names.index(item)],4)

    for key in features :
        features[key] = features[key] * features[key]
    s = sum(features.values())
    for key in features:
        features[key] = round(((features[key]/s)* 100),4)

    print(sorted(features.items(), key=operator.itemgetter(1), reverse=True))
    """
    
                
if __name__ == "__main__":
    main()

""" 

test_data = pd.read_csv('D_Test.csv')

for column in test_data:
    if (test_data[column].isnull().sum()) > 0:
        if column in ['VendorName', 'ChannelId', 'BlockReason', 'InvoiceStatus', 'Arrival Status', 'CriticalVendor']:
            test_data[column].fillna('Missing', inplace=True)
        if column in ['AmountLC']:
            test_data[column].fillna(0, inplace=True)
            test_data[column] = test_data[column].replace(np.NaN, 0)

test_data.drop(
    ['System', 'Region', 'Companycode', 'CurrencyLC', 'VendorNumber','InvoiceDate',
     'DpCreationDate', 'FiDueDate', 'ClearingDate'], axis=1, inplace=True)

test_data = test_data.set_index('DpDocumentNumber')

test_data.iloc[:, [0,6,7,8]] = preprocessing.scale(test_data.iloc[:, [0,6,7,8]])
# change test_data type as throwing error train.shape


test_data["VendorName"] = test_data["VendorName"].astype('category')
test_data["CriticalVendor"] = test_data["CriticalVendor"].astype('category')
#test_data["Block Reason"] = test_data["Block Reason"].astype('category')
test_data["ChannelId"] = test_data["ChannelId"].astype('category')
test_data["Arrival Status"] = test_data["Arrival Status"].astype('category')
test_data["InvoiceStatus"] = test_data["InvoiceStatus"].astype('category')

# perform label encoding    
# perform label encoding 
lbl_enc = preprocessing.LabelEncoder()
test_data['ChannelId'] = lbl_enc.fit_transform(test_data['ChannelId'].astype(str))
test_data['CriticalVendor'] = lbl_enc.fit_transform(test_data['CriticalVendor'].astype(str))
test_data['Block Reason'] = lbl_enc.fit_transform(test_data['Block Reason'].astype(str))
test_data['VendorName'] = lbl_enc.fit_transform(test_data['VendorName'].astype(str))
test_data['Arrival Status'] = lbl_enc.fit_transform(test_data['Arrival Status'].astype(str))
test_data['InvoiceStatus'] = lbl_enc.fit_transform(test_data['InvoiceStatus'].astype(str))



dd = pd.read_csv('D_Train_Vendor.csv')

df = dd.groupby(['VendorName'])['AverageDaysLate'].agg(['mean']).reset_index()

df.to_csv('vendor_avgs.csv')

"""