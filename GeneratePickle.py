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

#returns current working directory
#os.getcwd()
#changes working directory
#os.chdir("D:\PA\DSM\POC24Nov\predictions\SIM1")
os.chdir('/home/prudhvi/Downloads/PickleMOdel')
#INPUT_PATH = "D_Train.csv"

cols = [1,2,3,4,5,10]

columns = ['ArrivalStatus', 'BlockReason', 'BlockStatus', 'ChannelId','CriticalVendor','VendorName']
    
# 11, 'PaymentTerm',
def read_data(path):
    """
    Read the data into dataframe
    :param path:
    :return:
    """
    col = ['System', 'Region', 'DpDocumentNumber', 'Companycode', 'CurrencyLC',
           'AmountLC', 'VendorNumber','VendorName','CriticalVendor', 'ChannelId', 
           'BlockReason', 'BlockStatus','PTDays', 'InvtoCrtnDt','OverduePayment',
           'InvoiceDate','DpCreationDate', 'FiDueDate', 'ClearingDate', 
           'ArrivalStatus','InvoiceStatus']
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

def predict_with_cutoff(colname, y_prob, df):
    n_events = df[colname].values
    event_rate = sum(n_events) / float(df.shape[0]) * 100
    threshold = np.percentile(y_prob[:, 1], 100 - event_rate)
    print("Cutoff/threshold at: " + str(threshold))
    y_pred = [1 if x >= threshold else 0 for x in y_prob[:, 1]]
    return y_pred

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a 
    classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def main():
    #data = pd.read_csv('D_Test.csv')
    train_data = read_data('D_Train.csv')
    test_data = read_data('D_Test_final.csv')
    data = pd.concat([train_data,test_data])
    #data.columns #data.shape

    #impute missing vaues
    for column in data:
        if (data[column].isnull().sum()) >0:
            if column in ['VendorName', 'ChannelId', 'BlockReason',
                          'InvoiceStatus','Arrival Status', 'CriticalVendor']:
                data[column].fillna('Missing' ,inplace = True)
            if column in ['AmountLC']:
                data[column].fillna(0 ,inplace = True)
                data[column] = data[column].replace(np.NaN, 0)    
       
    data.drop(['System', 'Region', 'Companycode', 'CurrencyLC','VendorNumber',
               'InvoiceDate','DpCreationDate', 'FiDueDate', 'ClearingDate'],
        axis =1, inplace=True)
     
    
    data.iloc[:,[0,7,9]] = preprocessing.scale(data.iloc[:,[0,7,9]])
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
    #encoded_data.shape
    #encoded_data['AmountLC'] = train_scaled['AmountLC']
    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))
    #encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))
   
    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data,
    data.OverduePayment, shuffle=False, stratify=None,train_size=359038, test_size=4985)
    #Predict OverduePayment
    regr = LinearSVR(random_state=0, max_iter=10000, fit_intercept= False)
    regr.fit(X_train,y_train_regr)
    regr.score(X_train, y_train_regr)
    overdue = regr.predict(X_test)
    # write back predicted over due payment to original data
    data['OverduePayment'][-4985:] = overdue

    #filename = 'finalized_regression_model.sav'
    #pickle.dump(regr, open(filename, 'wb'))

    encoded_data,feature_names = OneHot_Encoder(data, cols)
    feature_names += ['AmountLC','PTDays','InvtoCrtnDt','OverduePayment']

    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))

    X_train, X_test, y_train, y_test = train_test_split(encoded_data, 
    data.InvoiceStatus,shuffle = False, stratify= None, train_size= 359038,
    test_size= 4985 )

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
         
    
    """ Classification Model """

    model_logreg = LogisticRegression(penalty= 'l1')
    model_logreg.fit(X_train, y_train)
    train_pred = model_logreg.predict(X_train)
    print(accuracy_score(y_train, train_pred))

    #filename = 'finalized_logistic_model.sav'
    #pickle.dump(model_logreg, open(filename, 'wb'))

    logreg_preds = model_logreg.predict(X_test)
    print Counter(logreg_preds)
    df = pd.DataFrame()
    df['log_reg_preds'] = logreg_preds
    #test_data['overdue'] = overdue
    #test_data.to_csv('overdue.csv')
    #del test_data['overdue']
    df.to_csv('log_reg_preds.csv')
    
    joblib.dump(regr, "LinearModel.pkl")
    joblib.dump(model_logreg, "LogModel.pkl")

    """ 
    feature_imps = list(model_logreg2.coef_)
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




df = pd.read_csv('log_reg_preds.csv')
test_data.reset_index(inplace= True)
test_data['preds'] = df['log_reg_preds']

counts = test_data.groupby('VendorNumber')['preds'].count().sort_values(ascending =  False)
counts.head(5)

test_data['type'] = test_data.apply(lambda x : 'Short' if x['PTDays'] < 15 else 'Regular' , axis = 1)

vendor_df = test_data.groupby(['VendorNumber','type'])['preds'].value_counts().unstack(fill_value=0)

vendor_df.columns = ['Delay','OnTime']
vendor_df['Total_invoices'] = vendor_df.Delay + vendor_df.OnTime

vendor_df['delay_percentage'] = (vendor_df.Delay / vendor_df.Total_invoices) * 100

vendor_df['Percentage_in_total_delays'] = (vendor_df.Delay / sum(vendor_df.Delay) ) * 100

vendor_df = vendor_df.round(decimals = 4)
vendor_df.reset_index(inplace= True)
vendor_df.sort_values(by = ['type','Percentage_in_total_delays'] , ascending = False , inplace = True )

vendor_df.to_csv('test_vendors.csv')