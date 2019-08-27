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
from scipy.stats import spearmanr, kendalltau, pearsonr                  # for chi-square
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, hstack
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from prettytable import PrettyTable
from rpy2.robjects import r, pandas2ri
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
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

pandas2ri.activate()

#returns current working directory
os.getcwd()
scaler = preprocessing.MinMaxScaler()
#changes working directory
os.chdir("D:\PA\DSM\POC24Nov")
INPUT_PATH = "D:\PA\DSM\POC24Nov\Train2Yr_Test.csv"
cols = [0,1,3,5,6,7,8]

columns = ['System', 'Region', 'Companycode', 
       'Vendor Name','Critical vendor',
       'ChannelId', 'Block Status']
    
# 11, 'PaymentTerm',
def read_data(path):
    """
    Read the data into dataframe
    :param path:
    :return:
    """
    data = pd.read_csv(path)
    return data

def getDay(colDate):
    """
    :param date:
    :return:day
    """
    return pd.DatetimeIndex(colDate).day

def Due_Dt_Position(row):
    DueDay = row['DueDay']
    DelayPos = 0
    if DueDay > 19:
        DelayPos = 1
    return DelayPos

def Due_Dt_HalfMonth(row):
    DueDay = row['DueDay']
    DelayPos = 0
    if DueDay <16:
        DelayPos = 1
    return DelayPos

def Inv_Late(row):
    DueCnt = row['DueDtToInvDt']
    DelayPos = 0
    if DueCnt <= 0:
        DelayPos = 1
    return DelayPos
    
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) 
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def getEncoded(colName):
    """
    encode label
    """
    lbl_enc = preprocessing.LabelEncoder()
    return lbl_enc.fit_transform(colName)

def OneHot_Encoder(df, cols):
    enc = OneHotEncoder(sparse = True)
    encoded_data = enc.fit_transform(df.iloc[:,cols].as_matrix())
    #print encoded_data.shape
    #print type(encoded_data)
    #encoded_date = pd.DataFrame(encoded_data.todense())
    """ Feature names code """
    features = [ ]
    for i, name in enumerate(columns) :
        l = len(Counter(df[name]))
        for k in range(l) :
            f_nm = str(name) + '_' + str(k+1)
            features.append(f_nm)
            
    '''for i, name in enumerate(features):
        encoded_data.rename(columns={i:name},inplace=True)'''
    encoded_data.col = features
    
    return encoded_data



def correlation_matrix(df):    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Invoice Status Feature Correlation')
    labels=['System', 'Companycode',
       'AmountLC', 'Vendor Name', 'PO Item number',
       'ChannelId', 'Block Status',
       'InvDtToCrtnDt','InvStatus']
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
    
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
    
def chi_value(col1, col2):
    df = pd.crosstab(col1, col2)
    return scipy.stats.chi2_contingency(df)

def main():
    data = read_data(INPUT_PATH)
    
    data.drop_duplicates('DpDocumentNumber', keep='first', inplace = True)
   
    
    for column in data:
        if (data[column].isnull().sum()) >0:
            if column in ['Vendor Name', 'ChannelId', 'POInvoiceType', 'Block Status',
                          'Critical vendor', 'Region', 'Companycode', 'System']:
                data[column].fillna('Missing' ,inplace = True)
            if column in [ 'AmountLC', 'DueDtToInvDt','InvDtToCrtnDt', 'FiDueDate']:
                data[column].fillna(0 ,inplace = True)
                data[column] = data[column].replace(np.NaN, 0)
            
    
     
    data['AmountLC'] = data['AmountLC'].replace(np.NaN, 0)  
    data['DueDtToInvDt'] = data['DueDtToInvDt'].replace(np.NaN, 0)  
    data['InvDtToCrtnDt'] = data['InvDtToCrtnDt'].replace(np.NaN, 0)
    data['FiDueDate'] = data['FiDueDate'].replace(np.NaN, 0)  
    #data.isnull().any() data.columns data.head(5)
   
    train = data
    train = train.drop(['BG Name','FiDueDate', 'CurrencyLC',
                       'InvoiceDate', 'DpCreationDate'], 1)
        
    #train.shape, train.columns
    del data
    
    quant_df = train.quantile([0.25,0.5,0.75,0.99])
    
    quant_df.loc[4,:] = quant_df.apply(lambda x : x[0.75]-x[0.25],axis = 0)  
    quant_df.loc[5,:] = quant_df.apply(lambda x : x[0.5]+1.5*x[4],axis = 0)
    quant_df.loc[6,:] = quant_df.apply(lambda x : x[0.5]-1.5*x[4] , axis = 0)
    
    train = train.loc[train['DueDtToInvDt'] < quant_df.iloc[5]['DueDtToInvDt']]
    train = train.loc[train['DueDtToInvDt'] > quant_df.iloc[6]['DueDtToInvDt']]
    
    train = train.loc[train['InvDtToCrtnDt'] < quant_df.iloc[5]['InvDtToCrtnDt']]
    train = train.loc[train['InvDtToCrtnDt'] > quant_df.iloc[6]['InvDtToCrtnDt']]
    
    df = preprocessing.scale(train.iloc[:,[4,9,10]])
    train.iloc[:,[4,9,10]] = preprocessing.scale(train.iloc[:,[4,9,10]])
    del quant_df
   #change data type as throwing error train.shape
      
    train["Companycode"] = train["Companycode"].astype('category') 
    train["Vendor Name"] = train["Vendor Name"].astype('category')
    #train["POInvoiceType"] = train["POInvoiceType"].astype('category')
    #train["POItemNo"] = train["POItemNo"].astype('category')
    train["ChannelId"] = train["ChannelId"].astype('category')
    train['Companycode'] = pd.to_numeric(train['Companycode'], errors='ignore')
    train['Companycode'].fillna(0 ,inplace = True)
    #train.dtypes.head(20)  train.head(10)
    lbl_enc = preprocessing.LabelEncoder()
    train['Companycode'] = lbl_enc.fit_transform(train['Companycode'].astype(str))
    train['Critical vendor'] = lbl_enc.fit_transform(train['Critical vendor'].astype(str))
    train['InvStatus'] = lbl_enc.fit_transform(train['InvStatus'].astype(str))
    #perform label encoding    
    train['System'] = getEncoded(train['System'])
    train['Region'] = getEncoded(train['Region'])
    #train['Companycode'] = getEncoded(train['Companycode'])#*
    train['Vendor Name'] = getEncoded(train['Vendor Name'])
    #train['Critical vendor'] = getEncoded(train['Critical vendor'])#*
    
    train['ChannelId'] = getEncoded(train['ChannelId'])
    train['Block Status'] = getEncoded(train['Block Status'])    
    #train['InvStatus'] = getEncoded(train['InvStatus'])#*
       

    #train_scaled = pd.DataFrame(scaler.fit_transform(train))
    train_scaled = train
    
    #train_scaled.columns train.head(5)   train_scaled.head(5) train.columns train_scaled.shape
    
    encoded_data = OneHot_Encoder(train_scaled, cols)
    
    enc_cols = encoded_data.col 
    enc_cols += ['AmountLC','DueDtToInvDt','InvDtToCrtnDt']
    #encoded_data.shape
    #encoded_data['AmountLC'] = train_scaled['AmountLC']
    encoded_data = hstack((encoded_data, np.array(train_scaled['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(train_scaled['DueDtToInvDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(train_scaled['InvDtToCrtnDt'])[:, None]))
    
    
    index = list(train_scaled.index)
    index = range(0,len(train_scaled))
    train_index = index[0:-8676]
    test_index = index[-8676:]
    
    X_train = encoded_data.tocsr()[train_index,:]
    X_train = encoded_data[train_index]
    X_test = encoded_data[test_index]
    y_train = train_scaled.iloc[:,11][train_index]
    y_test = train_scaled.iloc[:,11][test_index]
    
    #X_train, X_test, y_train, y_test = train_test_split(encoded_data, train_scaled.InvStatus,  
    #                 test_size=0.33, random_state=5)
    
    del encoded_data
    del train_scaled 
    del train

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    
    #Ensemble model
    model_log = LogisticRegression(random_state=1)
    model_rfc = RandomForestClassifier(random_state=1)
    model_gnb = GaussianNB()    
    ensmbl_clf = VotingClassifier(estimators=[('lr', model_log), ('rf', model_rfc),
                                              ('gnb', model_gnb)], voting='soft')
    eclf = ensmbl_clf.fit(X_train,y_train)
    pred_svc = eclf.predict(X_test)
    #GradientBoostingClassifier
    gbc = GradientBoostingClassifier(random_state=2017, learning_rate=0.1)
    gbc_grid = {'n_estimators':list(range(50,100,50))}
    model_gbc = model_selection.GridSearchCV(gbc, gbc_grid, cv=10, n_jobs=10)
    model_gbc.fit(X_train, y_train)
    pred_gbc = model_gbc.predict(X_test) 
    print(confusion_matrix(y_test,pred_gbc))
    print(accuracy_score(y_test,pred_gbc))
    # save the model to disk
    filename = 'GradientBoostingClassifier_model.sav'
    pickle.dump(model_gbc, open(filename, 'wb')) 
    
    # natively deploy decision tree model(pickle format)
    joblib.dump(model_gbc, "GBClf1.pkl")
    
    #At any time in future, you can load the model and test or use it    
    # load the model from disk
    '''loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    print(result)'''
     
    #Model Based Feature Selection    
    select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="median")

    select.fit(X_train, y_train)
    X_train_l1 = select.transform(X_train)
    X_test_l1 = select.transform(X_test)
    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_l1.shape: {}".format(X_train_l1.shape))
    
    mask = select.get_support()
    # visualize the mask -- black is True, white is False
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Sample index")
    plt.yticks(())
        
    """ Classification Model """  
    # Spot Check Algorithms
    # Define Support Vector Machine
    model_svc = svm.SVC(kernel='linear', gamma=1, max_iter=10000)
    model_svc.fit(X_train_l1,y_train)
    pred_svc = model_svc.predict(X_test_l1) 
    print(confusion_matrix(y_test,pred_svc))
    print(accuracy_score(y_test,pred_svc))
    
    feature_imps = list(model_svc.coef_.toarray()[0,:])
    features = {}
    for item in enc_cols :
        f = str(item).split('_')[0]
        if f in features.keys() :
            features[f] = round((features[f] + feature_imps[enc_cols.index(item)]),4)
        else :
            features[f] = round(feature_imps[enc_cols.index(item)],4)
    
    print(sorted(features.items(), key=operator.itemgetter(1), reverse=True))
    
                
if __name__ == "__main__":
    main()