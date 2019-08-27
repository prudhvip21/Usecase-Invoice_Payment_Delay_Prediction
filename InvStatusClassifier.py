import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, hstack
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
#returns current working directory
os.getcwd()
scaler = preprocessing.MinMaxScaler()
#changes working directory
os.chdir("D:\PA\DSM\POC24Nov")
INPUT_PATH = "D:\PA\DSM\POC24Nov\TrainPPDCalc.csv"
cols = [0,1,3,4,6,7,9,10,11,12,13]

columns = ['System', 'Region','Companycode', 'CurrencyLC',
       'Vendor Name', 'POInvoiceType',
       'Critical vendor', 'ChannelId', 'PaymentTerm', 'BG Name','Block Status', 
       ]
# 11, 'PaymentTerm',
def read_data(path):
    """
    Read the data into dataframe
    :param path:
    :return:
    """
    data = pd.read_csv(path)
    return data

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
    encoded_data.columns = features
    
    return encoded_data


def main():
    data = read_data(INPUT_PATH)
    data.drop_duplicates('DpDocumentNumber', keep='first', inplace = True)
    #impute missing vaues
    for column in data:
        if (data[column].isnull().sum()) >0:
            if column in ['POInvoiceType','PaymentTerm', 'Vendor Name', 'ChannelId']:
                data[column].fillna('Missing' ,inplace = True)
            if column in ['PO Item number']:
                data[column].fillna(1 ,inplace = True)
                data[column] = data[column].replace (np.NaN, 0)
            if column in ['AmountLC']:
                data[column].fillna(0 ,inplace = True)
                data[column] = data[column].replace(np.NaN, 0)
        
    data.isnull().any()
    # count the number of NaN values in each column
    print(data.isnull().sum())

    '''data = data.fillna(method='ffill')'''

    train = data[data['BG Name']== 'DNP']
    del data

    quant_df = train.quantile([0.25,0.5,0.75,0.99])
    quant_df.loc[4,:] = quant_df.apply(lambda x : x[0.75]-x[0.25],axis = 0)
    quant_df.loc[5,:] = quant_df.apply(lambda x : x[0.5]+1.5*x[4],axis = 0)
    quant_df.loc[6,:] = quant_df.apply(lambda x : x[0.5]-1.5*x[4] , axis = 0)
    
    train = train.loc[train['DueDtToInvDt'] < quant_df.iloc[5]['DueDtToInvDt']]
    train = train.loc[train['DueDtToInvDt'] > quant_df.iloc[6]['DueDtToInvDt']]
    
    train = train.loc[train['InvDtToCrtnDt'] < quant_df.iloc[5]['InvDtToCrtnDt']]
    train = train.loc[train['InvDtToCrtnDt'] > quant_df.iloc[6]['InvDtToCrtnDt']]
    
    train = train[train['RespToRefrDt'] <= quant_df.iloc[3]['RespToRefrDt']]
    
    train = train[train['RefrToPostDt'] <= quant_df.iloc[3]['RefrToPostDt']]
        
    train = train[train['BlockDays'] <= quant_df.iloc[3]['BlockDays']]
    
    
    train.iloc[:,[5,14,15,16,17,18]] = preprocessing.scale(train.iloc[:,[5,14,15,16,17,18]])
    del quant_df
   #change data type as throwing error
      
    train["Companycode"] = train["Companycode"].astype('category') 
    train["PaymentTerm"] = train["PaymentTerm"].astype('category')
    train["Vendor Name"] = train["Vendor Name"].astype('category')
    train["POInvoiceType"] = train["POInvoiceType"].astype('category')
    train['System'] = train['System'].astype('category')
    train['Region'] = train['Region'].astype('category')

    #train["POItemNo"] = train["POItemNo"].astype('category')
    train["ChannelId"] = train["ChannelId"].astype('category')
    train['Companycode'] = pd.to_numeric(train['Companycode'], errors='ignore')
    train['Companycode'].fillna(0 ,inplace = True)
    train.dtypes.head(20) 
    
    #perform label encoding    
    train['System'] = getEncoded(train['System'])
    train['Region'] = getEncoded(train['Region'])
    train['Companycode'] = getEncoded(train['Companycode'])
    train['CurrencyLC'] = getEncoded(train['CurrencyLC'])
    train['Vendor Name'] = getEncoded(train['Vendor Name'])
    train['POInvoiceType'] = getEncoded(train['POInvoiceType'])
    #train['POItemNo'] = getEncoded(train['POItemNo'])
    train['Critical vendor'] = getEncoded(train['Critical vendor'])
    
    train['ChannelId'] = getEncoded(train['ChannelId'])
    train['PaymentTerm'] = getEncoded(train['PaymentTerm'])
    train['BG Name'] = getEncoded(train['BG Name'])
    train['Block Status'] = getEncoded(train['Block Status'])

    train['InvStatus'] = getEncoded(train['InvStatus'])
    
    #train_scaled = train.drop(['RefrtoPostDt'])
    # del train['RefrToPostDt']

    #train_scaled = pd.DataFrame(scaler.fit_transform(train))
    train_scaled = train
    del train
    train_scaled.shape
    train_scaled.head(19)
    train_scaled.columns
    
    train_scaled.dtypes.head(19)
    '''
    writer = ExcelWriter('EncodedTrain.xlsx')
    encoded_data.to_excel(writer,'Sheet2')
    writer.save()
    '''
    #train_scaled.columns   train_scaled.head(1) train.columns
    '''perform one hot encoding'''
    encoded_data = OneHot_Encoder(train_scaled, cols)

    '''df = train_scaled
    enc = OneHotEncoder()
    encoded_data = enc.fit_transform(df.iloc[:,cols])
    encoded_data = pd.DataFrame(encoded_data.toarray())
    """ Feature names code """
    features = [ ]
    for i, name in enumerate(columns) :
        l = len(Counter(df[name]))
        for k in range(l) :
            f_nm = str(name) + '_' + str(k+1)
            features.append(f_nm)
           
    encoded_data.columns = features'''

    #train_scaled['AmountLC'] = train_scaled['AmountLC'].replace(np.NaN , 0)
    #train_scaled['PO Item number'] = train_scaled['PO Item number'].replace(np.NaN, 0)

    encoded_data = hstack((encoded_data, np.array(train_scaled['AmountLC'])[:,None ]))
    encoded_data = hstack((encoded_data, np.array(train_scaled['PO Item number'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(train_scaled['InvDtToCrtnDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(train_scaled['DueDtToInvDt'])[:, None]))
    #encoded_data = hstack((encoded_data, np.array(train_scaled['RefrToPostDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(train_scaled['RespToRefrDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(train_scaled['BlockDays'])[:, None]))

    #encoded_data['AmountLC'] = train_scaled['AmountLC']
    #encoded_data['POItemNo'] = train_scaled['POItemNo']
    
    #encoded_data['AmountLC'] = encoded_data['AmountLC'].replace(np.NaN, 0)
    #encoded_data['POItemNo'] = encoded_data['POItemNo'].replace(np.NaN, 0)
    #encoded_data.columns, encoded_data.head(5)
    
    # Data Partition -> train and test split
    #indices = range(len(encoded_data))
    '''X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(encoded_data, train_scaled.iloc[:,18], indices, 
                     test_size=0.33, random_state=1)'''
    X_train, X_test, y_train, y_test = train_test_split(encoded_data, train_scaled.iloc[:,19],
                     test_size=0.33, random_state=5)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    ''''train_scaled['PaymentTerm'].head()
    train_scaled['AmountLC'].value_counts().plot(kind="bar")
    plt.title('InvoiceAmount')
    plt.xlabel('InvAmt')
    plt.ylabel('Count')
    sns.despine
    
    my_tab = pd.crosstab(index = train_scaled["InvStatus"],  # Make a crosstab
                              columns="count")      # Name the count column

    my_tab.plot.bar()
    plt.scatter(train_scaled['InvStatus'], train_scaled['BlockDays'])
    plt.title('Delay Vs Block')
    '''
    
    """ Classification Model """  
    # Spot Check Algorithms
    # Define Support Vector Machine
    model_svc = svm.SVC(kernel='linear', gamma=1)
    model_svc.fit(X_train,y_train)
    pred_svc = model_svc.predict(X_test) 
    print(confusion_matrix(y_test,pred_svc))
    print(accuracy_score(y_test,pred_svc))
    interpreter = Interpretation(X_test, feature_names=encoded_data.columns)
    pyint_model = InMemoryModel(model_svc.predict_proba, examples=X_test)
    feature_imps = interpreter.feature_importance.feature_importance(pyint_model)

    features = {}
    for item in X_train.columns :
        f = item.split('_')[0]
        if f in features.keys() :
            features[f] = round((features[f] + feature_imps[item]),4)
        else :
            features[f] = feature_imps[item]
    print features
            
if __name__ == "__main__":
    main()
