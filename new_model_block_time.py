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
from sklearn.tree import  DecisionTreeRegressor
from scipy import stats

# returns current working directory
# os.getcwd()
# changes working directory
# os.chdir("D:\PA\DSM\POC24Nov\predictions\SIM1")

os.chdir("/home/prudhvi/Documents/DSM")
# INPUT_PATH = "D_Train.csv"

cols = [1,2,3,5,6,11]


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
    # data = pd.read_csv('D_Test.csv')
    train_data = read_data('D_Train_BT.csv')
    train_data = train_data[train_data['BlockStatus']=='Block']
    quant_df = train_data.BlockTime.quantile([0.25, 0.5, 0.75])
    upper_hinge = quant_df.loc[0.75] + 5*(quant_df.loc[0.75] - quant_df.loc[0.25])
    train_data = train_data[train_data['BlockTime'] < upper_hinge]
    #test_data = read_data('D_Test_final.csv')
    #test_data = pd.read_csv('DN_Test.csv')
    #test_data = read_data('D_test_BT_train.csv')
    test_data = read_data(('DN_Test_final.csv'))
    test_data = test_data[test_data['BlockStatus'] == 'Block']
    data = pd.concat([train_data, test_data])
    data = train_data
    # impute missing vaues
    for column in data:
        if (data[column].isnull().sum()) > 0:
            if column in ['VendorName', 'ChannelId', 'BlockReason','InvoiceStatus', 'Arrival Status', 'CriticalVendor']:
                data[column].fillna('Missing', inplace=True)
            if column in ['AmountLC','BlockTime']:
                data[column].fillna(0, inplace=True)
                data[column] = data[column].replace(np.NaN, 0)

    data.drop(['System', 'Region', 'Companycode', 'CurrencyLC', 'VendorNumber','InvoiceDate', 'DpCreationDate', 'FiDueDate', 'ClearingDate'],axis=1, inplace=True)

    #data.iloc[:, [0,8,10]] = preprocessing.scale(data.iloc[:, [0,8,10]])
    data.iloc[:, [0,8,11]] = preprocessing.scale(data.iloc[:, [0,8,11]])

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

    feature_names += ['AmountLC', 'PTDays', 'InvtoCrtnDt']

    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))

    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data,data.OverduePayment, shuffle=False, stratify=None,train_size=42733, test_size=2891)
    #5503

    # Predict OverduePayment
    model_lr = LinearRegression(fit_intercept=False)
    model_lr.fit(X_train, y_train_regr)
    model_lr.score(X_train, y_train_regr)

    overdue = model_lr.predict(X_test)
    # write back predicted over due payment to original data
    data['OverduePayment'][-2891:] = overdue

    encoded_data, feature_names = OneHot_Encoder(data, cols)
    feature_names += ['AmountLC', 'PTDays', 'InvtoCrtnDt', 'OverduePayment']

    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))

    """ Block time model """

    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data,data.BlockTime, shuffle=False, stratify=None,train_size=42733, test_size=2891)

    model_dtree = DecisionTreeRegressor()
    model_dtree.fit(X_train, y_train_regr)
    model_dtree.score(X_train, y_train_regr)

    test_block_time = model_dtree.predict(X_test)

    #test_block_time = test_block_time - 10
    #test_block_time = test_block_time - 15
    data['BlockTime'][-2891:] = test_block_time

    """ classification model """

    encoded_data, feature_names = OneHot_Encoder(data, cols)
    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))
    #encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['BlockTime'])[:, None]))

    X_train, X_test, y_train, y_test = train_test_split(encoded_data,data.InvoiceStatus, shuffle=False, stratify=None,train_size=42733,test_size=2891)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    """ Classification Model """

    model_logreg = LogisticRegression(penalty='l1')
    model_logreg.fit(X_train, y_train)
    train_pred = model_logreg.predict(X_train)
    print(accuracy_score(y_train, train_pred))

    logreg_preds = model_logreg.predict(X_test)
    print Counter(logreg_preds)
    test_data['preds'] = logreg_preds
    test_data.to_csv('new_preds_log_reg_sam.csv')
    #test_data['log_reg_preds_bt'] = logreg_preds
    #test_data.to_csv('log_reg_with_block_time.csv')
    df = pd.DataFrame()
    df['log_reg_preds'] = logreg_preds

    df.to_csv('log_reg_preds_bt.csv')

    joblib.dump(model_lr,"Overdue_Linear.pkl")
    joblib.dump(model_dtree, "Block_Linear.pkl")
    joblib.dump(model_logreg, "LogModel.pkl")

    """

    feature_imps = list(model_logreg.coef_[0])
    features = {}
    for item in feature_names :
        f = str(item).split('_')[0]
        if f in features.keys() :
            features[f] = round((features[f] +feature_imps[feature_names.index(item)]),4)
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

