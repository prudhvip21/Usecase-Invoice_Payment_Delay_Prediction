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
from sklearn.linear_model import Ridge

# returns current working directory
# os.getcwd()
# changes working directory
# os.chdir("C:\DSM")
# INPUT_PATH = "D_Train.csv"

cols = [1, 2, 3, 4, 5, 9]

columns = ['VendorName', 'CriticalVendor', 'ChannelId', 'BlockReason', 'BlockStatus', 'ArrivalStatus']


# 11, 'PaymentTerm',
def read_data(path):
    """
    Read the data into dataframe
    :param path:
    :return:
    """
    col = ['System', 'Region', 'DpDocumentNumber', 'Companycode', 'CurrencyLC', 'AmountLC', 'VendorNumber',
           'VendorName', 'CriticalVendor', 'ChannelId', 'BlockReason', 'BlockStatus', 'PTDays', 'InvtoCrtnDt',
           'OverduePayment', 'InvoiceDate', 'DpCreationDate', 'FiDueDate', 'ClearingDate', 'ArrivalStatus',
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


def main():
    # data = pd.read_csv('D_Test.csv')
    train_data = read_data('D_Train.csv')
    test_data = read_data('D_Test.csv')
    data = pd.concat([train_data, test_data])
    vendors = pd.read_csv('D_Train_Vendor.csv')
    train_data = train_data.reset_index()
    train_new = train_data.merge(vendors.iloc[:,[0,22,23,24,25,26,27]],how = 'left',on = 'DpDocumentNumber' , left_index= True)
    # data.columns #data.shape
    # impute missing vaues
    data = train_new

    for column in data:
        if (data[column].isnull().sum()) > 0:
            if column in ['VendorName', 'ChannelId', 'BlockReason', 'InvoiceStatus', 'Arrival Status','CriticalVendor']:
                data[column].fillna('Missing', inplace=True)
            if column in ['AmountLC']:
                data[column].fillna(0, inplace=True)
                data[column] = data[column].replace(np.NaN, 0)

    data.drop(['System', 'Region', 'Companycode', 'CurrencyLC', 'VendorNumber', 'InvoiceDate', 'DpCreationDate', 'FiDueDate',
         'ClearingDate'], axis=1, inplace=True)
    mean = np.mean(data.iloc[:,8])
    std  = np.std(data.iloc[:,8])
    data.iloc[:, [0,6,7,8]] = preprocessing.scale(data.iloc[:, [0,6,7,8]])
    # change data type as throwing error train.shape


    data["VendorName"] = data["VendorName"].astype('category')
    data["CriticalVendor"] = data["CriticalVendor"].astype('category')
    # data["Block Reason"] = data["Block Reason"].astype('category')
    data["ChannelId"] = data["ChannelId"].astype('category')
    data["ArrivalStatus"] = data["ArrivalStatus"].astype('category')
    data["InvoiceStatus"] = data["InvoiceStatus"].astype('category')

    # perform label encoding
    # perform label encoding
    lbl_enc = preprocessing.LabelEncoder()
    data['ChannelId'] = lbl_enc.fit_transform(data['ChannelId'].astype(str))
    data['CriticalVendor'] = lbl_enc.fit_transform(data['CriticalVendor'].astype(str))
    data['BlockReason'] = lbl_enc.fit_transform(data['BlockReason'].astype(str))
    data['BlockStatus'] = lbl_enc.fit_transform(data['BlockStatus'].astype(str))
    data['VendorName'] = lbl_enc.fit_transform(data['VendorName'].astype(str))
    data['ArrivalStatus'] = lbl_enc.fit_transform(data['ArrivalStatus'].astype(str))
    data['InvoiceStatus'] = lbl_enc.fit_transform(data['InvoiceStatus'].astype(str))

    cols = [2, 3, 4, 5, 6, 10]
    # for vendor levels

    encoded_data, feature_names = OneHot_Encoder(data, cols)
    encoded_data.shape
    # enc_cols = encoded_data.col
    # feature_names += ['AmountLC','PTDays','InvtoCrtnDt','OverduePayment']
    feature_names += ['AmountLC', 'PTDays', 'InvtoCrtnDt' , 'TotalLatePaidInv', 'TotalLatePaidInvAmt','TotalPaidInv','TotalPaidInvAmt', 'Ratio_LatePaidInv', 'Ratio_LatePaidInvAmt']
 ]
    # encoded_data.shape
    # encoded_data['AmountLC'] = train_scaled['AmountLC']
    encoded_data = hstack((encoded_data, np.array(data['AmountLC'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['PTDays'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['InvtoCrtnDt'])[:, None]))

    encoded_data = hstack((encoded_data, np.array(data['TotalLatePaidInv'])[:, None]))
encoded_data = hstack((encoded_data, np.array(data['TotalLatePaidInvAmt'])[:, None]))
encoded_data = hstack((encoded_data, np.array(data['TotalPaidInv'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['TotalPaidInvAmt'])[:, None]))
    encoded_data = hstack((encoded_data, np.array(data['Ratio_LatePaidInv'])[:, None]))
encoded_data = hstack((encoded_data, np.array(data['Ratio_LatePaidInvAmt'])[:, None]))

    model_svc = svm.SVC(kernel='linear', gamma=1, max_iter=2000)

    model_svc.fit(encoded_data, data.InvoiceStatus)
    train_pred = model_svc.predict(X_train)
    print(accuracy_score(y_train, train_pred))



    # encoded_data = hstack((encoded_data, np.array(data['OverduePayment'])[:, None]))=

    X_train, X_test, y_train_regr, y_test_regr = train_test_split(encoded_data, data.OverduePayment, shuffle=False,stratify=None, train_size=359851, test_size=7632)

    print(X_train.shape, y_train_regr.shape)
    print(X_test.shape, y_test.shape)

    """ regression prediction for Over time """

    regr = LinearSVR(random_state=0, max_iter=5000)
    regr.fit(X_train, y_train_regr)
    train_preds = regr.predict(X_train)
    clf = Ridge(alpha=0.5)
    clf.fit(X_train, y_train_regr)
    train_preds = regr.predict(X_train)

    # <6 is OnTime , else delays
    # after scaling  < -0.2309 is onTime

    delays = []
    for i in train_preds:
        if i > 0 :
            delays.append('Delay')
        else:
            delays.append('OnTime')
    accuracy_score(train_data.InvoiceStatus, delays)

    test_preds = regr.predict(X_test)

    test_delays = []
    for i in test_preds:
        if i > 0 :
            test_delays.append('Delay')
        else:
            test_delays.append('OnTime')

    test_data['regr_preds'] = test_delays
    test_data.to_csv('test_regr_predictions.csv')

    feature_imps = list(regr.coef_)
    features = {}
    for item in feature_names:
        f = str(item).split('_')[0]
        if f in features.keys():
            features[f] = round((features[f] + feature_imps[feature_names.index(item)]), 4)
        else:
            features[f] = round(feature_imps[feature_names.index(item)], 4)

    for key in features:
        features[key] = features[key] * features[key]
    s = sum(features.values())
    for key in features:
        features[key] = round(((features[key] / s) * 100), 4)

    print(sorted(features.items(), key=operator.itemgetter(1), reverse=True))


if __name__ == "__main__":
    main()

