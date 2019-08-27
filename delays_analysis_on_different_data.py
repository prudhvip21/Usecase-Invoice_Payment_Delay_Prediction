
import pandas as pd

data_ppd = pd.read_csv('PPD Template V_3.1.csv' , parse_dates = ['FiDueDate','DpCreationDate'])


data_ppd['arr'] = data_ppd.apply(lambda x : find_diff(x) , axis = 1)

def find_diff(row) :
    try :
        return pd.Timedelta(pd.to_datetime(row['FiDueDate']) - pd.to_datetime(row['DpCreationDate'])).days
    except :
        return 0

data_ppd = data_ppd[data_ppd['BG Name']=='DNP']

data_late = data_ppd[data_ppd['arr']<0]
Counter(data_late['PaymentTerm'])


data_normal = data_ppd[data_ppd['arr']>0]
Counter(data_normal['InvStatus'])


""" DNP clean Analysis  """

data = pd.read_csv('DNPClean.csv')
data['InvStatus'] = data['InvStatus'].fillna(value = 'test')


data_late = data[data['Arrival']=='LATE']
Counter(data_late['InvStatus'])


data_normal = data[data['Arrival']=='BEFORE']
data_normal['InvStatus'] = data_late['InvStatus'].fillna(value = 'test')
Counter(data_normal['InvStatus'])


data['arr'] = data.apply(lambda x : find_diff(x) , axis = 1)

kk = data.groupby(['InvStatus','Arrival'])['Arrival'].agg(['count'])


""" Train 2 year analysis """

data = pd.read_csv('Train2Yr.csv')

data['arr'] = data.apply(lambda x : find_diff(x) , axis = 1)

data_late = data[data['arr']<0]
Counter(data_late['InvStatus'])


data_normal = data[data['arr']>0]
Counter(data_normal['InvStatus'])

""" A_train and D_train analysis """

A_data = pd.read_csv('A_Train.csv')

A_data_status = A_data.groupby(['ArrivalStatus','InvoiceStatus'])['ArrivalStatus'].agg(['count'])


D_data = pd.read_csv('D_Train.csv')

D_data_status = D_data.groupby(['Arrival Status','InvoiceStatus'])['Arrival Status'].agg(['count'])


