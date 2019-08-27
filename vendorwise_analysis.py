os.chdir("/home/prudhvi/Documents/DSM")

from __future__ import division
from scipy.stats.mstats import  mode


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
           'ArrivalStatus', 'InvoiceStatus']
    data = pd.read_csv(path, usecols=col, index_col='DpDocumentNumber',
                       parse_dates=True, dayfirst=False)
    data['DpCreationDate'] = pd.to_datetime(data.DpCreationDate, format='%m/%d/%Y')
    data['ClearingDate'] = pd.to_datetime(data.ClearingDate, format='%m/%d/%Y')
    data['FiDueDate'] = pd.to_datetime(data.FiDueDate, format='%m/%d/%Y')
    return data






train_data = read_data('D_Train_BT.csv')

vendor_df = train_data.groupby('VendorName')['InvoiceStatus'].value_counts().unstack(fill_value=0)

vendor_df['Total_invoices'] = vendor_df.Delay + vendor_df.OnTime

vendor_df['delay_percentage'] = (vendor_df.Delay / vendor_df.Total_invoices) * 100

vendor_df['Percentage_in_total_delays'] = (vendor_df.Delay / sum(vendor_df.Delay) ) * 100

vendor_df = vendor_df.round(decimals = 4)

vendor_df.sort_values(by = 'Percentage_in_total_delays' , ascending = False , inplace = True )

top_vendors = list(vendor_df.index)[:200]


df = pd.DataFrame()

for vendor in top_vendors :
    #print vendor
    vendor_data = train_data[train_data.VendorName == vendor]
    d = dict()
    d['Vendor_Name'] = vendor
    d['Total_Invoices'] = len(vendor_data)
    clr_dates = np.unique(vendor_data.ClearingDate)
    popular_days = [i.dayofweek  for i in pd.to_datetime(clr_dates)]
    d['Number_of_Times_Cleared'] = len(clr_dates)
    d['Average_invoices_cleared_once'] = len(vendor_data)/len(clr_dates)

    diff = [ ]
    for i in range(1,len(clr_dates)) :
        diff.append(pd.Timedelta(clr_dates[i]-clr_dates[i-1]).days)

    frequent = Counter(diff).most_common(2)
    frequent = [i[0] for i in frequent]

    frequency_days = Counter(popular_days).most_common(2)
    frequency_days = [(ref[item[0]],item[1]) for item in frequency_days]
    d['most_frequent_clearing_frequency'] = frequent
    d['most_frequent_clearance_day'] = frequency_days
    df = df.append(d,ignore_index= True)

    print top_vendors.index(vendor) ,"Vendors done"



df.to_csv('Vendor_clearing_patterns.csv')







    """ 
    delay_dates =  []
    for i in range(1,len(clr_dates)) :
        diff.append(pd.Timedelta(clr_dates[i]-clr_dates[i-1]).days)
        if pd.Timedelta(clr_dates[i]-clr_dates[i-1]).days > med :
            delay_dates.append(clr_dates[i])

    vendor_refined = vendor_data[vendor_data.ClearingDate.isin(delay_dates)]
    print (Counter(vendor_data.InvoiceStatus)['Delay']/len(vendor_data)*100), " Old percentage"
    print (Counter(vendor_refined.InvoiceStatus)['Delay']/len(vendor_refined)*100), "Refined Percentage"

    print "-----------------------------" 

    """