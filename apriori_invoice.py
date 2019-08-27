import os
import pandas as pd
import numpy as np
import csv

# reading data and filtering only DNP

os.chdir("/home/prudhvi/Documents/invoice-delay")
data = pd.read_csv('Train2Yr.csv')
data_apriori = data[data['BG Name']== 'DNP']

# Taking only required olumns
data_apriori = data_apriori.iloc[:,[3,5,6,8,9,14,15,19,22]]
quant_df = data_apriori.quantile([0.25,0.5,0.75,0.99])
quant_df.loc[4, :] = quant_df.apply(lambda x: x[0.75] - x[0.25], axis=0)
quant_df.loc[5, :] = quant_df.apply(lambda x: x[0.5] + 1.5 * x[4], axis=0)
quant_df.loc[6, :] = quant_df.apply(lambda x: x[0.5] - 1.5 * x[4], axis=0)

# removing outliers based on quantiles

data_apriori = data_apriori.loc[data_apriori['DueDtToInvDt'] < quant_df.iloc[5]['DueDtToInvDt']]
data_apriori = data_apriori.loc[data_apriori['DueDtToInvDt'] > quant_df.iloc[6]['DueDtToInvDt']]
data_apriori['PO Item number'].fillna(1,inplace = True)

data_apriori = data_apriori.loc[data_apriori['InvDtToCrtnDt'] < quant_df.iloc[5]['InvDtToCrtnDt']]
data_apriori = data_apriori.loc[data_apriori['InvDtToCrtnDt'] > quant_df.iloc[6]['InvDtToCrtnDt']]

# Binning the data - equal frequency

data_apriori['Amount_Label'],amount_bins = pd.qcut(data_apriori['AmountLC'],5,retbins = True, labels = ['A_very_high','A_high','A_medium','A_low','A_very_low'])

data_apriori['PO_item_label'],po_bins = pd.qcut(data_apriori['PO Item number'], 2, retbins=True, labels=['P_low','P_high'], duplicates  = 'drop')

data_apriori['Due_Inv_label'],due_inv_bins = pd.qcut(data_apriori['DueDtToInvDt'], 5, retbins=True, labels=['D_very_low','D_low','D_medium','D_high','D_very_high'], duplicates  = 'drop')

data_apriori['Inv_crtn_label'],inv_crtn_bins = pd.qcut(data_apriori['InvDtToCrtnDt'], 5, retbins=True, labels=['I_very_low','I_low','I_medium','I_high','I_very_high'], duplicates  = 'drop')

""" Taking only delay invoices and generating patterns """

data_delay = data_apriori[data_apriori['InvStatus']=='Delay']
data_delay = data_delay.iloc[:,[1,6,7,8,9]]
# aprior algo is taking from CSV only. So writing into CSV and reading back
data_delay.to_csv('data_delay.csv',index = False ,header = False)

# Execute the file 'apriori.py' before running this step

items,rules = runApriori(dataFromFile('data_delay.csv'),0.001,0.8)
# support for itemssets - 0.001 , confidence for rules  0.8

""" Taking vendors names and finding itemsets specific to vendors """

vendors_list = list(data_delay['Vendor Name'])
vendor_patterns = [ ]
for item in items :
    if len(set(list(item[0])).intersection(vendors_list)) != 0 :
        vendor_patterns.append(item)

# creating empty data frame for  writing patterns

df = pd.DataFrame(columns=['Vendor name','Amount','Inv_to_crtn','Due_to_Inv','PO_item','support'])
for item in vendor_patterns :
    patterns = list(item[0])
    d = { }
    if len(patterns) > 2 :
        for pattern in patterns :
            p = pattern.split('_')
            if p[0] == 'P' :
                d['PO_item'] = pattern
            elif p[0] == 'D':
                d['Due_to_Inv'] = pattern
            elif p[0] == 'I':
                d['Inv_to_crtn'] = pattern
            elif p[0] == 'A' :
                d['Amount'] = pattern
            else :
                d['Vendor name'] = pattern
        d['support'] = round(item[1],3)
        df = df.append(d,ignore_index= True)

# filling NA with space
df = df.fillna(' ')
df.to_csv('observations_delay.csv')


""" Delay and no-delay both considered  """

# same steps

data_dnd = data_apriori.iloc[:,[1,5,6,7,8,9]]
data_dnd.to_csv('data_dnd.csv',index = False ,header = False)
items,rules = runApriori(dataFromFile('data_dnd.csv'),0.001,0.05)

rules_updated = [ ]
for k in rules :
    precendent = list(k[0][1])
    if 'OnTime' in precendent and len(precendent) == 1:
        rules_updated.append(k)
    elif 'Delay' in precendent and len(precendent) == 1:
        rules_updated.append(k)
    else :
        continue

vendors_list = list(data_dnd['Vendor Name'])
vendor_patterns = [ ]
for item in rules_updated :
    if len(set(list(item[0][0])).intersection(vendors_list)) != 0 :
        vendor_patterns.append(item)

df = pd.DataFrame(columns=['Vendor name','Amount','Inv_to_crtn','Due_to_Inv','PO_item','Status'])

for item in vendor_patterns :
    patterns = list(item[0][0])
    d = { }
    if len(patterns) > 3 :
        d['Status'] = item[0][1][0]
        for pattern in patterns :
            p = pattern.split('_')
            if p[0] == 'P' :
                d['PO_item'] = pattern
            elif p[0] == 'D':
                d['Due_to_Inv'] = pattern
            elif p[0] == 'I':
                d['Inv_to_crtn'] = pattern
            elif p[0] == 'A' :
                d['Amount'] = pattern
            else :
                d['Vendor name'] = pattern
        df = df.append(d,ignore_index= True)

df = df.fillna(' ')
df.to_csv('observations_3.csv')


""" Only ontime  """

data_ont = data_apriori[data_apriori['InvStatus']=='OnTime']
data_ont = data_ont.iloc[:,[1,6,7,8,9]]
data_ont.to_csv('data_ont.csv',index = False ,header = False)
items,rules = runApriori(dataFromFile('data_ont.csv'),0.001,0.4)

vendors_list = list(data_ont['Vendor Name'])
vendor_patterns = [ ]
for item in items :
    if len(set(list(item[0])).intersection(vendors_list)) != 0 :
        vendor_patterns.append(item)

df = pd.DataFrame(columns=['Vendor name','Amount','Inv_to_crtn','Due_to_Inv','PO_item','support'])
for item in vendor_patterns :
    patterns = list(item[0])
    d = { }
    if len(patterns) > 2 :
        for pattern in patterns :
            p = pattern.split('_')
            if p[0] == 'P' :
                d['PO_item'] = pattern
            elif p[0] == 'D':
                d['Due_to_Inv'] = pattern
            elif p[0] == 'I':
                d['Inv_to_crtn'] = pattern
            elif p[0] == 'A' :
                d['Amount'] = pattern
            else :
                d['Vendor name'] = pattern
        d['support'] = round(item[1],3)
        df = df.append(d,ignore_index= True)
df = df.fillna(' ')
df.to_csv('observations_ontime.csv')


""" Vendor Specific - top 5 vendors """

counts = data_apriori.groupby('Vendor Name')['InvStatus'].count().sort_values(ascending = False)[:7]
top_vendors = ['UTC OVERSEAS, INC.','Office Depot','Corporate Travel Services Pte Ltd','GEM','Swisscom (Schweiz) AG']

for v in top_vendors :
    data_vendor = data_apriori[data_apriori['Vendor Name'] == v ]
    data_dnd = data_vendor.iloc[:,[5,6,7,8,9]]
    data_dnd.to_csv('data_dnd.csv',index = False ,header = False)
    items,rules = runApriori(dataFromFile('data_dnd.csv'),0.01,0.01)

    rules_updated = [ ]
    for k in rules :
        precedent = list(k[0][1])
        if 'OnTime' in precedent and len(precedent) == 1:
            rules_updated.append(k)
        elif 'Delay' in precedent and len(precedent) == 1:
            rules_updated.append(k)
        else :
            continue

    df = pd.DataFrame(columns=['Amount','Inv_to_crtn','Due_to_Inv','PO_item','Status','Confidence'])

    for item in rules_updated :
        patterns = list(item[0][0])
        d = { }
        if len(patterns) > 0 :
            d['Confidence'] = round(item[1],3)
            d['Status'] = item[0][1][0]
            for pattern in patterns :
                p = pattern.split('_')
                if p[0] == 'P' :
                    d['PO_item'] = pattern
                elif p[0] == 'D':
                    d['Due_to_Inv'] = pattern
                elif p[0] == 'I':
                    d['Inv_to_crtn'] = pattern
                else  :
                    d['Amount'] = pattern
            df = df.append(d,ignore_index= True)

    df = df.fillna(' ')
    file_name = 'patterns' +  '_' + v + '.csv'
    df.to_csv(file_name)


""" Counting vendorwise """

vendor_df = data_apriori.groupby('Vendor Name')['InvStatus'].value_counts().unstack(fill_value=0)
vendor_df['Total_invoices'] = vendor_df.Delay + vendor_df.OnTime

vendor_df['delay_percentage'] = (vendor_df.Delay / vendor_df.Total_invoices) * 100

vendor_df['Percentage_in_total_delays'] = (vendor_df.Delay / sum(vendor_df.Delay) ) * 100

vendor_df = vendor_df.round(decimals = 4)
vendor_df.sort_values(by = 'Percentage_in_total_delays' , ascending = False , inplace = True )

vendor_df.to_csv('vendors.csv')

list_of_vendors = list(vendor_df.index[:500])


def freq_patterns(df,list_of_vendors,status,filename) :
    df = df[df['InvStatus'] == status ]
    df.to_csv('data.csv', index=False, header=False)
    items, rules = runApriori(dataFromFile('data.csv'), 0.005, 0.1)
    vendor_patterns = []
    for item in items:
        if len(set(list(item[0])).intersection(list_of_vendors)) != 0:
            vendor_patterns.append(item)
    df_patterns = pd.DataFrame(columns=['Vendor name', 'Amount', 'Inv_to_crtn', 'Due_to_Inv','CV_status', 'PO_item', 'support',])
    for item in vendor_patterns:
        patterns = list(item[0])
        d = {}
        if len(patterns) > 2:
            for pattern in patterns:
                p = pattern.split('_')
                if p[0] == 'P':
                    d['PO_item'] = pattern
                elif p[0] == 'D':
                    d['Due_to_Inv'] = pattern
                elif p[0] == 'I':
                    d['Inv_to_crtn'] = pattern
                elif p[0] == 'A':
                    d['Amount'] = pattern
                elif p[0] == 'Not' :
                    d['CV_status'] = pattern
                elif p[0] == 'CV' :
                    d['CV_status'] = pattern
                else:
                    d['Vendor name'] = pattern
            d['support'] = round(item[1], 3)
            df_patterns = df_patterns.append(d, ignore_index=True)
    df_patterns = df_patterns.fillna(' ')
    df_patterns.to_csv(filename)
    return df_patterns

data_patterns = data_apriori.iloc[:,[1,3,6,7,8,9,10]]
df_160 = data_patterns[data_patterns['Vendor Name'].isin(list_of_vendors)]
df_160['Critical vendor'] = df_160['Critical vendor'].astype(str)
df_160.replace(to_replace = ['True','False'] , value = ['CV','Not_CV'],inplace = True)


patterns_160 = freq_patterns(df_160,list_of_vendors,'Delay','Delay_patterns_160.csv')

patterns_160 = freq_patterns(df_160,list_of_vendors,'OnTime','OnTime_patterns_160.csv')


patterns_160.to_csv('delay_patterns.csv')

""" Vendor wise data frame with support """


def freq_patterns(df,list_of_vendors,status,filename) :
    df = df[df['InvStatus'] == status ]
    df_patterns = pd.DataFrame(columns=['Vendor name', 'Amount','Inv_to_crtn', 'Due_to_Inv', 'CV_status', 'PO_item', 'support'])
    for ven in list_of_vendors :
        df_ven = df[df['Vendor Name'] == ven]
        df_ven.to_csv('data.csv', index=False, header=False)
        items, rules = runApriori(dataFromFile('data.csv'), 0.6, 0.4)
        vendor_patterns = []
        for item in items:
            if len(set(list(item[0])).intersection([ven])) != 0:
                vendor_patterns.append(item)

        #print len(vendor_patterns)

        for item in vendor_patterns:
            patterns = list(item[0])
            d = {}
            if len(patterns) > 3:
                for pattern in patterns:
                    p = pattern.split('_')
                    if p[0] == 'P':
                        d['PO_item'] = pattern
                    elif p[0] == 'D':
                        d['Due_to_Inv'] = pattern
                    elif p[0] == 'I':
                        d['Inv_to_crtn'] = pattern
                    elif p[0] == 'A':
                        d['Amount'] = pattern
                    elif p[0] == 'Not' :
                        d['CV_status'] = pattern
                    elif p[0] == 'CV' :
                        d['CV_status'] = pattern
                    else:
                        d['Vendor name'] = pattern
                d['support'] = round(item[1], 3)
                df_patterns = df_patterns.append(d, ignore_index=True)

        #print len(df_patterns) , "patterns found"
        print list_of_vendors.index(ven) ,"Vendors done"
    df_patterns = df_patterns.fillna(' ')
    df_patterns.to_csv(filename)
    return df_patterns

data_patterns = data_apriori.iloc[:,[1,3,6,7,8,9,10]]
df_500 = data_patterns[data_patterns['Vendor Name'].isin(list_of_vendors)]
df_500['Critical vendor'] = df_500['Critical vendor'].astype(str)
df_500 = df_500.replace(to_replace = ['True','False'] , value = ['CV','Not_CV'])

patterns_500 = freq_patterns(df_500,list_of_vendors,'Delay','Delay_patterns_500.csv')

patterns_160 = freq_patterns(df_160,list_of_vendors,'OnTime','OnTime_patterns_160.csv')

patterns_500.to_csv('patterns_500.csv')


""" Company code wise """


""" Counting vendorwise """

company_df = data_apriori.groupby('Companycode')['InvStatus'].value_counts().unstack(fill_value=0)
company_df['Total_invoices'] = company_df.Delay + company_df.OnTime

company_df['delay_percentage'] = (company_df.Delay / company_df.Total_invoices) * 100

company_df['Percentage_in_total_delays'] = (company_df.Delay / sum(company_df.Delay) ) * 100

company_df = company_df.round(decimals = 4)

company_df.sort_values(by = 'Percentage_in_total_delays' , ascending = False , inplace = True )

company_df.to_csv('company.csv')

list_of_vendors = list(vendor_df.index[:500])


""" Vendor wise data frame with support """


def freq_patterns(df,list_of_vendors,status,filename) :
    df = df[df['InvStatus'] == status ]
    df_patterns = pd.DataFrame(columns=['Vendor name', 'Amount','Inv_to_crtn', 'Due_to_Inv', 'CV_status', 'PO_item', 'support'])
    for ven in list_of_vendors :
        df_ven = df[df['Vendor Name'] == ven]
        df_ven.to_csv('data.csv', index=False, header=False)
        items, rules = runApriori(dataFromFile('data.csv'), 0.6, 0.4)
        vendor_patterns = []
        for item in items:
            if len(set(list(item[0])).intersection([ven])) != 0:
                vendor_patterns.append(item)

        #print len(vendor_patterns)

        for item in vendor_patterns:
            patterns = list(item[0])
            d = {}
            if len(patterns) > 3:
                for pattern in patterns:
                    p = pattern.split('_')
                    if p[0] == 'P':
                        d['PO_item'] = pattern
                    elif p[0] == 'D':
                        d['Due_to_Inv'] = pattern
                    elif p[0] == 'I':
                        d['Inv_to_crtn'] = pattern
                    elif p[0] == 'A':
                        d['Amount'] = pattern
                    elif p[0] == 'Not' :
                        d['CV_status'] = pattern
                    elif p[0] == 'CV' :
                        d['CV_status'] = pattern
                    else:
                        d['Vendor name'] = pattern
                d['support'] = round(item[1], 3)
                df_patterns = df_patterns.append(d, ignore_index=True)

        #print len(df_patterns) , "patterns found"
        print list_of_vendors.index(ven) ,"Vendors done"
    df_patterns = df_patterns.fillna(' ')
    df_patterns.to_csv(filename)
    return df_patterns

data_patterns = data_apriori.iloc[:,[1,3,6,7,8,9,10]]
df_500 = data_patterns[data_patterns['Vendor Name'].isin(list_of_vendors)]
df_500['Critical vendor'] = df_500['Critical vendor'].astype(str)
df_500 = df_500.replace(to_replace = ['True','False'] , value = ['CV','Not_CV'])

patterns_500 = freq_patterns(df_500,list_of_vendors,'Delay','Delay_patterns_500.csv')


patterns_refined = patterns_500[patterns_500['support'] > 0.6 ]

d_very_low = patterns_refined.loc[(patterns_refined['Inv_to_crtn']=='I_very_low') | (patterns_refined['Inv_to_crtn']=='I_low')]


d_very_low = d_very_low.loc[(patterns_refined['Due_to_Inv']=='D_very_low')]


df_dvl = d_very_low.groupby(['Vendor name'], sort=False)['support'].max().reset_index()

df_dvl = df_dvl[df_dvl['Vendor name'] != 'Delay']



df_dvl['total_delays'] = df_dvl['Vendor name'].map(vendor_df.Delay)

df_dvl['impacted_invoices'] = (df_dvl['support'] * df_dvl['total_delays']).round(decimals = 0)

df_dvl.to_csv('dvl_ivl.csv')


due_very_low = pd.read_csv('dvl.csv')

dvl_vendors = list(due_very_low['Vendor name'])

""" general patterns """


data_gen = data_apriori.iloc[:,[8,11,12]]


data_gen.to_csv('data_gen.csv',index = False ,header = False)

# Execute the file 'apriori.py' before running this step

items,rules = runApriori(dataFromFile('data_gen.csv'),0.2,0.2)