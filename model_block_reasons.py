

os.chdir('/home/prudhvi/Documents/DSM')

data_initial = pd.read_csv('DNPClean.csv')

kk = data_initial.groupby(['Block Status','InvStatus']).size()