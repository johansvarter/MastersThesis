import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_exp = pd.read_csv('Final_cleaned_07APR.csv', parse_dates = ['saleDate_b'])

data_exp['mhtlExpens_b'] = data_exp['mhtlExpens_b']*1000

print(data_exp['Address_b'].value_counts().to_frame().value_counts().to_latex(index = True))
# The two below has been sold 'too many times' and are deleted.
# Vigerslev Allé 198, 2500 Valby                         44
# Willy Brandts Vej 1, st. tv, 2450 København SV         9
data_exp = data_exp[~(data_exp['Address_b'].isin(['Vigerslev Allé 198, 2500 Valby', 'Willy Brandts Vej 1, st. tv, 2450 København SV']))]
data_exp['saleYear_b'] = data_exp['saleDate_b'].astype(str).str[:4].astype(int)
data_exp['sqmPrice_bd'] = data_exp['price_b']/data_exp['areaWeighted_bd']

### PRICE HISTOGRAM
data_exp.hist(column = 'price_b', bins = 325, figsize = (12,7), alpha = 0.375, range = (0, 30000000))
plt.title('Histogram of sale prices (variable: price_b)')
plt.ylabel('Count')
plt.xlabel('Price in 10 mil. DKK')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/price_hist.png')
plt.show()
round(data_exp['price_b'].mean()) # 3.967.889
round(data_exp['price_b'].median()) # 3.400.000

#sale_prices = round(data_exp.groupby(['saleMY', 'itemTypeName_b']).agg(price_mean = ('price_b', np.mean),
#                                                                 sqmPrice_mean = ('sqmPrice_bd', np.mean)))

### NUMBER OF SALES -
data_exp['quarter_b'] = data_exp['saleDate_b'].dt.to_period("Q")
data_exp['quarter_b'] = data_exp['quarter_b'].astype(str)

quarters = ['2017Q1','2017Q2','2017Q3','2017Q4',
            '2018Q1','2018Q2','2018Q3','2018Q4',
            '2019Q1','2019Q2','2019Q3','2019Q4',
            '2020Q1','2020Q2','2020Q3','2020Q4',
            '2021Q1','2021Q2','2021Q3','2021Q4']


total = data_exp['quarter_b'].value_counts()
total = total.reindex(index = quarters)
apartments = data_exp.loc[data_exp['itemTypeName_b'] == 'Ejerlejlighed','quarter_b'].value_counts()
apartments = apartments.reindex(index = quarters)

plt.rcParams["figure.figsize"] = (15,7)
fig, ax = plt.subplots()

plt.plot(total, c ='steelblue', linewidth=3, label='Total')
plt.plot(apartments, c ='tomato', linewidth=3, label='Apartments')

fig.autofmt_xdate()
plt.ylabel('Sales')
plt.legend(fontsize=15)
plt.rc('axes', labelsize=20)
xticks = ax.xaxis.get_major_ticks()
for i,tick in enumerate(xticks):
    if i%2 != 0:
        tick.label1.set_visible(False)
plt.title('Number of sales for each quarter. Cph. and Frederiksberg')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/noOfSales.png')
plt.show()
# PLOT OK
# TENDENCIES ARE VERY DIFFERENT FROM MADSENS - OBVIOUSLY

### SQUARE METER PRICE
m2price = data_exp.groupby('quarter_b')['sqmPrice_bd'].mean()
plt.rcParams["figure.figsize"] = (15,7)
fig, ax = plt.subplots()

plt.plot(m2price, c ='steelblue', linewidth=3)


fig.autofmt_xdate()
plt.ylabel('sqmPrice_bd')
#plt.legend(fontsize=15)
plt.title('Square meter price in Cph. and Frederiksberg 2017-2021')
plt.rc('axes', labelsize=20)
xticks = ax.xaxis.get_major_ticks()
for i,tick in enumerate(xticks):
    if i%2 != 0:
        tick.label1.set_visible(False)
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/sqmPrice1721.png')
plt.show()
# PLOT OK - USE

# SQM PRICE PR CITY OVER TIME
import matplotlib
matplotlib.style.use('ggplot')

data_exp.loc[~(data_exp['city_b']=='Kastrup'), ['quarter_b', 'city_b', 'sqmPrice_bd']].groupby(['quarter_b', 'city_b']).mean().unstack().plot(figsize = (15, 7), legend = False)
plt.legend(['Brønshøj','Frb.','Frb. C','Hellerup','Cph. K','Cph. N','Cph. NV','Cph. S','Cph. SV','Cph. V','Cph. Ø','Nordhavn','Rødøvre','Valby','Vanløse'],
           ncol = 2, title = 'Part of city')
plt.ylabel('sqmPrice_bd')
plt.xlabel('')
plt.title('Square meter price in each part of city. 2017-2021')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/sqmPrice1721_by_city_b.png')
plt.show() # OK - BUT USELESS??? BIT MESSY BUT MAYBE STILL GOOD. KASTRUP MUST BE REMOVED

### TIL I MORGEN - INDSÆT OVENSTÅENDE PLOTS EVT. M. SMÅ ÆNDRINGER
### LAV KORTPLOTS - HØR FUGLSANG OM TRICK

### MAP PLOTS

BBox = ((12.4509, 12.6401, 55.6154, 55.7321)) #png6

background_map = plt.imread('map_6.png') #963x1054 BxH 0.913662
# gange to: 1926x2108

### SQUARE METER PRICE
px = 1/plt.rcParams['figure.dpi']
# ALT SHIFT - PIL OP NED
fig, ax = plt.subplots(figsize = (1063*px,1154*px))
cbar = fig.colorbar(ax.scatter(data_exp['longitude_b'], data_exp['latitude_b'],
           zorder=1, alpha= 0.6, c=data_exp['sqmPrice_bd'], cmap = plt.get_cmap('jet'), s=10,
                               vmin = 0, vmax = 105000),
                    ticks = [0, 25000, 50000, 75000, 100000])
cbar.ax.set_yticklabels(['0', '25000', '50000', '75000', '100000'])
cbar.ax.set_ylabel('In DKK', fontsize = 12)
ax.set_title('Square meter prices in Copenhagen. 2017-2021')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(background_map, zorder=0, extent = BBox, aspect = 'auto')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/map_sqmprice_1.png', dpi = 500)
plt.show()

### EXPECTATIONS
FORV2 = pd.read_csv('FORV2-plot.csv', sep = ";", decimal = ",")
FORV2 = FORV2[:20]

FORV2.plot(x = 'Kvartal', y = 'Index', ylim = [-95,-50], xlabel = 'Quarter',
           title = 'Expectations regarding purchase of house, StatBank (table: FORV2)',
           ylabel = 'Score')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/FORV2_plot.png')
plt.show()

##### SQUARE METER PRICE OUTLIER --- BOXPLOT
plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=data_exp["sqmPrice_bd"])
plt.axvline(x = (data_exp['sqmPrice_bd'].mean())/4, linestyle = "--", alpha = 0.25, c = 'red')
plt.axvline(x = (data_exp['sqmPrice_bd'].mean())*2.75, linestyle = "--", alpha = 0.25, c = 'red')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/sqmPrice_boxplot.png')
plt.show()

##### PRICE VS AREA --- OUTLIER VISUALIZATION

plt.rcParams["figure.figsize"] = (12,8)

fig, ax = plt.subplots()

# Plot data
plt.scatter(data_exp['areaWeighted_bd'], data_exp['price_b'],
            edgecolors='none', alpha=0.1, c ='darkblue', s = 75)

# Change fontsize
#plt.rcParams['font.size'] = 18
plt.xlabel('areaWeighted_bd')
plt.ylabel('salePrice_b')
plt.show()

### WITH EXCLUSIONS
mean = data_exp['sqmPrice_bd'].mean() #42754
len(data_exp.loc[(data_exp['sqmPrice_bd']> 2.75*mean) | (data_exp['sqmPrice_bd']< (1/4)*mean),:]) #209
outliers = data_exp.loc[(data_exp['sqmPrice_bd']> 2.75*mean) | (data_exp['sqmPrice_bd']< (1/4)*mean),:]
data_exp_1 = data_exp.loc[~data_exp['Address_b'].isin(outliers['Address_b']),:]

plt.rcParams["figure.figsize"] = (12,8)

fig, ax = plt.subplots()

# Plot data
plt.scatter(data_exp['areaWeighted_bd'], data_exp['price_b'],
            edgecolors='none', alpha=0.1, c ='darkblue', s = 75)
plt.scatter(outliers['areaWeighted_bd'], outliers['price_b'],
            edgecolors='none', alpha=0.1, c ='red', s = 75)

x = np.array([0, 100, 200, 300, 400, 500])
plt.plot(x, (1/4)*mean*x, c ='steelblue')
plt.plot(x, 2.75*mean*x, c ='steelblue')

plt.rc('axes', labelsize=20)

# Change fontsize
#plt.rcParams['font.size'] = 18
plt.xlabel('areaWeighted_bd', fontsize = 14)
plt.ylabel('price_b. Price in 10 mil. DKK', fontsize = 14)
plt.title('All observations. Red observations are excluded as outliers.', fontsize = 20)
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/outlier_exclusion.png')
plt.show()

data_exp = data_exp_1

pd.options.display.max_columns = 50
#pd.set_option('display.float_format', lambda x: '%.5f' % x)
#pd.reset_option('display.float_format')
data_exp.select_dtypes(include=np.number).describe().transpose()

### MONTHLY EXPENSES
plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=data_exp["mhtlExpens_b"])
plt.show()
## Several severe outliers - above 25k DKK it is set missing
data_exp["mhtlExpens_b"].nlargest(25)
data_exp.loc[data_exp["mhtlExpens_b"]>25000, "mhtlExpens_b"] = np.nan

### NO OF ROOMS - 45111 -> 41481
# OBS. W. 0 ARE DELETED AND SO ARE THE ONE WITH 50
data_exp = data_exp.loc[data_exp['numberOfRooms_bd'] != 0]
data_exp = data_exp.loc[data_exp['numberOfRooms_bd'] < 20]

### PROPERTY VALUATION
plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=data_exp["propValuation_b"])
plt.show()
data_exp.loc[data_exp["propValuation_b"]>75000000, "propValuation_b"] = np.nan

### BUILD YEAR
data_exp['buildYear_b'].nlargest(25)
data_exp['buildYear_b'].nsmallest(25)
data_exp = data_exp.loc[data_exp['buildYear_b'] > 1500]
data_exp = data_exp.loc[data_exp['buildYear_b'] < 2022]

### NO OF TOILETS
data_exp['noToilets_d'].nlargest(15)
data_exp['noToilets_d'].nsmallest(15)
data_exp = data_exp.loc[data_exp['noToilets_d'] > -1]
data_exp = data_exp.loc[data_exp['noToilets_d'] < 20]

### REBUILD YEAR
data_exp['rebuildYear_bd'].nsmallest(15)

### BASEMENT AREA
data_exp.loc[data_exp['itemTypeName_b'] == 'Ejerlejlighed', 'areaBasement_bd'] = 0

### PARCEL AREA
data_exp.loc[data_exp['itemTypeName_b'] == 'Ejerlejlighed', 'parcelArea_b'] = 0
# ERROR IN READING OF DATA. ONLY FOR OBS. W. OVER 1000 SQM. WRONG DEC.SEP. SO MULT. W. 1000.
data_exp.loc[(data_exp['parcelArea_b'] > 0) & (data_exp['parcelArea_b'] < 3) , 'parcelArea_b'] = \
    data_exp.loc[(data_exp['parcelArea_b'] > 0) & (data_exp['parcelArea_b'] < 3) , 'parcelArea_b']*1000
data_exp['parcelArea_b'].nlargest(40)
data_exp.loc[data_exp['parcelArea_b']%1>0, 'parcelArea_b']

### SALES PERIOD - DATA ERROR CORRECTION SAME AS PARCEL AREA
data_exp.loc[data_exp['salesPeriod_b'] > 1, 'salesPeriod_b'].nsmallest(50)
data_exp.loc[(data_exp['salesPeriod_b'] > 1) & (data_exp['salesPeriod_b'] < 2), 'salesPeriod_b'] = \
    data_exp.loc[(data_exp['salesPeriod_b'] > 1) & (data_exp['salesPeriod_b'] < 2), 'salesPeriod_b']*1000
data_exp['salesPeriod_b'].nlargest(50)

####################
### CORRELATIONS ###
####################

numerics = data_exp.select_dtypes(include=['float64', 'int64', 'int32'])
numerics = numerics.drop(columns = ['postalId_b'])

## Heat map. Madsens code
cm = numerics.corr(method='pearson')
hm = sns.heatmap(cm,  vmax=0.9, center=0, xticklabels=True, yticklabels=True,
            square=True, linewidths=.1, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
sns.despine()
hm.figure.set_size_inches(20,20)
# to be dropped or changed
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/heat-map.png')
plt.show()

#correlation list
corr_num_list = (cm.where(np.triu(np.ones(cm.shape), k=1).astype(np.bool))
                      .stack().sort_values(ascending=False))
corr_num_list.head(35)
# Looks like the AVM model is very area dependent. Not so strange maybe.

# correlation list tail
corr_num_list.tail(20)

round(numerics.corrwith(numerics['price_b']).sort_values(ascending=False).head(15), 2)
round(numerics.corrwith(numerics['price_b']).sort_values(ascending=False).tail(5), 2)

### PAIR PLOT

sns.pairplot(data=numerics,
                  y_vars=['price_b'],
                  x_vars=['buildYear_b', 'sqmPrice_bd', 'mhtlExpens_b', 'areaResidential_bd'])
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/pair_plot_1.png')
sns.pairplot(data=numerics,
                  y_vars=['price_b'],
                  x_vars=[ 'areaWeighted_bd', 'propValuation_b', 'numberOfRooms_bd', 'noToilets_d'])
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/pair_plot_2.png')
sns.pairplot(data=numerics,
                  y_vars=['price_b'],
                  x_vars=['parcelArea_b', 'areaBasement_bd', 'voterTurnout_d', 'coast_h'])
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/pair_plot_3.png')
plt.show()

#

most_corr = ['price_b', 'propValuation_b', 'saleYear_b',
             'buildYear_b', 'areaResidential_bd', 'mhtlExpens_b',
             'longitude_b', 'sqmPrice_bd', 'parcelArea_b',
             'rebuildYear_bd', 'areaBasement_bd', 'areaWeighted_bd',
             'numberOfRooms_bd', 'MAS_d', 'AVM_price_d',
             'voterTurnout_d', 'strain_h', 'airport_h', 'coast_h',
             'assets_f', 'arbløshed_s', 'rate_s', 'dst_forv_var_s', 'latitude_b']

plt.rcParams['font.size'] = 12

# Heast map most correlated
cm = numerics[most_corr].corr(method='pearson')

hm = sns.heatmap(cm,  vmax=0.9, center=0, xticklabels=True, yticklabels=True,
            square=True, linewidths=.1, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
sns.despine()

hm.figure.set_size_inches(17,12)
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/heat_map_most_corr.png')
plt.show()

#end_of_numeric = data_exp
#############################
### CATEGORICAL VARIABLES ###
#############################

nonnumerics = data_exp.select_dtypes(exclude=['int32', 'float64', 'int64'])
#nonnumerics.dtypes()

data_exp['Address_b'] = data_exp['Address_b'].astype('category')
# Address not used for prediction but it (uniquely, together with a date,) identifies an observation


## Categorizing city and checking variance
data_exp['city_b'] = data_exp['city_b'].astype('category')
sns.countplot(x='city_b', data=data_exp)
plt.xticks(rotation=45)
plt.show()
# Not enough observations in Hellerup, Kastrup and Rødovre
data_exp = data_exp.loc[~data_exp['city_b'].isin(['Hellerup', 'Kastrup', 'Rødovre'])]
# We remove these 336 observations and end at 41069 observations


### Postal ID
data_exp['postalId_b'] = data_exp['postalId_b'].astype('int')
# Handling Christianshavn
data_exp.loc[(data_exp['postalId_b'].astype('int') >= 1400) & (data_exp['postalId_b'].astype('int') <= 1448), ['postalId_b']] = 666
data_exp['postalId_b'] = data_exp['postalId_b'].astype('category')


bins = [0,      999,      1500,        1800,        2000,
        2001,   2101,  2151,  2201,  2301,   2401,  2451,   2501,
        2701,  2721] #16
names = ['1400' ,'<1500xC', '1500-1800', '1800-1999', '2000',
         '2100','2150','2200','2300', '2400', '2450',
         '2500', '2700', '2720'] #15
# <1500xC means zip codes up to 1500 but eXclusive (C)hristianshavn which has 1400-1448

d = dict(enumerate(names,1))
data_exp['postalId_b_range'] = np.vectorize(d.get)(np.digitize(data_exp['postalId_b'],bins))
data_exp['postalId_b_range'] = data_exp.postalId_b_range.astype('category')
data_exp['postalId_b_range'].value_counts()
sns.countplot(x = 'postalId_b_range', data = data_exp)
plt.show()


data_exp['itemTypeName_b'] = data_exp['itemTypeName_b'].astype('category')
data_exp['itemTypeName_b'].value_counts()
# VARIATION OK

data_exp['usageType_d'].value_counts()
data_exp = data_exp.loc[~data_exp['usageType_d'].isin(['Kollegiebolig', 'Anden enhed til helårsbeboelse', '0'])]

# Kollegiebolig, Anden enhed til helårsbebeolse and 0 are deleted due to missing variation and
# because 'student housing' should not count here so they are removed. Removing these 56 obs.
# brings us down to 41405 observations

nonnumerics.columns
##       'saleDate_b', 'Address_b', 'city_b', 'itemTypeName_b', 'usageType_d', ALL CHECKED
##       'kitchenType_d', 'outwMaterial_d', 'roofType_d', 'heatType_d', ALL CHECKED
##       'energyMark_b', 'radonRisk_d', 'noiseLvl_d', 'floodRisk_d', ALL CHECKED
##       'biggestParty_d', 'votingArea_d', 'indicatorSalesP_own', 'saleMY',
##       'quarter_b']


## Categorizing kitchenType_d and checking variance
data_exp['kitchenType_d'] = data_exp['kitchenType_d'].astype('category')
sns.countplot(x='kitchenType_d', data=data_exp)
plt.xticks(rotation=12)
plt.title('Distribution of kitchen type')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/kitchen_Type_varianceExample.png')
plt.show()
# Not enough variation - VARIABLE IS DROPPED
data_exp = data_exp.drop(columns = ['kitchenType_d'])

## Categorizing outwMaterial_d and checking variance
data_exp['outwMaterial_d'] = data_exp['outwMaterial_d'].astype('category')

sns.countplot(x='outwMaterial_d', data=data_exp)
plt.xticks(rotation=12)
plt.title('Distribution of outer wall material')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/outerwallMaterial_varianceExample.png')
plt.show()
# Around 35000 is 'Mursten', around 4000 'Betonelementer' and (much) below ~1000 for the rest. We drop the variable
# due to the variance being too small. Assumably the category 'Betonelementer' will be closely correlated to build Year
data_exp = data_exp.drop(columns = ['outwMaterial_d'])


## Categorizing roofType_d and checking variance
data_exp['roofType_d'] = data_exp['roofType_d'].astype('category')
data_exp['roofType_d'].value_counts()

def new_roof(x):
    if x in ('Tagpap med lille hældning', 'Tagpap med stor hældning'):
        return 'Tagpap'
    if x in ('Betontagsten', 'Metal', 'Levende tage', 'Glas', 'Stråtag'):
        return 'Andet materiale'
    if x in ('Fibercement herunder asbest', 'Fibercement uden asbest'):
        return 'Fibercement'
    else:
        return x

data_exp['roofType_d'].apply(new_roof).value_counts()

sns.countplot(x='roofType_d', data=data_exp)
plt.xticks(rotation=25)
plt.title('Distribution of roof type')
plt.show()
# VARIANCE OKAY BUT SOME CATEGORIES HAVE VERY LOW NO OF OBS. WE LEAVE IT AS IT IS

## Categorizing heatType_d and checking variance
data_exp['heatType_d'] = data_exp['heatType_d'].astype('category')

sns.countplot(x='heatType_d', data=data_exp)
plt.xticks(rotation=12)
plt.title('Distribution of heat type')
plt.show()
# As almost all housing in Copenhagen is heated by district heating this variable is deleted.
# However it could be relevant outside copenhagen where a larger share are heated by gas etc.
# Especially after the very rapid rising gas prices.
data_exp = data_exp.drop(columns = ['heatType_d'])


## Categorizing energyMark_b and checking variance
data_exp['energyMark_b'] = data_exp['energyMark_b'].astype('str')

sns.countplot(x='energyMark_b', data=data_exp)
plt.title('Distribution of energy mark')
plt.show()
# Variance mostly OK. The A-category should maybe be made into one category.
data_exp['energyMark_b'].value_counts()

data_exp.loc[data_exp['energyMark_b'] == 'a1', 'energyMark_b'] = 'a'
data_exp.loc[data_exp['energyMark_b'] == 'a2', 'energyMark_b'] = 'a'
data_exp.loc[data_exp['energyMark_b'] == 'a2010', 'energyMark_b'] = 'a'
data_exp.loc[data_exp['energyMark_b'] == 'a2015', 'energyMark_b'] = 'a'
data_exp.loc[data_exp['energyMark_b'] == 'a2020', 'energyMark_b'] = 'a'

ordered_Energy = ['a','b','c','d','e','f','g']
data_exp['energyMark_b'] = data_exp['energyMark_b'].astype('category').cat.set_categories(ordered_Energy,
                                                                                          ordered = True)
data_exp['energyMark_b'].value_counts()
### ORDERED CATEGORIES OK

## Categorizing radonRisk_d and checking variance
data_exp['radonRisk_d'] = data_exp['radonRisk_d'].astype('category')

sns.countplot(x='radonRisk_d', data=data_exp)
plt.xticks(rotation=12)
plt.title('Distribution of radon risk')
plt.show()
# The big majority has 'very low risk' so this variable is dropped
data_exp = data_exp.drop(columns = ['radonRisk_d'])



## Categorizing noiseLvl_d and checking variance
data_exp['noiseLvl_d'] = data_exp['noiseLvl_d'].astype('category')

sns.countplot(x='noiseLvl_d', data=data_exp)
plt.xticks(rotation=12)
plt.title('Distribution of noise lvl')
plt.show()
data_exp['noiseLvl_d'].value_counts()
# Something have to be done. Maybe too few over 75dB but it is important. What about missing?
data_exp.loc[data_exp['noiseLvl_d'] == "Mangler", "noiseLvl_d"] = "Ingen trafikstøj"
ordered_noise = ['Ingen trafikstøj', '55-60 dB','60-65 dB','65-70 dB','70-75 dB','over 75 dB']
data_exp['noiseLvl_d'] = data_exp['noiseLvl_d'].astype('category').cat.set_categories(ordered_noise, ordered = True)
data_exp['noiseLvl_d'].value_counts()


## Categorizing floodRisk_d and checking variance
data_exp['floodRisk_d'] = data_exp['floodRisk_d'].astype('category')

sns.countplot(x='floodRisk_d', data=data_exp)
plt.xticks(rotation=12)
plt.title('Distribution of flood risk')
plt.show()
# Too little variance. It is dropped. People are not rational here - flood risk should be taken more serious.
data_exp = data_exp.drop(columns = ['floodRisk_d'])



## Categorizing biggestParty_d and checking variance
data_exp['biggestParty_d'] = data_exp['biggestParty_d'].astype('category')

sns.countplot(x='biggestParty_d', data=data_exp)
plt.xticks(rotation=12)
plt.title('Distribution of biggest party')
plt.show()
#  VARIANCE OK


#  'votingArea_d', 'indicatorSalesP_own', 'saleMY',
#         'quarter_b']

## Categorizing votingArea_d and checking variance
data_exp['votingArea_d'] = data_exp['votingArea_d'].astype('category')

sns.countplot(x='votingArea_d', data=data_exp)
plt.xticks(rotation=90)
plt.title('Distribution of voting area')
plt.show()
# VARIANCE OK



### ORDINAL CATEGORICAL VARIABLES
# Using the same mapping as Madsen
energy_mapping = {'a': 1,'b': 2,'c': 3,'d': 4,'e': 5,'f': 6 ,'g': 7}
data_exp['energyMark_b'] = data_exp['energyMark_b'].map(energy_mapping)

noise_mapping = {'Ingen trafikstøj': 1, '55-60 dB': 2,'60-65 dB': 3,'65-70 dB': 4,'70-75 dB': 5,'over 75 dB': 6}
data_exp['noiseLvl_d'] = data_exp['noiseLvl_d'].map(noise_mapping)

quarter_mapping = {'Q1': 1,'Q2': 2,'Q3': 3,'Q4': 4} #not used but gives intuition
data_exp['quarter_numeric'] = data_exp['quarter_b'].astype(str).str[5:]
data_exp['quarter_onehot'] = data_exp['quarter_b'].astype(str).str[4:]
data_exp.info() #58 var (quarter_numeric variable er ligemeget og tæller ikke. Burde droppes)

### ONE HOT ENCODING
# Used for postal-range, biggest party, roof type
data_exp['postalId_b_range'] = data_exp['postalId_b_range'].cat.remove_unused_categories()

finalData = pd.get_dummies(data_exp, columns = ['postalId_b_range', 'biggestParty_d', 'roofType_d',
                                               'itemTypeName_b', 'usageType_d', 'quarter_onehot'], drop_first = True)
finalData.head()
finalData.columns
finalData.info()

finalData.corrwith(finalData['price_b']).sort_values(ascending=False)

cm = finalData.corr(method='pearson')

corr_all_list = (cm.where(np.triu(np.ones(cm.shape), k=1).astype(np.bool))
                      .stack().sort_values(ascending=False))
corr_all_list.head(50)
corr_all_list.tail(20)

print(pd.crosstab(data_exp['itemTypeName_b'], data_exp['usageType_d'],
                  margins = False).to_latex(index=True))
finalData.drop(columns=['usageType_d_Dobbelthus', 'usageType_d_Fritliggende enfamiliehus',
       'usageType_d_Række-, kæde- eller dobbelthus (lodret adskillelse mellem enhederne).',
       'usageType_d_Række-, kæde- og klyngehus'], inplace=True)

most_corr_1 = ['price_b',  'areaResidential_bd',  'parcelArea_b',
               'numberOfRooms_bd', 'areaWeighted_bd', 'propValuation_b',
               'mhtlExpens_b', 'strain_h',
               'areaBasement_bd', 'coast_h', 'assets_f',
               'usageType_d_Fritliggende enfamiliehus', 'itemTypeName_b_Villa', 'AVM_price_d',
               'postalId_b_range_2300', 'MAS_d', 'saleYear_b', 'airport_h']

plt.rcParams['font.size'] = 12

# Heast map most correlated
cm = finalData[most_corr_1].corr(method='pearson')

hm = sns.heatmap(cm,  vmax=0.9, center=0, xticklabels=True, yticklabels=True,
            square=True, linewidths=.1, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
sns.despine()

hm.figure.set_size_inches(17,12)
plt.title('Heat map with one hot encoded variables')
#plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/heat_map_onehot.png')
plt.show()


finalData.to_csv('FinalData.csv')

## IMPUTING MONTHLY EXPENSES
def kvm_grp(x):
    if x <= 35:
        return '<35'
    if x <= 45:
        return '35-45'
    if x <= 55:
        return '45-55'
    if x <= 65:
        return '55-65'
    if x <= 75:
        return '65-75'
    if x <= 85:
        return '75-85'
    if x <= 95:
        return '85-95'
    if x <= 110:
        return '95-110'
    if x <= 130:
        return '110-130'
    if x <= 150:
        return '130-150'
    if x <= 170:
        return '150-170'
    if x <= 190:
        return '170-190'
