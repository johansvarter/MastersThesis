import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
#2021: 7939
#2020: 8778
#2019: 7647
#2018: 8171
#2017: 9401
#Frederiksberg: 6102

raw_data = pd.read_csv("raw_data_1.csv")
raw_data.shape # (48038, 201)
all_former_sales = pd.read_csv('final_hist_prices.csv')
raw_data.info(verbose = True)
list_drop = ['address.gstKvhx', 'address.municipalityNumber', 'address.oisPropertyNumber',
                 'address.itemType', 'address.itemTypeNumber', 'address.mapPosition.hasCoordinates',
                 'address.wishPropertyLocationLink', 'address.hasEnergyMark', 'address.energyMarkLink',
                 'address.environmentData.soilContamination', 'address.environmentData.serviceStatus.renewTicket',
                 'address.environmentData.serviceStatus.errorCode', 'address.environmentData.serviceStatus.errorText',
                 'address.environmentData.serviceStatus.errorId', 'address.latestValuation.farmhouseParcelValuation',
                 'address.latestValuation.farmhousePropertyValuation', 'address.latestSale.saleTypeId',
                 'address.latestForSale.id', 'address.latestForSale.addressId', 'address.latestForSale.isArchive',
                 'address.latestForSale.uniqueNumber', 'address.latestForSale.description',
                 'address.latestForSale.descriptionHeadline', 'address.latestForSale.itemType',
                 'address.latestForSale.itemTypeNumber', 'address.latestForSale.marketingItemType',
                 'address.latestForSale.placeName', 'address.latestForSale.placeNameSeparator',
                 'address.latestForSale.imageLink600X400',
                 'address.latestForSale.canShowSalesPeriodTotal', 'address.latestForSale.areaWeightedAsterix',
                 'address.latestForSale.areaWeightedTitleMessage', 'address.latestForSale.areaWeightedKrM2Title',
                 'address.latestForSale.agentChainName', 'address.latestForSale.agentId',
                 'address.latestForSale.agentsLogoLink', 'address.latestForSale.propertyLink',
                 'address.latestForSale.redirectLink', 'address.latestForSale.memberOfDe',
                 'address.latestForSale.hasEnergyMark', 'address.latestForSale.energyMarkLink',
                 'address.latestForSale.hasOpenHouse', 'address.latestForSale.nextOpenHouse',
                 'address.latestForSale.nextOpenHouseShort', 'address.latestForSale.nextOpenHouseSignup',
                 'address.latestForSale.municipalityNumber', 'address.latestForSale.oisPropertyNumber',
                 'address.latestForSale.isFavorite', 'address.latestForSale.hasComment',
                 'address.latestForSale.comment', 'address.latestForSale.rentalLink',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData',
                 'address.latestForSale.mapPosition.hasCoordinates', 'address.latestForSale.videoRedirectLink',
                 'address.latestForSale.openHouseRedirectLink', 'address.latestForSale.projectSale',
                 'address.latestForSale.kvhx', 'address.latestForSale.gstKvhx',
                 'address.latestForSale.wishPropertyLocationLink', 'address.latestForSale.hasAreaWeighted',
                 'address.latestForSale.hasRentalLink', 'address.latestForSale.hasVideoLink',
                 'address.latestForSale.oisHidden', 'address.latestForSale.nextOpenHouseTime',
                 'address.latestForSale.calculateLoanAgentChain', 'address.latestForSale.label',
                 'dingeo_link', 'address.latestForSale.propertyPartiallyOwnedFinancialData.purchasePrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.maximumPriceRatio',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.maximumPrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.housingAssociationDebtShare',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.housingAssociationDebt',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.financingInformation',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.expenseNet',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.expenseGross',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.estimatedTechnicalPrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.estimatedTechnicalAreaPrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.downPayment',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.distributionRatio',
                 'address.latestForSale.rating.ratings.conditionRating',
                 'address.latestForSale.rating.ratings.kitchenRating',
                 'address.latestForSale.rating.ratings.locationRating',
                 'address.latestForSale.rating.ratings.bathRating', 'address.latestForSale.rating.averageRating',
                 'address.latestForSale.rating.roundAverageRating', 'address.latestForSale',
                 'address.latestForSale.mapPosition.latLng', 'address.latestValuation',
                 'address.environmentData.breakInStatistic', 'address.mapPosition.latLng',
                 'address.latestSale', 'Location']
df_slim_1 = raw_data.drop(columns = list_drop)
df_slim_1.info(verbose = True)
all_former_sales.info(verbose = True)


def remove_variables(df):

    raw = df

    list_drop = ['address.gstKvhx', 'address.municipalityNumber', 'address.oisPropertyNumber',
                 'address.itemType', 'address.itemTypeNumber', 'address.mapPosition.hasCoordinates',
                 'address.wishPropertyLocationLink', 'address.hasEnergyMark', 'address.energyMarkLink',
                 'address.environmentData.soilContamination', 'address.environmentData.serviceStatus.renewTicket',
                 'address.environmentData.serviceStatus.errorCode', 'address.environmentData.serviceStatus.errorText',
                 'address.environmentData.serviceStatus.errorId', 'address.latestValuation.farmhouseParcelValuation',
                 'address.latestValuation.farmhousePropertyValuation', 'address.latestSale.saleTypeId',
                 'address.latestForSale.id', 'address.latestForSale.addressId', 'address.latestForSale.isArchive',
                 'address.latestForSale.uniqueNumber', 'address.latestForSale.description',
                 'address.latestForSale.descriptionHeadline', 'address.latestForSale.itemType',
                 'address.latestForSale.itemTypeNumber', 'address.latestForSale.marketingItemType',
                 'address.latestForSale.placeName', 'address.latestForSale.placeNameSeparator',
                 'address.latestForSale.imageLink600X400',
                 'address.latestForSale.canShowSalesPeriodTotal', 'address.latestForSale.areaWeightedAsterix',
                 'address.latestForSale.areaWeightedTitleMessage', 'address.latestForSale.areaWeightedKrM2Title',
                 'address.latestForSale.agentChainName', 'address.latestForSale.agentId',
                 'address.latestForSale.agentsLogoLink', 'address.latestForSale.propertyLink',
                 'address.latestForSale.redirectLink', 'address.latestForSale.memberOfDe',
                 'address.latestForSale.hasEnergyMark', 'address.latestForSale.energyMarkLink',
                 'address.latestForSale.hasOpenHouse', 'address.latestForSale.nextOpenHouse',
                 'address.latestForSale.nextOpenHouseShort', 'address.latestForSale.nextOpenHouseSignup',
                 'address.latestForSale.municipalityNumber', 'address.latestForSale.oisPropertyNumber',
                 'address.latestForSale.isFavorite', 'address.latestForSale.hasComment',
                 'address.latestForSale.comment', 'address.latestForSale.rentalLink',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData',
                 'address.latestForSale.mapPosition.hasCoordinates', 'address.latestForSale.videoRedirectLink',
                 'address.latestForSale.openHouseRedirectLink', 'address.latestForSale.projectSale',
                 'address.latestForSale.kvhx', 'address.latestForSale.gstKvhx',
                 'address.latestForSale.wishPropertyLocationLink', 'address.latestForSale.hasAreaWeighted',
                 'address.latestForSale.hasRentalLink', 'address.latestForSale.hasVideoLink',
                 'address.latestForSale.oisHidden', 'address.latestForSale.nextOpenHouseTime',
                 'address.latestForSale.calculateLoanAgentChain', 'address.latestForSale.label',
                 'dingeo_link', 'address.latestForSale.propertyPartiallyOwnedFinancialData.purchasePrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.maximumPriceRatio',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.maximumPrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.housingAssociationDebtShare',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.housingAssociationDebt',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.financingInformation',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.expenseNet',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.expenseGross',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.estimatedTechnicalPrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.estimatedTechnicalAreaPrice',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.downPayment',
                 'address.latestForSale.propertyPartiallyOwnedFinancialData.distributionRatio',
                 'address.latestForSale.rating.ratings.conditionRating',
                 'address.latestForSale.rating.ratings.kitchenRating',
                 'address.latestForSale.rating.ratings.locationRating',
                 'address.latestForSale.rating.ratings.bathRating', 'address.latestForSale.rating.averageRating',
                 'address.latestForSale.rating.roundAverageRating', 'address.latestForSale',
                 'address.latestForSale.mapPosition.latLng', 'address.latestValuation',
                 'address.environmentData.breakInStatistic', 'address.mapPosition.latLng',
                 'address.latestSale', 'Location']

    list_keep = ['address.address', 'address.buildYear', 'address.rebuildYear', 'address.postalId', 'address.city',
                 'address.street', 'address.streetName', 'address.itemTypeName', 'address.mapPosition.latLng.lat',
                 'address.mapPosition.latLng.lng', 'address.energyMark', 'address.environmentData.radonRiskCategory',
                 'address.environmentData.drinkingWaterHardnessCategory',
                 'address.environmentData.breakInStatistic.municipalityAverage',
                 'address.environmentData.breakInStatistic.countyAverage',
                 'address.environmentData.breakInStatistic.countryAverage',
                 'address.environmentData.breakInStatistic.riskCategory', 'address.latestValuation.valuationYear',
                 'address.latestValuation.valuationDate', 'address.latestValuation.parcelValuation',
                 'address.latestValuation.propertyValuation', 'address.latestSale.saleDate',
                 'address.latestSale.salePrice', 'address.latestSale.saleType', 'address.latestSale.salesYear',
                 'address.latestForSale.buildYear', 'address.latestForSale.priceDevelopment',
                 'address.latestForSale.priceDevelopmentHistoric', 'address.latestForSale.propertyCharges',
                 'address.latestForSale.usageExpenses', 'address.latestForSale.paymentGross',
                 'address.latestForSale.paymentNet', 'address.latestForSale.paymentExpenses',
                 'address.latestForSale.itemTypeName', 'address.latestForSale.address',
                 'address.latestForSale.streetName', 'address.latestForSale.houseNumber',
                 'address.latestForSale.city', 'address.latestForSale.postal', 'address.latestForSale.paymentCash',
                 'address.latestForSale.downPayment', 'address.latestForSale.numberOfRooms',
                 'address.latestForSale.salesPeriod', 'address.latestForSale.salesPeriodTotal',
                 'address.latestForSale.area', 'address.latestForSale.areaResidential',
                 'address.latestForSale.areaParcel', 'address.latestForSale.areaBasement',
                 'address.latestForSale.areaWeighted', 'address.latestForSale.areaPaymentCash',
                 'address.latestForSale.numberOfFloors', 'address.latestForSale.floor',
                 'address.latestForSale.floorName', 'address.latestForSale.energyMark',
                 'address.latestForSale.dateAdded', 'address.latestForSale.dateRemoved',
                 'address.latestForSale.linkDomain', 'address.latestForSale.dateAnnounced',
                 'address.latestForSale.mapPosition.latLng.lat', 'address.latestForSale.mapPosition.latLng.lng',
                 'address.latestForSale.rebuildYear',
                 'Radonrisiko', 'Støjmåling', 'Oversvømmelsesrisiko_skybrud', 'Meter_over_havet', 'Anvendelse',
                 'Opførselsesår', 'Ombygningsår', 'Fredning', 'Køkkenforhold', 'Antal Etager', 'Antal toiletter',
                 'Antal badeværelser', 'Antal værelser', 'Ydervægsmateriale', 'Tagmateriale', 'Varmeinstallation',
                 'Bygning, Samlet areal', 'Boligstørrelse', 'Vægtet Areal', 'Energimærke', 'Indbrudsrisiko',
                 'Bevaringsværdig', 'Største_parti', 'Valgdeltagelse', 'Afstemningsområde', 'Kælder', 'Location',
                 'school', 'roadtrain', 'junction', 'daycare', 'metro', 'doctor', 'soccerfield', 'hospital', 'stop',
                 'lake', 'supermarket', 'pharmacy', 'strain', 'airport', 'train', 'library', 'publicbath', 'coast',
                 'sportshall', 'forest', 'AVM_pris']

    for col in list_drop:
        del raw[col]

    df_slimmed = raw

    return df_slimmed

def second_drop(df):

    second_slim = df

    list_drop = ['address.latestForSale.address', 'address.latestForSale.streetName',
                 'address.latestForSale.houseNumber', 'address.latestForSale.postal', 'address.latestForSale.city',
                 'bbrData.mostImportantBuilding.municipalityName', 'address.latestForSale.mapPosition.latLng.lat',
                 'address.latestForSale.mapPosition.latLng.lng']

    for col in list_drop:
        del second_slim[col]

    return second_slim


def reorder(df):
    df_reordered = df[['address.address', 'address.latestForSale.address', 'address.street',
                       'address.latestForSale.streetName', 'address.streetName', 'address.latestForSale.houseNumber',
                       'address.postalId', 'address.latestForSale.postal', 'address.city',
                       'address.latestForSale.city', 'address.mapPosition.latLng.lat', 'address.mapPosition.latLng.lng',
                       'address.latestForSale.mapPosition.latLng.lat', 'address.latestForSale.mapPosition.latLng.lng',
                       'address.latestValuation.valuationYear',
                       'address.latestValuation.valuationDate', 'address.latestValuation.parcelValuation',
                       'address.latestValuation.propertyValuation', 'address.latestSale.salePrice',
                       'address.latestForSale.paymentGross', 'address.latestForSale.paymentCash',
                       'address.latestForSale.downPayment', 'address.latestForSale.paymentNet',
                       'address.latestForSale.paymentExpenses', 'address.latestSale.saleType',
                       'address.latestSale.salesYear', 'address.latestForSale.priceDevelopment',
                       'address.latestForSale.priceDevelopmentHistoric', 'address.latestForSale.propertyCharges',
                       'address.latestForSale.usageExpenses', 'address.latestSale.saleDate',
                       'address.latestForSale.dateAnnounced', 'address.latestForSale.dateAdded',
                       'address.latestForSale.dateRemoved', 'address.latestForSale.salesPeriod',
                       'address.latestForSale.salesPeriodTotal', 'AVM_pris',
                       'address.itemTypeName', 'address.latestForSale.itemTypeName',
                       'Anvendelse', 'address.buildYear',
                       'address.latestForSale.buildYear', 'Opførselsesår', 'address.rebuildYear',
                       'address.latestForSale.rebuildYear',
                       'Ombygningsår',
                        'Fredning', 'Bevaringsværdig',
                       'address.latestForSale.area',
                       'Bygning, Samlet areal',
                       'address.latestForSale.areaResidential',
                        'Boligstørrelse',
                       'address.latestForSale.areaParcel', 'address.latestForSale.areaBasement', 'Kælder',
                       'address.latestForSale.areaWeighted', 'Vægtet Areal', 'address.latestForSale.areaPaymentCash',
                       'address.latestForSale.numberOfFloors',
                       'Antal Etager', 'address.latestForSale.floor',
                       'address.latestForSale.floorName',
                       'address.latestForSale.numberOfRooms',
                       'Antal værelser', 'Antal toiletter',
                       'Køkkenforhold', 'Ydervægsmateriale', 'Tagmateriale',
                       'Varmeinstallation', 'address.energyMark', 'address.latestForSale.energyMark',
                       'Energimærke', 'address.environmentData.radonRiskCategory', 'Radonrisiko', 'Støjmåling',
                       'Oversvømmelsesrisiko_skybrud', 'Meter_over_havet',
                       'address.environmentData.drinkingWaterHardnessCategory',
                       'address.environmentData.breakInStatistic.municipalityAverage',
                       'address.environmentData.breakInStatistic.countyAverage',
                       'address.environmentData.breakInStatistic.countryAverage',
                       'address.environmentData.breakInStatistic.riskCategory',
                       'Største_parti', 'Valgdeltagelse', 'Afstemningsområde', 'school', 'roadtrain', 'junction',
                       'daycare', 'metro', 'doctor', 'soccerfield', 'hospital', 'stop', 'lake', 'supermarket',
                       'pharmacy', 'strain','airport', 'train', 'library', 'publicbath', 'coast', 'sportshall',
                       'forest']]

    return df_reordered

    # divided by: address, salesrelated, physical characteristics, environment/society, distances

df_s_reo = reorder(df_slim_1)

all_former_sales.columns

def reorder_sales(df):
    df_reordered = df[['Address', 'dateTime', 'event_bs', 'paymentCash_bs', 'propertyHistoricModel',
                       'description', 'price', 'date', 'tag', 'dateAdded',
                       'dateRemoved', 'paymentCash', 'downPayment', 'paymentExpenses',
                       'areaResidential', 'areaWeighted', 'numberOfFloors', 'numberOfRooms',
                       'salesPeriod', 'priceDevelopment', 'areaParcel', 'event', 'itemType']]
    return df_reordered


all_sales_reordered = reorder_sales(all_former_sales)
full_raw = all_former_sales.merge(df_s_reo, how = 'left', left_on = 'Address', right_on = 'address.address')
df_s_reo.isna().sum().nlargest(50)
#add_list = full_raw.loc[full_raw['address.address'].isna()][['Address']].value_counts()
#add_list.to_csv('address_list.csv')
df_s_reo.shape #(48038, 105)
df_s_reo['address.address'].nunique() # 40283
df_s_reo.loc[df_s_reo['address.latestSale.saleType'] == 'Fri handel']['address.address'].nunique() # 35765
full_raw['dateTime_dt'] = pd.to_datetime(full_raw['dateTime'], format='%Y-%m-%d', errors='coerce')
full_raw.shape #(465257, 129)
full_raw_1721 = full_raw.loc[(full_raw['dateTime_dt'] >= '2017-01-01') & (full_raw['dateTime_dt'] <= '2021-12-31')]
full_raw_1721.shape #(205722, 129)
#tmp_newestSale = pd.DataFrame(full_raw_1721.loc[full_raw_1721['event_bs'].isin([3,4,5,6]), ['Address', 'description']].groupby('Address').first())
#full_raw_1721_1 = full_raw_1721.merge(tmp_newestSale, how = 'left', on = 'Address')
#print(pd.crosstab(full_raw_1721_1['address.latestSale.saleType'], full_raw_1721_1.loc[full_raw['event_bs'].isin([3,4,5,6]),'description_y'], margins = False).to_latex(index = True))
#How many is 'Fri handel' - the only one we are interested in! Familiehandel is pure tax evasion.
full_raw_1721 = full_raw_1721[full_raw_1721['event_bs'].isin([3,4,5,6])] # only keeping sales
full_raw_1721.shape #(66976, 129)

full_raw_1721[['Address', 'dateTime_dt']].nunique()
#full_raw_1721.to_csv('full_raw_1721.csv', index = False)
full_raw_1721_nodup = full_raw_1721.drop_duplicates(['Address', 'dateTime_dt', 'event_bs'])
full_raw_1721_nodup.shape #49985
print(full_raw_1721_nodup.loc[:,'description'].value_counts().to_latex(index=True))

cleaning = full_raw_1721_nodup[full_raw_1721_nodup['event_bs'] == 3]
cleaning.shape #(43249, 129)
cleaning[~cleaning['address.address'].isna()]['Address'].nunique() # 777
cleaning = cleaning[~cleaning['address.address'].isna()]
cleaning.drop(['address.latestSale.saleType', 'description', 'event_bs'], axis=1, inplace=True)

print(cleaning.shape) #Shape is now (42354, 126) due to removal of other than Fri Handel and removing var. afterwards
# cleaning.to_csv('rough_cleaned_oldsales.csv', index=False)
#cleaning = pd.read_csv('rough_cleaned_oldsales.csv')

# ### DETAILED CLEANING ###
# Most cleaning is taken from Frederik Madsen's thesis
cleaning.replace('-', np.nan, inplace=True)
#cleaning = pd.read_csv("rough_cleaned_oldsales.csv")

# ## Address variables
cleaning[['address_w_no', 'postal_and_city']] =  cleaning['Address'].str.rsplit(',', 1, expand = True)
cleaning[['asdf','postal', 'city']] = cleaning['postal_and_city'].str.split(' ', 2, expand = True)
cleaning['address.address'] = cleaning['address.address'].combine_first(cleaning['Address'])
cleaning['address.postalId'] = cleaning['address.postalId'].combine_first(cleaning['postal'])
cleaning['address.street'] = cleaning['address.street'].combine_first(cleaning['address_w_no'])
cleaning['address.city'] = cleaning['address.city'].combine_first(cleaning['city']).astype(str)
cleaning.shape # (43249, 131)
cleaning['length_help'] = cleaning.postal.str.len()
cleaning = cleaning[cleaning.length_help > 3]
cleaning.shape #(42353, 132)
#cleaning['address.address'] = cleaning['address.address'].astype(str)
#cleaning['address.postalId'] = cleaning['address.postalId'].astype(int)
#cleaning['address.city'].isna().sum() = cleaning['address.city'].astype(str)
#cleaning['address_only'] = re.split(r'(^[^\d]+)', cleaning['address_w_no'])[1].strip()
#cleaning['address.street'].isna().sum() = cleaning['address.street'].astype(str)
#cleaning['address.streetName'].isna().sum() = cleaning['address.streetName'].astype(str)
cleaning.shape #(42353, 132)
# NO MISSING VARIABLES ABOVE

# ## Sale date variables
cleaning = cleaning[~cleaning['address.latestSale.salesYear'].isna()]
cleaning.shape #(42352, 132)
cleaning['address.latestSale.salesYear'] = cleaning['address.latestSale.salesYear'].astype(int)
cleaning['address.latestSale.saleDate'] = pd.to_datetime(cleaning['address.latestSale.saleDate'],
                                          format='%d-%m-%Y', errors='coerce')
cleaning['address.latestForSale.dateAnnounced'] = pd.to_datetime(cleaning['address.latestForSale.dateAnnounced'],
                                                  format='%d-%m-%Y', errors='coerce')
cleaning['address.latestForSale.dateAdded'] = pd.to_datetime(cleaning['address.latestForSale.dateAdded'],
                                              format='%d-%m-%Y', errors='coerce')
cleaning['address.latestForSale.dateRemoved'] = pd.to_datetime(cleaning['address.latestForSale.dateRemoved'],
                                                format='%d-%m-%Y', errors='coerce')
cleaning['address.latestForSale.salesPeriod'] = cleaning['address.latestForSale.salesPeriod'].astype(float)
cleaning['address.latestForSale.salesPeriodTotal'] = cleaning['address.latestForSale.salesPeriodTotal'].astype(float)
cleaning['dateTime_dt'] = pd.to_datetime(cleaning['dateTime_dt'], format = '%Y-%m-%d', errors = 'coerce')

cleaning['latestSaleYN'] = (cleaning['dateTime_dt']==cleaning['address.latestSale.saleDate'])
cleaning['Remove_if_sp_matter'] = False
cleaning.loc[(cleaning['latestSaleYN'] == False) & (cleaning['salesPeriod'].isna())]['Remove_if_sp_matter'] = True
cleaning.loc[(cleaning['latestSaleYN'] == True) & (cleaning['salesPeriod'].isna()) &
             (cleaning['address.latestForSale.salesPeriodTotal'].isna()) & (cleaning['address.latestForSale.salesPeriod'].isna())]['Remove_if_sp_matter'] = True

# # From now on loading the csv-file will require parsing dates

# cleaning.to_csv('cleaning.csv', index=False)
# cleaning = pd.read_csv("cleaning.csv", parse_dates=['address.latestSale.saleDate',
#                                                    'address.latestForSale.dateAnnounced',
#                                                    'address.latestForSale.dateAdded',
#                                                    'address.latestForSale.dateRemoved',
#                                                    'dateTime_dt'])

# # Create variables that also could specify the salesperiod if period missing

cleaning['AddedRemoved'] = (cleaning['address.latestForSale.dateRemoved'] \
                            - cleaning['address.latestForSale.dateAdded']).dt.days
cleaning['AnnouncedRemoved'] = (cleaning['address.latestForSale.dateRemoved'] \
                                - cleaning['address.latestForSale.dateAnnounced']).dt.days

# # Let SalesPeriodTotal have priority, then SalesPeriod, then AddedRemoved, then AnnouncedRemoved
cleaning = cleaning.rename(columns={"salesPeriod": "salesPeriod_bs"})
cleaning['SalesPeriod'] = cleaning['address.latestForSale.salesPeriodTotal'].combine_first(cleaning[ \
                        'address.latestForSale.salesPeriod'])
cleaning['SalesPeriod'] = cleaning['SalesPeriod'].combine_first(cleaning['AddedRemoved'])
cleaning['SalesPeriod'] = cleaning['SalesPeriod'].combine_first(cleaning['AnnouncedRemoved'])

# Combining SalesPeriod and salesPeriod_bs if latest sale == True giving SalesPeriod priority.
cleaning.loc[(cleaning['latestSaleYN'] == True)]['SalesPeriod_comb'] = cleaning['SalesPeriod'].\
    combine_first(cleaning['salesPeriod_bs'])
cleaning.loc[(cleaning['latestSaleYN'] == False)]['SalesPeriod_comb'] = cleaning['salesPeriod_bs']

# Removing addresses without salesperiod - skal det gøres

#cleaning = cleaning[~cleaning['SalesPeriod'].isna()]
cleaning.drop(['address.latestForSale.dateAnnounced', 'address.latestForSale.dateAdded',
               'address.latestForSale.dateRemoved', 'address.latestForSale.salesPeriod',
               'address.latestForSale.salesPeriodTotal', 'address.latestForSale.salesPeriodTotal',
               'AddedRemoved', 'AnnouncedRemoved'], axis=1, inplace=True)

# # Delete all unnecesary variables
# Keeping  'address.latestForSale.paymentExpenses' as it is the monthly expenses

cleaning.drop(['address.latestForSale.paymentGross', 'address.latestForSale.downPayment',
               'address.latestForSale.paymentNet',
               'address.latestForSale.priceDevelopment', 'address.latestForSale.priceDevelopmentHistoric',
               'address.latestForSale.propertyCharges', 'address.latestForSale.usageExpenses'], axis=1, inplace=True)

# # Saleprice, cashprice, saledate, and AVM are kept (even though 4640 AVM-values missing)

cleaning.replace(' - ', np.nan, inplace=True)
cleaning['address.latestSale.salePrice'] = cleaning['address.latestSale.salePrice'].str.replace('.','').astype(float)
cleaning['address.latestForSale.paymentCash'] = cleaning['address.latestForSale.paymentCash'].str.replace('.',
                                                '').astype(float)
cleaning['AVM_pris'] = cleaning['AVM_pris'].str.replace('.','').astype(float)

# ### VALUATION VARIABLES ###
# # Delete all unnecessary variables (parcel valuation redundant, and year included in date)

cleaning.drop(['address.latestValuation.parcelValuation',
               'address.latestValuation.valuationYear'], axis=1, inplace=True)
print(cleaning.shape) # (42352, 121)

# # Delete 307 observations, where no valuation, which is deemed too important. OLD LINE OLD LINE OLD LINE
# # This leaves 32241 addresses. OLD LINE OLD LINE OLD LINE
# # We don't delete observations with missing valuation for now at least. --JSL

#cleaning = cleaning[~cleaning['address.latestValuation.propertyValuation'].isna()]
cleaning['address.latestValuation.propertyValuation'] \
        = cleaning['address.latestValuation.propertyValuation'].str.replace('.', '').astype(float)
cleaning['address.latestValuation.valuationDate'] \
        = pd.to_datetime(cleaning['address.latestValuation.valuationDate'], errors='coerce')

# # From now on loading the csv-file will require parsing dates

# cleaning.to_csv('cleaning.csv', index=False)
# cleaning_1 = pd.read_csv("cleaning.csv", parse_dates=['address.latestSale.saleDate',
#                                                    'address.latestValuation.valuationDate'])

# Krydstabulering af bolig-type. Fin overensstemmelse
#cleaning.to_csv('tmp21MAR.csv')
#cleaning_1 = pd.read_csv('cleaning.csv',
#                       parse_dates =
#                       ['address.latestSale.saleDate', 'address.latestValuation.valuationDate',
#                        'address.latestForSale.dateAnnounced',
#                        'address.latestForSale.dateAdded', 'address.latestForSale.dateRemoved',
#                        'dateTime_dt'])

cleaning.shape # 42352, 121
cleaning.loc[cleaning['address.latestForSale.itemTypeName'].isna()]['address.itemTypeName'].value_counts() #counting the missing types
cleaning.loc[cleaning['address.itemTypeName'] == 'Andelsbolig'][['Address', 'address.postalId', 'address.itemTypeName']] # Four andelsboliger. Only the one on in 2100 KBH Ø are a Andel
cleaning.loc[(cleaning['address.itemTypeName'] == 'Andelsbolig') & (cleaning['address.postalId'] != 2100), 'address.itemTypeName'] = 'Ejerlejlighed'
cleaning = cleaning.loc[~(cleaning['address.itemTypeName'] == 'Andelsbolig')] # dropping the andelsbolig
print(pd.crosstab(cleaning['address.latestForSale.itemTypeName'], cleaning['address.itemTypeName'],
                  margins = False).to_latex(index=True))

print(cleaning.shape) #(42351, 121)
cleaning = cleaning[cleaning['address.itemTypeName'].isin(['Ejerlejlighed', 'Villa', 'Rækkehus'])]
print(cleaning.shape) # (42344, 121)

# # Counting number of sales for each property - then summarizing how many have been sold only once, twice and so on.
# # One property has been sold 44 times in 5 years !
print(cleaning['Address'].value_counts().to_frame().value_counts().to_latex(index = True))
# Copied from Madsen's thesis:
# # Since address.itemTypeName has less categories than address.latestForSale.itemTypeName,
# # and the categories seem to be scrambled, we use the former (no missing data in either).
# # Addresses that have address.latestForSale.itemTypeName as andelsbolig are removed.
# # Adresses of that don't have address.itemTypeName as Ejerlejlighed, Rækkehus or Villa are removed (very few).
# print(cleaning.shape) # (42344, 121)
# cleaning = cleaning[cleaning['address.itemTypeName'].isin(['Ejerlejlighed', 'Rækkehus', 'Villa']) \
#                    & ~cleaning['address.latestForSale.itemTypeName'].isin(['Andelsbolig'])]
# # Delete 21 addresses and we now have 42666 addresses left.

cleaning.drop(['address.latestForSale.itemTypeName'], axis=1, inplace=True)

len(cleaning[~(cleaning['address.buildYear'] == cleaning['Opførselsesår'])])
# 170 have build year and Opførselsår not matching. Boligsiden is prioritized.
# We drop the two redundant variables:
cleaning.drop(['address.latestForSale.buildYear', 'Opførselsesår'], axis=1, inplace=True)

cleaning['address.rebuildYear'].replace(0, np.nan, inplace=True)
cleaning['address.latestForSale.rebuildYear'].replace(0, np.nan, inplace=True)
cleaning['rebuildYear'] = cleaning['address.rebuildYear'].combine_first(cleaning['address.latestForSale.rebuildYear'])
cleaning['rebuildYear'] = cleaning['rebuildYear'].combine_first(cleaning['Ombygningsår'])
# Er denne nødvendigt?
#cleaning['rebuildYear'] = cleaning['rebuildYear'].combine_first(cleaning['address.buildYear'])
cleaning.drop(['address.rebuildYear', 'address.latestForSale.rebuildYear',
                'Ombygningsår'], axis=1, inplace=True)

cleaning['Fredning'].isna().sum()
cleaning['Bevaringsværdig'].isna().sum()
# Fredning and bevaringsværdig is unfortunately not accessible for our scraping mechanism so we drop them

cleaning.drop(['Fredning', 'Bevaringsværdig'], axis=1, inplace=True)

# ##
# cleaning.to_csv('tmp4APR.csv', index = False)
#########################################
######### AREA VARIABLES ################
#########################################
print(cleaning.shape) #(42344, 117)
# # Boligsiden data has priority, so when Area has nan (doesnt) Boligstørrelse overrides.
# # Area residential is taken from Boligsiden.
#(cleaning.loc[(cleaning['areaResidential'].notna()) & (cleaning['address.latestForSale.areaResidential'].notna())]['areaResidential'] !=
# cleaning.loc[(cleaning['areaResidential'].notna()) & (cleaning['address.latestForSale.areaResidential'].notna())]['address.latestForSale.areaResidential']).sum()
#cleaning.loc[(cleaning['areaResidential'].notna())]['areaResidential']

# # Basement is again taken with priority from Boligsiden
#Fix Kælder = NA to zero (assumption)
cleaning['Boligstørrelse'] = cleaning['Boligstørrelse'].str.extract('(\d+)').astype(float)
cleaning['Area'] = cleaning['address.latestForSale.area'].combine_first(cleaning['Boligstørrelse'])
cleaning['areaResidential'] = cleaning['areaResidential'].combine_first(cleaning['Area'])
cleaning['Kælder'] = cleaning['Kælder'].str.replace(' m2', '').astype(float)
cleaning['address.latestForSale.areaBasement'].replace(0, np.nan, inplace=True)
cleaning['areaBasement'] = cleaning['address.latestForSale.areaBasement'].combine_first(cleaning['Kælder'])
cleaning['areaBasement'].replace(np.nan, 0, inplace=True)


# # Prioritizing Boligsiden weighted area.
# # If no data for weighted area, let total area override, and vice versa.
cleaning['Vægtet Areal'] = cleaning['Vægtet Areal'].str.extract('(\d+)').astype(float)
cleaning['areaWeighted_tmp'] = cleaning['address.latestForSale.areaWeighted'].combine_first(cleaning['Vægtet Areal'])
cleaning['areaWeighted'] = cleaning['areaWeighted'].combine_first(cleaning['areaWeighted_tmp'])
cleaning = cleaning[cleaning['areaWeighted'].notna()]
# # If weighted area is now nan then delete, which is done for 1, which leaves 42343 addresses
cleaning.drop(['areaWeighted_tmp'], axis = 1, inplace = True)
cleaning['ParcelArea'] = cleaning['areaParcel'].combine_first(cleaning['address.latestForSale.areaParcel'])
cleaning.loc[cleaning['ParcelArea'].isna()]['address.itemTypeName'].value_counts() #Ejerlejlighed: 2166 Villa: 290 Rækkehus: 193
cleaning['address.itemTypeName'].value_counts() #Ejerlejlighed: 37556 Villa: 2983 Rækkehus: 1804
cleaning.loc[(cleaning['ParcelArea'].isna()) & (cleaning['address.itemTypeName'].isin(['Ejerlejlighed'])), 'ParcelArea'] = 0
cleaning = cleaning.loc[(cleaning['ParcelArea'].notna())]

cleaning.drop(['address.latestForSale.area',
               'areaParcel', 'address.latestForSale.areaParcel',
               'address.latestForSale.areaBasement',
               'address.latestForSale.areaWeighted',
               'address.latestForSale.areaPaymentCash',
               'Boligstørrelse',
               'Bygning, Samlet areal',
               'Kælder', 'Vægtet Areal',], axis=1, inplace=True)

print(cleaning.shape) #(41860, 107)


#########################################
### PHYSICAL CHARATERISTICS VARIABLES ###
#########################################

# # Boligsiden data is used for number of floors, floor of the apartment.
# # Variables that carry no or sparse information are deleted.
# # When Boligsiden nr of rooms has nan, Antal værelser overrides.

cleaning['numberOfRooms_tmp'] = cleaning['address.latestForSale.numberOfRooms'].combine_first(cleaning['Antal værelser'])
cleaning['numberOfRooms'] = cleaning['numberOfRooms'].combine_first(cleaning['numberOfRooms_tmp'])
cleaning['numberOfRooms'].isna().sum() # 0 - alles gut.

#########################################
### Environment/society variables #######
#########################################

cleaning['address.latestForSale.energyMark'].isna().sum() #we have 3580 obs. w/o energyMark. We let them be
cleaning['Energimærke'].isna().sum()
cleaning['address.energyMark'].isna().sum()
# # Radon risk classification from dingeo is used, since has more categories.
# # Furthermore from dingeo, oversvømmelse og støjmåling are kept.
# # Break in stat classification from dingeo is used, since has more categories.
cleaning.drop(['address.energyMark',  'Energimærke',
               'address.environmentData.radonRiskCategory',
               'address.environmentData.drinkingWaterHardnessCategory',
               'address.environmentData.breakInStatistic.municipalityAverage',
               'address.environmentData.breakInStatistic.countyAverage',
               'address.environmentData.breakInStatistic.countryAverage',
               'address.environmentData.breakInStatistic.riskCategory'], axis=1, inplace=True)


#########################################
######### Coordinate variables ##########
#########################################

cleaning['address.mapPosition.latLng.lat'] = cleaning['address.mapPosition.latLng.lat'].astype(float)
cleaning['address.mapPosition.latLng.lng'] = cleaning['address.mapPosition.latLng.lng'].astype(float)

# # Coordinates are missing for 110 addresses
# # Found manually on google maps, turns out because newly developed housing.
# # Addresses with missing coordinates are written out in a csv, and manually put in the coordinates.
# # Lat and Lng are created in this file.
# # The following line should not be run again again, would overwrite the manually written coordinates!
####### cleaning.loc[cleaning["address.mapPosition.latLng.lng"].isna(), 'address.address'].to_csv('latlong_missing_MAR.csv')
# # cleaning['Longitude'].isna().sum()

coordinates = pd.read_csv("latlong_missing_MAR.csv", sep = ";")
cleaning = cleaning.merge(coordinates, how='left', on='address.address')
cleaning['Latitude'] = cleaning['lat'].combine_first(cleaning['address.mapPosition.latLng.lat'])
cleaning['Longitude'] = cleaning['lng'].combine_first(cleaning['address.mapPosition.latLng.lng'])
cleaning.drop(['lat', 'lng', 'address.mapPosition.latLng.lat', 'address.mapPosition.latLng.lng'],
              axis=1, inplace=True)
cleaning = cleaning.loc[:, ~cleaning.columns.str.contains('^Unnamed')]


#########################################
#########   Voting variables   ##########
#########################################

missing_vote = cleaning.loc[cleaning["Afstemningsområde"].isna(), ["address.postalId", "Afstemningsområde",
                                                                   'Latitude', 'Longitude']]
missing_vote['address.postalId'].value_counts()
print(missing_vote.loc[:,'address.postalId'].value_counts().to_latex(index=True))

missing_vote['Latitude'] = missing_vote['Latitude'].astype(float)
missing_vote['Longitude'] = missing_vote['Longitude'].astype(float)

BBox_png_4 = ((12.4509, 12.6401, 55.6154, 55.7321)) #png6
ruh_m = plt.imread('map_6.png')

fig, ax = plt.subplots()

ax.scatter(missing_vote['Longitude'],missing_vote['Latitude'],
           zorder=1, alpha= 0.6, s=10)
cmap=plt.cm.jet
norm = plt.Normalize(1500000, 7500000)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
ax.figure.colorbar(sm)
ax.set_title('Missing voting information')
ax.set_xlim(BBox_png_4[0],BBox_png_4[1])
ax.set_ylim(BBox_png_4[2],BBox_png_4[3])
ax.imshow(ruh_m, zorder=0, extent = BBox_png_4, aspect = 'auto')
#plt.savefig('Missing_Voting.png', bbox_inches='tight')
plt.show()

cleaning.loc[(cleaning['Afstemningsområde'].isna()) & (cleaning['address.postalId'] == 2100),
             'Afstemningsområde'] = '1. Øst'
cleaning.loc[(cleaning['Afstemningsområde'].isna()) & (cleaning['address.postalId'] == 2150),
             'Afstemningsområde'] = '1. Øst'
cleaning.loc[(cleaning['Afstemningsområde'].isna()) & (cleaning['address.postalId'] == 2200),
             'Afstemningsområde'] = '5. Nørrebrohallen'
cleaning.loc[(cleaning['Afstemningsområde'].isna()) & (cleaning['address.postalId'] == 2300),
             'Afstemningsområde'] = '2. Øst'
cleaning.loc[(cleaning['Afstemningsområde'].isna()) & (cleaning['address.postalId'] == 2400),
             'Afstemningsområde'] = '6. Vest'
cleaning.loc[(cleaning['Afstemningsområde'].isna()) & (cleaning['address.postalId'] == 2770),
             'Afstemningsområde'] = '2. Øst'

# # From https://www.kmdvalg.dk/fv/2019/KMDValgFV.html one can conclude

cleaning.loc[(cleaning['Valgdeltagelse'].isna()) & (cleaning['address.postalId'] == 2100), 'Valgdeltagelse'] = 88.7
cleaning.loc[(cleaning['Valgdeltagelse'].isna()) & (cleaning['address.postalId'] == 2150), 'Valgdeltagelse'] = 88.7
cleaning.loc[(cleaning['Valgdeltagelse'].isna()) & (cleaning['address.postalId'] == 2200), 'Valgdeltagelse'] = 83.4
cleaning.loc[(cleaning['Valgdeltagelse'].isna()) & (cleaning['address.postalId'] == 2300), 'Valgdeltagelse'] = 86.0
cleaning.loc[(cleaning['Valgdeltagelse'].isna()) & (cleaning['address.postalId'] == 2400), 'Valgdeltagelse'] = 75.6
cleaning.loc[(cleaning['Valgdeltagelse'].isna()) & (cleaning['address.postalId'] == 2770), 'Valgdeltagelse'] = 86.0
cleaning.loc[(cleaning['Største_parti'].isna()) & (cleaning['address.postalId'] \
                                                   == 2100), 'Største_parti'] = 'venstre'
cleaning.loc[(cleaning['Største_parti'].isna()) & (cleaning['address.postalId'] \
                                                   == 2150), 'Største_parti'] = 'venstre'
cleaning.loc[(cleaning['Største_parti'].isna()) & (cleaning['address.postalId'] \
                                                   == 2200), 'Største_parti'] = 'enhedslisten'
cleaning.loc[(cleaning['Største_parti'].isna()) & (cleaning['address.postalId'] \
                                                   == 2300), 'Største_parti'] = 'socialdemokratiet'
cleaning.loc[(cleaning['Største_parti'].isna()) & (cleaning['address.postalId'] \
                                                   == 2400), 'Største_parti'] = 'enhedslisten'
cleaning.loc[(cleaning['Største_parti'].isna()) & (cleaning['address.postalId'] \
                                                   == 2770), 'Største_parti'] = 'socialdemokratiet'

missing_vote = cleaning.loc[cleaning['Afstemningsområde'].isin(['1. Øst','5. Nørrebrohallen',
                                                                '2. Øst','6. Vest','2. Øst'])]

missing_vote = missing_vote.loc[:,['address.postalId', 'Afstemningsområde','Valgdeltagelse','Største_parti']]

missing_vote.value_counts()
print(missing_vote.value_counts().to_latex(index=True))

# CHECK FOR MISSING VARIABLES WRT TO NO OF ROOMS, TOILETS ETC.
cleaning.drop(['no'], axis = 1, inplace = True)
cleaning.info(verbose = True)
cleaning.isna().sum().nlargest(50)
cleaning['paymentExpenses'] = cleaning['paymentExpenses'].combine_first(cleaning['address.latestForSale.paymentExpenses'])
cleaning.drop(['propertyHistoricModel', 'asdf', 'address.latestForSale.city',
               'address.latestForSale.mapPosition.latLng.lat', 'address.latestForSale.mapPosition.latLng.lng',
               'address.latestForSale.houseNumber', 'address.latestForSale.streetName',
               'address.latestForSale.address', 'tag', 'address.latestForSale.postal',
               'address.latestForSale.numberOfRooms',
               'address.latestForSale.floor', 'address.latestForSale.floorName',
               'address.latestForSale.numberOfRooms', 'downPayment', 'dateAdded',
               'dateRemoved', 'itemType', 'priceDevelopment', 'address.latestForSale.paymentExpenses',
               'address.latestForSale.paymentCash', 'length_help', 'address_w_no', 'postal_and_city',
               'postal'], axis = 1, inplace = True)
cleaning.info(verbose = True)
cleaning.drop(['dateTime_dt', 'numberOfRooms_tmp', 'latestSaleYN', 'Area', 'Antal Etager',
               'address.latestForSale.numberOfFloors', 'address.latestSale.salePrice', 'address.latestValuation.valuationDate'],
              axis = 1, inplace = True)
# paymentCash_bs IS THE PRICE TO USE. EQUAL TO PRICE AND THE address.latestSale.salePrice ARE ONLY LATEST SALE
cleaning.info(verbose = True)
cleaning.drop(['event', 'numberOfFloors', 'paymentCash', 'address.latestForSale.areaResidential'], axis = 1, inplace = True)
cleaning[cleaning['Antal toiletter'].isna()]['Address'] #Locating the missing information. 7 is missing anvendelse, antal etager,
# antal værelser, antal toiletter etc.
## MISSING ARE:
#                               address.address  \
#1040    Nordre Fasanvej 167, 2. th, 2000 Frederiksberg
#14601  Dronningens Tværgade 50, 3. 3, 1302 København K
#18127      Burmeistersgade 23, 4. th, 1429 København K
#33405                    Islevhusvej 61, 2700 Brønshøj
#37930         Rumæniensgade 4, 2. th, 2300 København S
#40339      Burmeistersgade 23, 4. th, 1429 København K
#41958        Østerbrogade 118, 1. tv, 2100 København Ø

# We insert it manually - Toilet, dtype: float64
# Nordre Fasanvej
cleaning.loc[(cleaning['Anvendelse'].isna()) & (cleaning['address.postalId'] == 2000), 'Anvendelse'] = \
    'Bolig i etageejendom, flerfamiliehus eller to-familiehus'
cleaning.loc[(cleaning['Køkkenforhold'].isna()) & (cleaning['address.postalId'] == 2000), 'Køkkenforhold'] = \
'Eget køkken med afløb'
cleaning.loc[(cleaning['Antal toiletter'].isna()) & (cleaning['address.postalId'] == 2000), 'Antal toiletter'] = 1
#cleaning.loc[(cleaning['Antal badeværelser'].isna()) & (cleaning['address.postalId'] == 2000), 'Antal badeværelser'] = 1
cleaning.loc[(cleaning['Antal værelser'].isna()) & (cleaning['address.postalId'] == 2000), 'Antal værelser'] = 2
cleaning.loc[(cleaning['Ydervægsmateriale'].isna()) & (cleaning['address.postalId'] == 2000), 'Ydervægsmateriale'] = 'Mursten'
cleaning.loc[(cleaning['Tagmateriale'].isna()) & (cleaning['address.postalId'] == 2000), 'Tagmateriale'] = 'Fibercement herunder asbest'
cleaning.loc[(cleaning['Varmeinstallation'].isna()) & (cleaning['address.postalId'] == 2000), 'Varmeinstallation'] = 'Fjernvarme/blokvarme'

# Dronningens Tværgade
cleaning.loc[(cleaning['Anvendelse'].isna()) & (cleaning['address.postalId'] == 1302), 'Anvendelse'] = \
    'Bolig i etageejendom, flerfamiliehus eller to-familiehus'
cleaning.loc[(cleaning['Køkkenforhold'].isna()) & (cleaning['address.postalId'] == 1302), 'Køkkenforhold'] = \
'Eget køkken med afløb'
cleaning.loc[(cleaning['Antal toiletter'].isna()) & (cleaning['address.postalId'] == 1302), 'Antal toiletter'] = 1
cleaning.loc[(cleaning['Antal værelser'].isna()) & (cleaning['address.postalId'] == 1302), 'Antal værelser'] = 3
cleaning.loc[(cleaning['Ydervægsmateriale'].isna()) & (cleaning['address.postalId'] == 1302), 'Ydervægsmateriale'] = 'Mursten'
cleaning.loc[(cleaning['Tagmateriale'].isna()) & (cleaning['address.postalId'] == 1302), 'Tagmateriale'] = 'Tegl'
cleaning.loc[(cleaning['Varmeinstallation'].isna()) & (cleaning['address.postalId'] == 1302), 'Varmeinstallation'] = 'Fjernvarme/blokvarme'

#Islevhusvej
cleaning.loc[(cleaning['Anvendelse'].isna()) & (cleaning['address.postalId'] == 2700), 'Anvendelse'] = \
    'Fritliggende enfamiliehus'
cleaning.loc[(cleaning['Køkkenforhold'].isna()) & (cleaning['address.postalId'] == 2700), 'Køkkenforhold'] = \
'Eget køkken med afløb'
cleaning.loc[(cleaning['Antal toiletter'].isna()) & (cleaning['address.postalId'] == 2700), 'Antal toiletter'] = 3
cleaning.loc[(cleaning['Antal værelser'].isna()) & (cleaning['address.postalId'] == 2700), 'Antal værelser'] = 7
cleaning.loc[(cleaning['Ydervægsmateriale'].isna()) & (cleaning['address.postalId'] == 2700), 'Ydervægsmateriale'] = 'Mursten'
cleaning.loc[(cleaning['Tagmateriale'].isna()) & (cleaning['address.postalId'] == 2700), 'Tagmateriale'] = 'Fibercement herunder asbest'
cleaning.loc[(cleaning['Varmeinstallation'].isna()) & (cleaning['address.postalId'] == 2700), 'Varmeinstallation'] = 'Centralvarme med én fyringsenhed'

#Rumæniensgade
cleaning.loc[(cleaning['Anvendelse'].isna()) & (cleaning['address.postalId'] == 2300), 'Anvendelse'] = \
    'Bolig i etageejendom, flerfamiliehus eller to-familiehus'
cleaning.loc[(cleaning['Køkkenforhold'].isna()) & (cleaning['address.postalId'] == 2300), 'Køkkenforhold'] = \
'Eget køkken med afløb'
cleaning.loc[(cleaning['Antal toiletter'].isna()) & (cleaning['address.postalId'] == 2300), 'Antal toiletter'] = 1
cleaning.loc[(cleaning['Antal værelser'].isna()) & (cleaning['address.postalId'] == 2300), 'Antal værelser'] = 2
cleaning.loc[(cleaning['Ydervægsmateriale'].isna()) & (cleaning['address.postalId'] == 2300), 'Ydervægsmateriale'] = 'Mursten'
cleaning.loc[(cleaning['Tagmateriale'].isna()) & (cleaning['address.postalId'] == 2300), 'Tagmateriale'] = 'Tegl'
cleaning.loc[(cleaning['Varmeinstallation'].isna()) & (cleaning['address.postalId'] == 2300), 'Varmeinstallation'] = 'Fjernvarme/blokvarme'

#Østerbrogade 118, 1. tv, 2100
cleaning.loc[(cleaning['Anvendelse'].isna()) & (cleaning['address.postalId'] == 2100), 'Anvendelse'] = \
    'Bolig i etageejendom, flerfamiliehus eller to-familiehus'
cleaning.loc[(cleaning['Køkkenforhold'].isna()) & (cleaning['address.postalId'] == 2100), 'Køkkenforhold'] = \
'Eget køkken med afløb'
cleaning.loc[(cleaning['Antal toiletter'].isna()) & (cleaning['address.postalId'] == 2100), 'Antal toiletter'] = 1
cleaning.loc[(cleaning['Antal værelser'].isna()) & (cleaning['address.postalId'] == 2100), 'Antal værelser'] = 5
cleaning.loc[(cleaning['Ydervægsmateriale'].isna()) & (cleaning['address.postalId'] == 2100), 'Ydervægsmateriale'] = 'Mursten'
cleaning.loc[(cleaning['Tagmateriale'].isna()) & (cleaning['address.postalId'] == 2100), 'Tagmateriale'] = 'Fibercement herunder asbest'
cleaning.loc[(cleaning['Varmeinstallation'].isna()) & (cleaning['address.postalId'] == 2100), 'Varmeinstallation'] = 'Fjernvarme/blokvarme'

#Burmeistersgade 23, 4. th,
cleaning.loc[(cleaning['Anvendelse'].isna()) & (cleaning['address.postalId'] == 1429), 'Anvendelse'] = \
    'Bolig i etageejendom, flerfamiliehus eller to-familiehus'
cleaning.loc[(cleaning['Køkkenforhold'].isna()) & (cleaning['address.postalId'] == 1429), 'Køkkenforhold'] = \
'Eget køkken med afløb'
cleaning.loc[(cleaning['Antal toiletter'].isna()) & (cleaning['address.postalId'] == 1429), 'Antal toiletter'] = 1
cleaning.loc[(cleaning['Antal værelser'].isna()) & (cleaning['address.postalId'] == 1429), 'Antal værelser'] = 2
cleaning.loc[(cleaning['Ydervægsmateriale'].isna()) & (cleaning['address.postalId'] == 1429), 'Ydervægsmateriale'] = 'Mursten'
cleaning.loc[(cleaning['Tagmateriale'].isna()) & (cleaning['address.postalId'] == 1429), 'Tagmateriale'] = 'Fibercement herunder asbest'
cleaning.loc[(cleaning['Varmeinstallation'].isna()) & (cleaning['address.postalId'] == 1429), 'Varmeinstallation'] = 'Fjernvarme/blokvarme'

cleaning.info(verbose = True)

############################################
########      MACRO VARIABLES       ########
############################################

# All data has been preprocessed in Excel, this means making dates the right format and lagging them (month + 1) as needed

#Loading data from the Fed.
Fed = pd.read_csv('Fed.csv', sep = ";")
cleaning['Join_date'] = cleaning['dateTime'].astype(str).str[:7]
cleaning = cleaning.merge(Fed, how = "left", on = 'Join_date')
# cleaning[['address.latestSale.saleDate','Assets']].head()

# Loading the rate from Statistics Denmark
Dst = pd.read_csv('DST.csv', sep = ";", decimal = ",")
cleaning = cleaning.merge(Dst, how = "left", on = 'Join_date')

# Loading the expectations about housing from Statistics Denmark
# Planer om køb eller opførsel af bolig indenfor de næste 12 måneder?
Dst_forv2 = pd.read_csv('DST_FORV2_data.csv', sep = ";", decimal = ",")
cleaning = cleaning.merge(Dst_forv2, how = "left", on = 'Join_date')

# Loading unemployment numbers from Statistics Denmark
# Fuldtidsledige (sæsonkorrigeret) efter område, tid og sæsonkorrigering
# og faktiske tal
# Landsdel Byen København
# Sæsonkorrigeret i pct. af arbejdsstyrken
Dst_aus08 = pd.read_csv('AUS08.csv', sep = ";", decimal = ",")
cleaning = cleaning.merge(Dst_aus08, how = "left", on = 'Join_date')

cleaning['MOH'] = cleaning['Meter_over_havet'].astype(str).str[:4]
cleaning['MOH'] = cleaning['MOH'].astype(float)
cleaning.drop(['salesPeriod_bs', 'Meter_over_havet'], axis = 1, inplace = True)
cleaning.drop(['price', ])
cleaning.info(verbose = True)
(cleaning['city'] == cleaning['address.city']).sum()
cleaning.drop(['DST_FORV_VAR_x', 'Rate_x', 'ARBLØSHED_x'], axis = 1, inplace = True)
cleaning.rename(columns={"DST_FORV_VAR_y": "DST_FORV_VAR", 'Rate_y': 'Rate', 'ARBLØSHED_y': "ARBLØSHED"}, inplace = True)

# cleaning.to_csv('TMP06APR.csv', index = False)
cleaning.drop(['price', 'date', 'address.address', 'address.street',
               'address.streetName', 'address.latestSale.salesYear',
               'address.latestSale.saleDate', 'Antal værelser',
               'city', 'Join_date', 'Date'], axis = 1, inplace = True)

#renaming
cleaning.rename(columns = {"dateTime": "saleDate_b", "paymentCash_bs": "price_b", "Address": "Address_b",
                           "paymentExpenses": "mhtlExpens_b", "areaResidential": "areaResidential_bd",
                           "areaWeighted": "areaWeighted_bd", "numberOfRooms": "numberOfRooms_bd",
                           "areaBasement": "areaBasement_bd", "ParcelArea": "parcelArea_b",
                           "address.postalId": "postalId_b", "address.city": "city_b",
                           "address.latestValuation.propertyValuation": "propValuation_b",
                           "address.itemTypeName": "itemTypeName_b",
                           "address.latestForSale.energyMark": "energyMark_b",
                           "address.buildYear": "buildYear_b", "AVM_pris": "AVM_price_d",
                           "Anvendelse": "usageType_d", "Antal toiletter": "noToilets_d",
                           "Køkkenforhold": "kitchenType_d", "Ydervægsmateriale": "outwMaterial_d",
                           "Tagmateriale": "roofType_d", "Varmeinstallation": "heatType_d",
                           "Radonrisiko": "radonRisk_d", "Støjmåling": "noiseLvl_d",
                           "Oversvømmelsesrisiko_skybrud": "floodRisk_d", "MOH": "MAS_d",
                           "Største_parti": "biggestParty_d", "Valgdeltagelse": "voterTurnout_d",
                           "Afstemningsområde": "votingArea_d", "school": "school_h",
                           "roadtrain": "roadtrain_h", "junction": "junction_h", "daycare": "daycare_h",
                           "metro": "metro_h", "doctor": "doctor_h", "soccerfield": "soccerfield_h",
                           "hospital": "hospital_h", "stop": "stop_h", "lake": "lake_h",
                           "supermarket": "supermarket_h", "pharmacy": "pharmacy_h", "strain": "strain_h",
                           "airport": "airport_h", "train": "train_h", "library": "library_h",
                           "publicbath": "publicbath_h", "coast": "coast_h", "sportshall": "sportshall_h",
                           "forest": "forest_h", "Remove_if_sp_matter": "indicatorSalesP_own",
                           "SalesPeriod": "salesPeriod_b", "rebuildYear": "rebuildYear_bd",
                           "Latitude": "latitude_b", "Longitude": "longitude_b", "Assets": "assets_f",
                           "ARBLØSHED": "arbløshed_s", "Rate": "rate_s", "DST_FORV_VAR": "dst_forv_var_s"},
                inplace = True)

cleaning.info(verbose = True)
cleaning['parcelArea_b'].max()
cleaning.to_csv('Final_cleaned_07APR.csv', index = False)

