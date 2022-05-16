
# coding: utf-8

# # Scraping data

# * ## Import relevant packages

# In[2]:

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from tqdm import tqdm
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import json
import pandas as pd
import time
import requests
import concurrent.futures
import lxml
from datetime import datetime
from datetime import timedelta
from ast import literal_eval



# * ## Functions for scraping Boligsiden

# In[ ]:

#### Get URL for Boligsiden search for specified period in selected Kommune

def get_url_boligsiden(kommune, startdate, enddate, p):
    url = 'http://www.boligsiden.dk/salgspris/solgt/alle/{}'
    params = '?periode.from={}&periode.to={}&displaytab=mergedtab&sort'              '=salgsdato&salgstype=%5Bobject%20Object%5D&kommune={}'
    full_url = url + params
    return full_url.format(p, startdate, enddate, kommune)

#### Get number of pages for Boligsiden search

def get_max_pages_boligsiden(url):
    options = webdriver.chrome.options.Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome('/Users/Johan/PycharmProjects/Speciale_kode/chromedriver.exe', options=options)
    driver.get(url)
    page_text = driver.find_element_by_class_name("salesprice-result").text

    last_page_num = (page_text.split("af ")[1]).split("\n")[0]
    return last_page_num

#### Get all address links on search page

def get_all_urls_on_page_boligsiden(url):
    options = webdriver.chrome.options.Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')

    driver = webdriver.Chrome('/Users/Johan/PycharmProjects/Speciale_kode/chromedriver.exe', options=options)
    driver.get(url)

    all_https = []
    with_reentries_https = []
    #WebDriverWait(driver, 25).until(EC.presence_of_element_located((By.ID, 'page-salesprice-result')))
    #driver.implicitly_wait(25)
    for elem in driver.find_elements_by_tag_name('a'):
        all_https.append(elem.get_attribute("href"))
    print(type(all_https))
    print(all_https)
    #bolig-links wanted appear multiple times, so we take away all single time occuring links
    for i in range(len(all_https)):
        if all_https[i] in all_https[:i]:
            with_reentries_https.append(all_https[i])
    #Take away first two entries, which are not bolig links
    with_reentries_https = with_reentries_https[2:]
    reduced_list = list(set(with_reentries_https))
    #To make sure no other links are included
    boliger_https = []
    condition = 'https://www.boligsiden.dk/adresse/'
    for i in reduced_list:
        if isinstance(i, str):
            if condition in i:
                boliger_https.append(i)
    return boliger_https

#### Get list of all address URLs for search

def get_all_links_boligsiden(kommune, startdate, enddate):
    # Returns first https-page with given variables
    first_page = get_url_boligsiden(kommune, startdate, enddate, 1)

    # Getting number of total pages
    total_pages = get_max_pages_boligsiden(first_page)
    print(total_pages)

    # Empty lists
    link_to_all_pages = []
    list_of_all_pages = []

    # Collects a list with all the pages that we want to collect
    for x in tqdm(range(int(total_pages))):
        all_pages = get_url_boligsiden(kommune, startdate, enddate, x + 1)
        link_to_all_pages.append(all_pages)

        page_list = get_all_urls_on_page_boligsiden(link_to_all_pages[x])
        list_of_all_pages.extend(page_list)

    # Returns list with all the wanted url's
    return (list_of_all_pages)

#### Scrape information for single address on address URL 

def get_simple_single_page_boligsiden(url):

    url = url
    html = urlopen(url)
    soup = BeautifulSoup(html.read(), 'html.parser')
    head = str(soup.find('head'))
    try:
        json_string = re.search(r'__bs_addresspresentation__ = ([^;]*)', head).group(1)
        data = json.loads(json_string)
        df1 = pd.json_normalize(data)
        df2 = pd.DataFrame()
    except:
        json_string = re.search(r'__bs_propertypresentation__ = ([^;]*)', head).group(1)
        data = json.loads(json_string)
        df2 = pd.json_normalize(data)
        df1 = pd.DataFrame()

    return df1, df2

#### Collect scraped information for all addresses in two dataframes

def get_data_boligsiden(links):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for x in tqdm(range(0, len(links))):
        try:
            df_pages1, df_pages2 = get_simple_single_page_boligsiden(links[x])
            df1 = pd.concat([df1, df_pages1])
            df2 = pd.concat([df2, df_pages2])
        except:
            pass


    return df1, df2


# * ## Functions for scraping DinGeo.dk

# In[ ]:

#### Get DinGeo-URLs for all addresses in Boligsiden dataframes 

def get_geolinks1(df):
    df["dingeo_link"] = ""

    for x in range(0, len(df)):
        if '-' in (df['address.street'][x]):
            df['address.street'][x] = df['address.street'].str.split('-').str[0][x] + '--'                                       + df['address.street'].str.split('-').str[1][x]

        if ',' in (df['address.street'][x]):
            add_part = str(df['address.postalId'][x]) + '-' + df['address.city'][x].replace(" ", "-") + '/'                        + df['address.street'].str.split(',').str[0][x].replace(" ","-") + '/'                        + df['address.street'].str.split(', ').str[1][x].replace(".", "").replace(" ", "-")
            url = 'https://www.dingeo.dk/adresse/' + add_part
        elif 'Adressen er ikke tilgængelig' in (df['address.street'][x]):
            url = 'Utilgængelig'
        else:
            add_part = str(df['address.postalId'][x]) + '-' + df['address.city'][x].replace(" ", "-") + '/'                        + df['address.street'].str.split(',').str[0][x].replace(" ","-")
            url = 'https://www.dingeo.dk/adresse/' + add_part

        if '-lejl-' in url:
            url = url.replace('-lejl-','-')

        df['dingeo_link'][x] = url

    return df

def get_geolinks2(df):
    df["dingeo_link"] = ""

    for x in range(0, len(df)):
        if '-' in (df['property.address'][x]):
            df['property.address'][x] = df['property.address'].str.split('-').str[0][x] + '--'                                         + df['property.address'].str.split('-').str[1][x]

        if ',' in (df['property.address'][x]):
            ad_part = str(df['property.postal'][x]) + '-' + df['property.city'][x].replace(" ", "-") + '/'                       + df['property.address'].str.split(',').str[0][x].replace(" ","-") + '/'                       + df['property.address'].str.split(', ').str[1][x].replace(".", "").replace(" ", "-")
            url = 'https://www.dingeo.dk/adresse/' + ad_part
        elif 'Adressen er ikke tilgængelig' in (df['property.address'][x]):
            url = 'Utilgængelig'
        else:
            ad_part = str(df['property.postal'][x]) + '-' + df['property.city'][x].replace(" ", "-") + '/'                       + df['property.address'].str.split(',').str[0][x].replace(" ","-")
            url = 'https://www.dingeo.dk/adresse/' + ad_part

        if '-lejl-' in url:
            url = url.replace('-lejl-','-')

        df['dingeo_link'][x] = url

    return df

#### Scrape information for each individual address on DinGeo.dk

def dingeo_page(url):
    url = url

    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Dictionary
    data = {}
    data['dingeo_link'] = url
    try:
        data['Radonrisiko'] = [soup.find_all("div", {"id": 'radon'})[0].find_all("strong")[0].get_text()]
    except:
        pass

    if 'ikke registreret trafikstøj' in soup.find_all("div", {"id": 'trafikstoej'})[0].get_text():
        data['Støjmåling'] = ['Ingen trafikstøj']
    elif 'mangler desværre at indsamle trafikstøj' in soup.find_all("div", {"id": 'trafikstoej'})[0].get_text():
        data['Støjmåling'] = ['Mangler']
    else:
        data['Støjmåling'] = [soup.find_all("div", {"id": 'trafikstoej'})[0].find_all("b")[1].get_text()]

    data['Oversvømmelsesrisiko_skybrud'] = [soup.find_all("div", {"id": 'skybrud'})[0].find_all("b")[0].get_text()]
    data['Meter_over_havet'] = [soup.find_all("div", {"id": 'stormflod'})[0].find_all("b")[0].get_text()]

    table_0 = pd.read_html(str(soup.find_all('table')))[0].iloc[:, 0:2]
    table_0 = table_0.set_axis(['Tekst', 'Værdi'], axis=1, inplace=False)

    table_1 = pd.read_html(str(soup.find_all('table')))[1].iloc[:, 0:2]
    table_1 = table_1.set_axis(['Tekst', 'Værdi'], axis=1, inplace=False)

    table_2 = pd.read_html(str(soup.find_all('table')))[2].iloc[:, 0:2]
    table_2 = table_2.set_axis(['Tekst', 'Værdi'], axis=1, inplace=False)

    table_3 = pd.read_html(str(soup.find_all('table')))[3:-2]
    table_3 = pd.concat(table_3).iloc[:, 0:2]
    table_3 = table_3.set_axis(['Tekst', 'Værdi'], axis=1, inplace=False)

    table = pd.concat([table_0, table_1, table_2, table_3])

    table = table.loc[table['Tekst'].isin(['Anvendelse', 'Opførselsesår', 'Ombygningsår', 'Fredning',
                                           'Køkkenforhold', 'Antal Etager', 'Antal toiletter', 'Antal badeværelser',
                                           'Antal værelser',
                                           'Ydervægsmateriale', 'Tagmateriale', 'Varmeinstallation',
                                           'Bygning, Samlet areal', 'Boligstørrelse', 'Kælder', 'Vægtet Areal'])]
    mydict = dict(zip(table.Tekst, list(table.Værdi)))
    data.update(mydict)

    try:
        if 'ikke finde energimærke' in soup.find_all("div", {"id": 'energimaerke'})[0].get_text():
            data['Energimærke'] = ['Mangler']
        else:
            data['Energimærke'] = [soup.find_all("div", {"id": 'energimaerke'})[0].find_all("p")[0].get_text()[-3:-2]]
        data['Indbrudsrisiko'] = [soup.find_all("div", {"id": 'indbrud'})[0].find_all("u")[0].get_text()]
    except:
        pass

    try:
        if 'ikke fredet' in str(soup.find_all("div", {"id": 'fbb'})[0].find_all("h2")[0]):
            data['Bevaringsværdig'] = [0]
        elif 'Bygningen er Bevaringsværdig' in str(soup.find_all("div", {"id": 'fbb'})[0].find_all("h2")[0]):
            data['Bevaringsværdig'] = re.findall(r'\d+', str(soup.find_all("div", {"id": 'fbb'})[0].find_all("p")[4]))
        elif 'Fejl ved opslag af' in str(soup.find_all("div", {"id": 'fbb'})[0].find_all("h2")[0]):
            data['Bevaringsværdig'] = 'Mangler' #Seems to be flaw on site, all get mangler
        else:
            data['Bevaringsværdig'] = 'Ukendt'
    except:
        pass

    try:
        data['Største_parti'] = re.findall(r'valg/(.*?)(?<!\\).png',
                                           str(soup.find_all("div", {"id": 'valgdata'})[0].find_all('h2')[0]))
        data['Valgdeltagelse'] =         re.findall("\d+.\d+", str(soup.find_all("div", {"id": 'valgdata'})[0].find_all('p')[1]))[1]
        data['Afstemningsområde'] = [soup.find_all("div", {"id": 'valgdata'})[0].find_all("strong")[0].get_text()]
    except:
        pass

    try:
        url_vurdering = url + '/vurdering'
        resp_vurdering = requests.get(url_vurdering)
        soup_vurdering = BeautifulSoup(resp_vurdering.text, 'html.parser')
        data['AVM_pris'] =         soup_vurdering.find_all("div", {"id": 'avmnumber'})[0].get_text() #made correction
    except:
        pass

        # Make dataframe
    df_page = pd.DataFrame(data)

    return df_page

#### Collect all scraped data from DinGeo for the addresses and ad to Boligsiden-dataframes

def for_threading(url):

    try:
        df_pages = dingeo_page(url)
        # df_geo = pd.concat([df_geo, df_pages])
        #   time.sleep(1)
        return df_pages
    except:
       pass

def add_dingeo(df):

    url_list = df['dingeo_link'].tolist()

    df_geo = pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = tqdm(executor.map(for_threading, url_list))

        for result in results:
            df_geo = pd.concat([df_geo, result])


    df_Boligsiden_Dingeo = pd.merge(df, df_geo, how='inner', on='dingeo_link', right_index=False).drop_duplicates()

    return df_Boligsiden_Dingeo



# * ## Functions for scraping hvorlangterder.dk

# In[ ]:

#### Scrape information for single address from hvorlangterder.dk

def get_hvorlangterder(address):
    try:
        url = 'https://hvorlangterder.poi.viamap.net/v1/nearestpoi/?poitypes'               '=daycare,doctor,hospital,junction,metro,school,stop,strain,supermarket,train,library,pharmacy,coast'               ',forest,lake,airport,sportshall,publicbath,soccerfield,roadtrain&fromaddress=' + address               + '&mot=foot&token=eyJkcGZ4IjogImh2b3JsYW5ndGVyZGVyIiwgInByaXZzIjogInIxWjByMEYwazZCdFdxUWNPVXlrQi95N'                 'lNVcEp2MlFiZ3lYZXRxNEhZNFhPLzNZclcwK0s5dz09In0.fP4JWis69HmaSg5jVHiK8nemiCu6VaMULSGGJyK4D4PkWq4iA1'                 '+nSHWMaHxepKwJ83sEiy9nMNZhv7BcktRNrA'
        resp = requests.get(url)
        cont = resp.json()
        df = pd.DataFrame(cont).loc[['routedmeters']]
        df['Location'] = address

        return (df)
    except:
        pass


#### Scrape data from hvorlangterder.dk for all adresses and merge with data from Boligsiden and DinGeo.dk
    
def add_hvorlangterder(df):


    df_hvorlangt = pd.DataFrame()

    for i in tqdm(range(0,len(df))):
        try:
            data = get_hvorlangterder(str(df['Location'][i]))
            df_hvorlangt = pd.concat([df_hvorlangt, data])
        except Exception:
            pass
        time.sleep(0.2)


    merged = pd.merge(df, df_hvorlangt, how='inner', on='Location', right_index=False).drop_duplicates()
    return merged


#Dates has format YYYY-MM-DD
links = get_all_links_boligsiden('Frederiksberg', '2017-01-01', '2021-12-31') # This was also done for Frederiksberg
#links_21 = links
with open('links_boligsiden_F_17-21.txt', 'w') as file:
        file.write(str(links))
df1, df2 = get_data_boligsiden(links)


with open("links_boligsiden_K_21.txt", "r") as file:
   links = eval(file.readline())

df1, df2 = get_data_boligsiden(links)
df1.to_csv('boligsiden_1_F_17-21.csv', index=False)
df2.to_csv('boligsiden_2_F_17-21.csv', index=False)



df_Boligsiden1 = pd.read_csv("boligsiden_1_K_17.csv", error_bad_lines = False)
df_Boligsiden_Geo1 = get_geolinks1(df_Boligsiden1) #Here we choose how many to do at a time
df_Boligsiden_Dingeo1 = add_dingeo(df_Boligsiden_Geo1)
pd.set_option('display.max_columns', None)
print(df_Boligsiden_Dingeo1)
df_Boligsiden_Dingeo1.to_csv('boligsiden_dingeo_1_K7_17.csv', index=False)


df_Boligsiden1 = pd.read_csv("boligsiden_1_K_18.csv", error_bad_lines = False)
df_Boligsiden_Geo1 = get_geolinks1(df_Boligsiden1) #Here we choose how many to do at a time
df_Boligsiden_Dingeo1 = add_dingeo(df_Boligsiden_Geo1)
df_Boligsiden_Dingeo1.to_csv('boligsiden_dingeo_1_K7_18.csv', index=False)
df_Boligsiden1 = pd.read_csv("boligsiden_1_K_19.csv", error_bad_lines = False)
df_Boligsiden_Geo1 = get_geolinks1(df_Boligsiden1) #Here we choose how many to do at a time
df_Boligsiden_Dingeo1 = add_dingeo(df_Boligsiden_Geo1)
df_Boligsiden_Dingeo1.to_csv('boligsiden_dingeo_1_K7_19.csv', index=False)

df_Boligsiden1 = pd.read_csv("boligsiden_1_K_20.csv", error_bad_lines = False)
df_Boligsiden_Geo1 = get_geolinks1(df_Boligsiden1) #Here we choose how many to do at a time
df_Boligsiden_Dingeo1 = add_dingeo(df_Boligsiden_Geo1)
df_Boligsiden_Dingeo1.to_csv('boligsiden_dingeo_1_K7_20.csv', index=False)
df_Boligsiden1 = pd.read_csv("boligsiden_1_K_21.csv", error_bad_lines = False)
df_Boligsiden_Geo1 = get_geolinks1(df_Boligsiden1) #Here we choose how many to do at a time
df_Boligsiden_Dingeo1 = add_dingeo(df_Boligsiden_Geo1)
df_Boligsiden_Dingeo1.to_csv('boligsiden_dingeo_1_K7_21.csv', index=False)
df_Boligsiden1 = pd.read_csv("boligsiden_1_F_17-21.csv", error_bad_lines = False)
df_Boligsiden_Geo1 = get_geolinks1(df_Boligsiden1) #Here we choose how many to do at a time
df_Boligsiden_Dingeo1 = add_dingeo(df_Boligsiden_Geo1)
df_Boligsiden_Dingeo1.to_csv('boligsiden_dingeo_1_F7_17-21.csv', index=False)


geo_bolig1 = pd.read_csv("boligsiden_dingeo_1_K7_17.csv")
geo_bolig1['Location'] = geo_bolig1['address.street'].str.split(',').str[0] + ', ' \
                       + geo_bolig1['address.postalId'].astype(str)
df_Boligsiden_Dingeo_Hvorlangterder1 = add_hvorlangterder(geo_bolig1)
df_Boligsiden_Dingeo_Hvorlangterder1.to_csv('bdh_1_K7_17.csv', index=False)


geo_bolig1 = pd.read_csv("boligsiden_dingeo_1_K7_18.csv")
geo_bolig1['Location'] = geo_bolig1['address.street'].str.split(',').str[0] + ', ' \
                       + geo_bolig1['address.postalId'].astype(str)
df_Boligsiden_Dingeo_Hvorlangterder1 = add_hvorlangterder(geo_bolig1)
df_Boligsiden_Dingeo_Hvorlangterder1.to_csv('bdh_1_K7_18.csv', index=False)


geo_bolig1 = pd.read_csv("boligsiden_dingeo_1_K7_19.csv")
geo_bolig1['Location'] = geo_bolig1['address.street'].str.split(',').str[0] + ', ' \
                       + geo_bolig1['address.postalId'].astype(str)
df_Boligsiden_Dingeo_Hvorlangterder1 = add_hvorlangterder(geo_bolig1)
df_Boligsiden_Dingeo_Hvorlangterder1.to_csv('bdh_1_K7_19.csv', index=False)


geo_bolig1 = pd.read_csv("boligsiden_dingeo_1_K7_20.csv")
geo_bolig1['Location'] = geo_bolig1['address.street'].str.split(',').str[0] + ', ' \
                       + geo_bolig1['address.postalId'].astype(str)
df_Boligsiden_Dingeo_Hvorlangterder1 = add_hvorlangterder(geo_bolig1)
df_Boligsiden_Dingeo_Hvorlangterder1.to_csv('bdh_1_K7_20.csv', index=False)


geo_bolig1 = pd.read_csv("boligsiden_dingeo_1_K7_21.csv")
geo_bolig1['Location'] = geo_bolig1['address.street'].str.split(',').str[0] + ', ' \
                       + geo_bolig1['address.postalId'].astype(str)
df_Boligsiden_Dingeo_Hvorlangterder1 = add_hvorlangterder(geo_bolig1)
df_Boligsiden_Dingeo_Hvorlangterder1.to_csv('bdh_1_K7_21.csv', index=False)


geo_bolig1 = pd.read_csv("boligsiden_dingeo_1_F7_17-21.csv")
geo_bolig1['Location'] = geo_bolig1['address.street'].str.split(',').str[0] + ', ' \
                       + geo_bolig1['address.postalId'].astype(str)
df_Boligsiden_Dingeo_Hvorlangterder1 = add_hvorlangterder(geo_bolig1)
df_Boligsiden_Dingeo_Hvorlangterder1.to_csv('bdh_1_F7_17-21.csv', index=False)

bdh_1_F = pd.read_csv("bdh_1_F7_17-21.csv")
bdh_1_K1 = pd.read_csv("bdh_1_K7_17.csv")
bdh_1_K2 = pd.read_csv("bdh_1_K7_18.csv")
bdh_1_K3 = pd.read_csv("bdh_1_K7_19.csv")
bdh_1_K4 = pd.read_csv("bdh_1_K7_20.csv")
bdh_1_K5 = pd.read_csv("bdh_1_K7_21.csv")


raw_data_1 = pd.concat([bdh_1_F, bdh_1_K1, bdh_1_K2, bdh_1_K3, bdh_1_K4,
                        bdh_1_K5], sort=False)
raw_data_1.to_csv('raw_data_1.csv', index=False)



# * ## Collecting all data

# In[ ]:

# All code below is commented out as not to risk running the very time consuming scraping process again


############### Get Boligsiden-data ###############

# links = get_all_links_boligsiden('København', '2016-01-01', '2020-12-31') # This was also done for Frederiksberg
#
# with open('links_boligsiden_K.txt', 'w') as file:
#        file.write(str(links))
#
# with open("links_boligsiden_K.txt", "r") as file:
#   links = eval(file.readline())
#
# df1, df2 = get_data_boligsiden(links[30000:])
# df1.to_csv('boligsiden_1_K7.csv', index=False)
# df2.to_csv('boligsiden_2_K7.csv', index=False)
# # The above was done 7 times for Copenhagen and 1 time for Frederiksberg, as to keep data frames small


# # This gives data for 41024, where the df2's are dropped since they have missing data.
# # In most cases this is due to that the unit is still for sale.
# # Other addresses were likewise dropped due to problem with Boligsiden page.


############### Get Dingeo-data ##################

# df_Boligsiden1 = pd.read_csv("boligsiden_1_K7.csv")
# df_Boligsiden_Geo1 = get_geolinks1(df_Boligsiden1) #[:]) #Here we choose how many to do at a time
# df_Boligsiden_Dingeo1 = add_dingeo(df_Boligsiden_Geo1)
# # pd.set_option('display.max_columns', None)
# # print(df_Boligsiden_Dingeo1)
# df_Boligsiden_Dingeo1.to_csv('boligsiden_dingeo_1_K7.csv', index=False)

# # Data is lost for adresses where there is a problem with the address name or missing data on the DinGeo-page.
# # This results in a data set of 40657 addresses

########## Get Hvorlangterder data ##############

# geo_bolig1 = pd.read_csv("boligsiden_dingeo_1_K7.csv")
# geo_bolig1['Location'] = geo_bolig1['address.street'].str.split(',').str[0] + ', ' \
#                        + geo_bolig1['address.postalId'].astype(str)
# df_Boligsiden_Dingeo_Hvorlangterder1 = add_hvorlangterder(geo_bolig1)
# df_Boligsiden_Dingeo_Hvorlangterder1.to_csv('bdh_1_K7.csv', index=False)

# # Again loss of addresses, where no information is available on the hvorlangerder.dk web page
# # This finally gives 40606 adresses.

########### Creating final raw data frame ##########

# bdh_1_F = pd.read_csv("bdh_1_F.csv")
# bdh_1_K1 = pd.read_csv("bdh_1_K1.csv")
# bdh_1_K2 = pd.read_csv("bdh_1_K2.csv")
# bdh_1_K3 = pd.read_csv("bdh_1_K3.csv")
# bdh_1_K4 = pd.read_csv("bdh_1_K4.csv")
# bdh_1_K5 = pd.read_csv("bdh_1_K5.csv")
# bdh_1_K6 = pd.read_csv("bdh_1_K6.csv")
# bdh_1_K7 = pd.read_csv("bdh_1_K7.csv")

#
# raw_data_1 = pd.concat([bdh_1_F, bdh_1_K1, bdh_1_K2, bdh_1_K3, bdh_1_K4,
#                         bdh_1_K5, bdh_1_K6, bdh_1_K7], sort=False)
# raw_data_1.to_csv('raw_data_1.csv', index=False)

############################################################
##########      SCRAPING OF OLD SALES PRICES      ##########
############################################################

#pd.set_option('display.max_columns', None)


links_17 = []
links_18 = []
links_19 = []
links_20 = []
links_21 = []
links_F = []

with open("links_boligsiden_K_17.txt", "r") as file:
   links_17 = eval(file.readline())
with open("links_boligsiden_K_18.txt", "r") as file:
   links_18 = eval(file.readline())
with open("links_boligsiden_K_19.txt", "r") as file:
   links_19 = eval(file.readline())
with open("links_boligsiden_K_20.txt", "r") as file:
   links_20 = eval(file.readline())
with open("links_boligsiden_K_21.txt", "r") as file:
   links_21 = eval(file.readline())
with open("links_boligsiden_F_17-21.txt", "r") as file:
   links_F = eval(file.readline())

links_all = links_17 + links_18 + links_19 + links_20 + links_21 + links_F
len(links_all)
links_nodub = list(dict.fromkeys(links_all))
links_5test = links_nodub[:5]

def get_hist_prices(url):

    url = url
    html = urlopen(url)
    soup = BeautifulSoup(html.read(), 'html.parser')
    head = str(soup.find('head'))
    try:
        json_string = re.search(r'__bs_presentation__ = ([^;]*)', head).group(1)
        data = json.loads(json_string)
        df1 = pd.json_normalize(data)
    except:
        pass

    return df1

#### Collect scraped information for all addresses in two dataframes

def get_hist_prices_data_boligsiden(links):
    df1 = pd.DataFrame()

    for x in tqdm(range(0, len(links))):
        try:
            df_pages1 = get_hist_prices(links[x])
            df1 = pd.concat([df1, df_pages1])
        except:
            pass


    return df1


len(links_nodub) #41035
# 41035/4 = 10258
links_1 = links_nodub[0:10258]
links_2 = links_nodub[10258:20516]
links_3 = links_nodub[20516:30774]
links_4 = links_nodub[30774:]

hist_prices_1 = get_hist_prices_data_boligsiden(links_1)
hist_prices_2 = get_hist_prices_data_boligsiden(links_2)
hist_prices_3 = get_hist_prices_data_boligsiden(links_3)
hist_prices_4 = get_hist_prices_data_boligsiden(links_4)

all_hist_prices_tmp = [hist_prices_1, hist_prices_2, hist_prices_3, hist_prices_4]
all_hist_prices = pd.concat(all_hist_prices_tmp)
len(all_hist_prices) #40932
# all_hist_prices.to_csv('all_hist_prices.csv', index = False)
# all_hist_prices = pd.read_csv('all_hist_prices.csv')
all_hist_prices.info(verbose = True)
hist_prices = all_hist_prices[['property.mapPosition.latLng.lat', 'property.mapPosition.latLng.lng',
                               'property.accessAddressId', 'saveEnergy.address', 'historic.items']]
hist_prices['saveEnergy.address'].isna().sum() #is.na = 382
hist_prices.to_csv('hist_prices.csv', index = False)

hist_prices['historic.items'].isna().sum() # 7 to be removed
hist_prices['saveEnergy.address'].isna().sum() #is.na = 382 to be removed
hist_prices = hist_prices[~hist_prices['saveEnergy.address'].isna()]
len(hist_prices) #40543



def nearest(items, pivot):
    df_clean = items.drop(items.loc[(items['dateTime'] == pivot) | (~items['event_bs'].isin([1,2]))].index)
    if df_clean.empty:
        return 0
    return df_clean.loc[(pd.to_datetime(df_clean['dateTime']) - pd.to_datetime(pivot)).abs().idxmin(), ['dateTime', 'event_bs']]


def expand_old_prices(df):
    df1 = pd.DataFrame()
    df1_tmp = pd.DataFrame()
    fail_list = []

    for x in tqdm(range(0, len(df))):
        try:
            df1_tmp = pd.DataFrame(literal_eval(df['historic.items'][x]))
            df1_tmp['Address'] = df['saveEnergy.address'][x]
            df1_tmp = df1_tmp.rename(columns={"event": "event_bs", "paymentCash": "paymentCash_bs"})
            df1_tmp = pd.concat([df1_tmp, df1_tmp['propertyHistoricModel'].apply(pd.Series)], axis = 1)
            if df1_tmp.shape[1] == 9:
                df1_tmp[['dateAdded',	'dateRemoved', 'paymentCash', 'downPayment', 'paymentExpenses',
                         'areaResidential', 'areaWeighted','numberOfFloors',
                         'numberOfRooms', 'salesPeriod', 'priceDevelopment', 'areaParcel', 'event', 'itemType']] = np.nan
            df1_tmp['dateTime'] = df1_tmp['dateTime'].apply(lambda x: x[:10])
            df1_tmp['dateTime'] = pd.to_datetime(df1_tmp['dateTime'],
                                              format='%Y-%m-%d', errors='coerce')
            sale_list_internal = (df1_tmp.loc[df1_tmp['event_bs'] == 3]['dateTime']).to_list()
            if len(sale_list_internal) > 0:
                for i in range(0, len(sale_list_internal)):
                    date_key_internal = nearest(df1_tmp, sale_list_internal[i])
                    if not isinstance(date_key_internal, int):
                        if ((date_key_internal[0] - pd.to_datetime(df1_tmp.loc[df1_tmp['dateTime'] == sale_list_internal[i], 'dateTime'].values)[0] <= timedelta(days=69))
                            & (date_key_internal[0] - pd.to_datetime(df1_tmp.loc[df1_tmp['dateTime'] == sale_list_internal[i], 'dateTime'].values)[0] >= timedelta(days=-69))):
                            print(df1_tmp['Address'])
                            print(df1_tmp.loc[df1_tmp['dateTime'] == date_key_internal[0], 'dateAdded':'itemType'].values)
                            df1_tmp.loc[df1_tmp['dateTime'] == sale_list_internal[i], 'dateAdded':'itemType'] = \
                                df1_tmp.loc[(df1_tmp['dateTime'] == date_key_internal[0]) & (df1_tmp['event_bs'] == date_key_internal[1]), 'dateAdded':'itemType'].values[0]
            df1 = pd.concat([df1, df1_tmp])
        except KeyError:
            fail_list.append(x)
            print('Failure at index: ', x)
    return df1, fail_list

hist_prices_expanded, fail_list = expand_old_prices(hist_prices)
len(fail_list) # 547
hist_prices_expanded.head()
hist_prices_expanded.to_csv('hist_prices_expanded.csv', index = False)

hist_prices_expanded_1, fail_list_1 = expand_old_prices(hist_prices)
hist_prices_expanded_1.head()
hist_prices_expanded_1.to_csv('hist_prices_expanded_1.csv', index = False)

hist_exp_test = expand_old_prices(hist_prices)
len(hist_exp_test[1])
hist_exp_test[0].to_csv('final_hist_prices.csv', index = False)
pd.DataFrame(hist_exp_test[1]).to_csv('final_hist_prices_faillist.csv', index = False)


#property.mapPosition.latLng.lat
#property.mapPosition.latLng.lng
#property.accessAddressId
#saveEnergy.address
#historic.items
