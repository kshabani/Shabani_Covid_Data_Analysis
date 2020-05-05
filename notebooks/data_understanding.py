#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import os

import pandas as pd

import requests
from bs4 import BeautifulSoup

import json


pd.set_option('display.max_rows', 500)


# ![CRISP_DM](../reports/figures/CRISP_DM.png)

# # Data Understanding

# * RKI, webscrape (webscraping) https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Fallzahlen.html
# * John Hopkins (GITHUB) https://github.com/CSSEGISandData/COVID-19.git
# * REST API services to retreive data https://npgeo-corona-npgeo-de.hub.arcgis.com/

# ## GITHUB csv data
# 
# git clone/pull https://github.com/CSSEGISandData/COVID-19.git

# In[2]:



git_pull = subprocess.Popen( "/usr/bin/git pull" , 
                     cwd = os.path.dirname( '../data/raw/COVID-19/' ), 
                     shell = True, 
                     stdout = subprocess.PIPE, 
                     stderr = subprocess.PIPE )
(out, error) = git_pull.communicate()


print("Error : " + str(error)) 
print("out : " + str(out))


# In[3]:


data_path='../data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
pd_raw=pd.read_csv(data_path)


# In[4]:


pd_raw.head()


# ## Webscrapping

# In[5]:


page = requests.get("https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Fallzahlen.html")


# In[6]:


soup = BeautifulSoup(page.content, 'html.parser')


# In[7]:


html_table=soup.find('table') # find the table, attention this works if one table exists


# In[8]:


all_rows=html_table.find_all('tr')


# In[9]:


final_data_list=[]


# In[10]:


for pos,rows in enumerate(all_rows):
   
    col_list=[each_col.get_text(strip=True) for each_col in rows.find_all('td')] #td for data element
    final_data_list.append(col_list)
    


# In[11]:


pd_daily_status=pd.DataFrame(final_data_list).dropna().rename(columns={0:'state',
                                                       1:'cases',
                                                       2:'changes',
                                                       3:'cases_per_100k',
                                                       4:'fatal',
                                                       5:'comment'})


# In[12]:


pd_daily_status.head()


# ## REST API calls

# In[13]:


## data request for Germany
data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/Coronaf%C3%A4lle_in_den_Bundesl%C3%A4ndern/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')


# In[14]:


json_object=json.loads(data.content) 


# In[15]:


type(json_object)


# In[16]:


json_object.keys()


# In[17]:


full_list=[]
for pos,each_dict in enumerate (json_object['features'][:]):
    full_list.append(each_dict['attributes'])
    


# In[18]:


pd_full_list=pd.DataFrame(full_list)
pd_full_list.head()


# In[19]:



pd_full_list.to_csv('../data/raw/NPGEO/GER_state_data.csv',sep=';')


# In[20]:


pd_full_list.shape[0]


# # API access via REST service, e.g. USA data 
# 
# example of a REST conform interface (attention registration mandatory)
# 
# www.smartable.ai

# In[21]:


import requests

# US for full list
headers = {
    'Cache-Control': 'no-cache',
    'Subscription-Key': '28ee4219700f48718be78b057beb',
}

response = requests.get('https://api.smartable.ai/coronavirus/stats/US', headers=headers)
print(response)


# In[22]:



US_dict=json.loads(response.content) # imports string
with open('../data/raw/SMARTABLE/US_data.txt', 'w') as outfile:
    json.dump(US_dict, outfile,indent=2)


# In[23]:


print(json.dumps(US_dict,indent=2)) #string dump


# In[ ]:




