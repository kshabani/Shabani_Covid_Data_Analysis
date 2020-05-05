#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np

from datetime import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 500)


# ![CRISP_DM](../reports/figures/CRISP_DM.png)

# # Groupby apply on large (relational) data set
# 
# ## Attentions all writen functions assume a data frame where the date is sorted!!

# In[178]:


pd_JH_data=pd.read_csv('../data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
pd_JH_data=pd_JH_data.sort_values('date',ascending=True).reset_index(drop=True).copy()
pd_JH_data.head()


# # Test data

# In[179]:


test_data=pd_JH_data[((pd_JH_data['country']=='US')|
                      (pd_JH_data['country']=='Germany'))&
                     (pd_JH_data['date']>'2020-03-20')]


# In[180]:


test_data.head()


# In[181]:


test_data.groupby(['country']).agg(np.max)


# In[182]:


# %load ../src/features/build_features.py

import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)

def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate'''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope



# In[183]:


test_data.groupby(['state','country']).agg(np.max)


# In[184]:


# this command will only work when adapting the get_doubling_time_via_regression function

#test_data.groupby(['state','country']).apply(get_doubling_time_via_regression)


# In[185]:


def rolling_reg(df_input,col='confirmed'):
    ''' input has to be a data frame'''
    ''' return is single series (mandatory for group by apply)'''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)
    return result
    


# In[186]:


test_data[['state','country','confirmed']].groupby(['state','country']).apply(rolling_reg,'confirmed')


# In[187]:


pd_DR_result=pd_JH_data[['state','country','confirmed']].groupby(['state','country']).apply(rolling_reg,'confirmed').reset_index()


# In[188]:


pd_DR_result=pd_DR_result.rename(columns={'confirmed':'doubling_rate',
                             'level_2':'index'})
pd_DR_result.head()


# In[189]:


pd_JH_data=pd_JH_data.reset_index()
pd_JH_data.head()


# In[190]:


pd_result_larg=pd.merge(pd_JH_data,pd_DR_result[['index','doubling_rate']],on=['index'],how='left')
pd_result_larg.head()


# In[191]:


#pd_result_larg[pd_result_larg['country']=='Germany']


# # Filtering the data with groupby apply 

# In[192]:


from scipy import signal

def savgol_filter(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function 
        it ensures that the data structure is kept'''
    window=5, 
    degree=1
    df_result=df_input
    
    filter_in=df_input[column].fillna(0) # attention with the neutral element here
    
    result=signal.savgol_filter(np.array(filter_in),
                           5, # window size used for filtering
                           1)
    df_result[column+'_filtered']=result
    return df_result
        


# In[193]:


pd_filtered_result=pd_JH_data[['state','country','confirmed']].groupby(['state','country']).apply(savgol_filter).reset_index()


# In[194]:


pd_result_larg=pd.merge(pd_result_larg,pd_filtered_result[['index','confirmed_filtered']],on=['index'],how='left')
pd_result_larg.head()


# # Filtered doubling rate

# In[195]:



pd_filtered_doubling=pd_result_larg[['state','country','confirmed_filtered']].groupby(['state','country']).apply(rolling_reg,'confirmed_filtered').reset_index()

pd_filtered_doubling=pd_filtered_doubling.rename(columns={'confirmed_filtered':'doubling_rate_filtered',
                             'level_2':'index'})

pd_filtered_doubling.head()


# In[ ]:





# In[196]:


pd_result_larg=pd.merge(pd_result_larg,pd_filtered_doubling[['index','doubling_rate_filtered']],on=['index'],how='left')
pd_result_larg.head()


# In[197]:


mask=pd_result_larg['confirmed']>100
pd_result_larg['doubling_rate_filtered']=pd_result_larg['doubling_rate_filtered'].where(mask, other=np.NaN) 


# In[198]:


pd_result_larg.head()


# In[199]:


pd_result_larg.to_csv('../data/processed/COVID_final_set.csv',sep=';',index=False)


# In[ ]:




