#!/usr/bin/env python
# coding: utf-8

# # One run full walktrhough

# * Do the full walk through on the large data set
# * Refactor the source code and bring it to individual scripts
# * Ensure a full run with one click

# In[13]:


## check some parameters
## depending where you launch your notebook, the relative path might not work
## you should start the notebook server from your base path
## when opening the notebook, typically your path will be ../ads_covid-19/notebooks
import os
if os.path.split(os.getcwd())[-1]=='notebooks':
    os.chdir("../")

'Your base path is at: '+os.path.split(os.getcwd())[-1]


# ## 1 Update all data

# In[23]:


# %load src/data/get_data.py

import subprocess
import os

import pandas as pd
import numpy as np

from datetime import datetime

import requests
import json

def get_johns_hopkins():
    ''' Get data by a git pull request, the source code has to be pulled first
        Result is stored in the predifined csv structure
    '''
    git_pull = subprocess.Popen( "/usr/bin/git pull" ,
                         cwd = os.path.dirname( 'data/raw/COVID-19/' ),
                         shell = True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE )
    (out, error) = git_pull.communicate()


    print("Error : " + str(error))
    print("out : " + str(out))


def get_current_data_germany():
    ''' Get current data from germany, attention API endpoint not too stable
        Result data frame is stored as pd.DataFrame

    '''
    # 16 states
    #data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/Coronaf%C3%A4lle_in_den_Bundesl%C3%A4ndern/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    # 400 regions / Landkreise
    data=requests.get('https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_Landkreisdaten/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json')

    json_object=json.loads(data.content)
    full_list=[]
    for pos,each_dict in enumerate (json_object['features'][:]):
        full_list.append(each_dict['attributes'])

    pd_full_list=pd.DataFrame(full_list)
    pd_full_list.to_csv('data/raw/NPGEO/GER_state_data.csv',sep=';')
    print(' Number of regions rows: '+str(pd_full_list.shape[0]))

if __name__ == '__main__':
    get_johns_hopkins()
    get_current_data_germany()


# ## 2. Process pipeline

# In[ ]:


# %load src/data/process_JH_data.py
import pandas as pd
import numpy as np

from datetime import datetime


def store_relational_JH_data():
    ''' Transformes the COVID data in a relational data set

    '''

    data_path='data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw=pd.read_csv(data_path)

    pd_data_base=pd_raw.rename(columns={'Country/Region':'country',
                      'Province/State':'state'})

    pd_data_base['state']=pd_data_base['state'].fillna('no')

    pd_data_base=pd_data_base.drop(['Lat','Long'],axis=1)


    pd_relational_model=pd_data_base.set_index(['state','country'])                                 .T                                                              .stack(level=[0,1])                                             .reset_index()                                                  .rename(columns={'level_0':'date',
                                                   0:'confirmed'},
                                                  )

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')

    pd_relational_model.to_csv('data/processed/COVID_relational_confirmed.csv',sep=';',index=False)
    print(' Number of rows stored: '+str(pd_relational_model.shape[0]))

if __name__ == '__main__':

    store_relational_JH_data()


# ## 3  Filter and Doubling Rate Calculation

# In[20]:


# %load src/features/build_features.py

import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal


def get_doubling_time_via_regression(in_array):
    ''' Use a linear regression to approximate the doubling rate

        Parameters:
        ----------
        in_array : pandas.series

        Returns:
        ----------
        Doubling rate: double
    '''

    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1, 1)

    assert len(in_array)==3
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope


def savgol_filter(df_input,column='confirmed',window=5):
    ''' Savgol Filter which can be used in groupby apply function (data structure kept)

        parameters:
        ----------
        df_input : pandas.series
        column : str
        window : int
            used data points to calculate the filter result

        Returns:
        ----------
        df_result: pd.DataFrame
            the index of the df_input has to be preserved in result
    '''

    degree=1
    df_result=df_input

    filter_in=df_input[column].fillna(0) # attention with the neutral element here

    result=signal.savgol_filter(np.array(filter_in),
                           window, # window size used for filtering
                           1)
    df_result[column+'_filtered']=result
    return df_result

def rolling_reg(df_input,col='confirmed'):
    ''' Rolling Regression to approximate the doubling time'

        Parameters:
        ----------
        df_input: pd.DataFrame
        col: str
            defines the used column
        Returns:
        ----------
        result: pd.DataFrame
    '''
    days_back=3
    result=df_input[col].rolling(
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)
    return result




def calc_filtered_data(df_input,filter_on='confirmed'):
    '''  Calculate savgol filter and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    pd_filtered_result=df_input[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter).reset_index()
    df_output=pd.merge(df_input,pd_filtered_result[['index',filter_on+'_filtered']],on=['index'],how='left')

    return df_output





def calc_doubling_rate(df_input,filter_on='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()
    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    df_output=pd.merge(df_input,pd_DR_result[['index',filter_on+'_DR']],on=['index'],how='left')
    return df_output


if __name__ == '__main__':
    test_data_reg=np.array([2,4,6])
    result=get_doubling_time_via_regression(test_data_reg)
    print('the test slope is: '+str(result))

    pd_JH_data=pd.read_csv('data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).reset_index().copy()

    pd_result_larg=calc_filtered_data(pd_JH_data)
    pd_result_larg=calc_doubling_rate(pd_result_larg)
    pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')
    print(pd_result_larg.head())


# ## 4 Visual Board

# In[21]:


# %load src/visualization/visualize.py
import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

import plotly.graph_objects as go

import os
print(os.getcwd())
df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Applied Data Science on COVID-19 data

    Goal of the project is to teach data science by applying a cross industry standard process,
    it covers the full walkthrough of: automated data gathering, data transformations,
    filtering and machine learning to approximating the doubling time, and
    (static) deployment of responsive dashboard.

    '''),

    dcc.Markdown('''
    ## Multi-Select Country for visualization
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['US', 'Germany','Italy'], # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'doubling_rate'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'doubling_rate_filtered'},
    ],
    value='confirmed',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])
def update_figure(country_list,show_doubling):


    if 'doubling_rate' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }


    traces = []
    for each in country_list:

        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','doubling_rate','doubling_rate_filtered','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','doubling_rate','doubling_rate_filtered','date']].groupby(['country','date']).agg(np.sum).reset_index()
        #print(show_doubling)


        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=each
                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }

if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False)


# In[ ]:
