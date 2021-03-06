import numpy as np
from sklearn import linear_model
import pandas as pd
import logging

from scipy import signal

reg = linear_model.LinearRegression(fit_intercept=True)

import numpy as np
from sklearn import linear_model

reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal


def make_relatinoal_data_struture():
    path_save = '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/processed/COVID_relational_confirmed.csv'
    data_path = '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw = pd.read_csv(data_path).copy()

    pd_data_base = pd_raw.rename(columns={'Country/Region': 'country', 'Province/State': 'state'})
    pd_data_base = pd_data_base.drop(['Lat', 'Long'], axis=1)

    test_pd = pd_data_base.set_index(['state', 'country']).T
    pd_relational_model = test_pd.stack(level=[0, 1]).reset_index().rename(columns={'level_0': 'date', 0: 'confirmed'})
    pd_relational_model['date'] = pd_relational_model.date.astype('datetime64[ns]')
    logging.info("The dates are {}".format(pd_relational_model.date))
    pd_relational_model.to_csv(path_save, sep=';')

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
    X = np.arange(-1, 2).reshape(-1, 1)

    assert len(in_array) == 3
    reg.fit(X, y)
    intercept = reg.intercept_
    slope = reg.coef_

    return intercept / slope

def rolling_reg(df_input, col='confirmed'):
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
    days_back = 3
    result = df_input[col].rolling(
        window=days_back,
        min_periods=days_back).apply(get_doubling_time_via_regression, raw=False)
    return result


def calc_doubling_rate(df_input, filter_on='confirmed'):
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

    must_contain = set(['state', 'country', filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    pd_DR_result = df_input[['state','country',filter_on]].groupby(['state', 'country']).apply(rolling_reg, filter_on).reset_index()
    pd_DR_result = pd_DR_result.rename(columns={filter_on: filter_on +'_doubling_rate',
                                                'level_2': 'index'})

    df_input = df_input.reset_index()
    df_output = pd.merge(df_input, pd_DR_result[['index', filter_on + '_doubling_rate']], on=['index'], how='left')
    logging.warning("Doubling rate for {} calculated".format(filter_on))
    return df_output

def savgol_filter(df_input, column='confirmed', window=5):
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

    degree = 1
    df_result = df_input

    filter_in = df_input[column].fillna(0)  # attention with the neutral element here

    result = signal.savgol_filter(np.array(filter_in),
                                  window,  # window size used for filtering
                                  1)
    df_result[column + '_filtered'] = result
    return df_result

def calc_filtered_data(df_input, filter_on='confirmed'):
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

    must_contain = set(['state', 'country', filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    pd_filtered_result = df_input[['state', 'country', filter_on]].groupby(['state', 'country']).apply(
        savgol_filter).reset_index()
    df_output = pd.merge(df_input, pd_filtered_result[['index', filter_on + '_filtered']], on=['index'], how='left')

    logging.warning("Doubling rate for {} calculated".format(filter_on))
    return df_output

def main():
    path_main = '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/processed/COVID_relational_confirmed.csv'

    make_relatinoal_data_struture()
    pd_JH_data = pd.read_csv(path_main, sep=';', parse_dates=[0])

    #pd_JH_data = pd_JH_data.sort_values('date', ascending=True).reset_index().copy()

    pd_JH_data = pd_JH_data.sort_values('date', ascending=True).reset_index(drop=True).copy()
    pd_JH_data = pd_JH_data.drop(['Unnamed: 0'], axis=1).copy()
    #p = pd_JH_data
    pd_result_larg = calc_doubling_rate(pd_JH_data) # attach doubling rate for confirmed

    pd_result_larg = calc_filtered_data(pd_result_larg) # attach filter of confirmed

    pd_result_larg = calc_doubling_rate(pd_result_larg, 'confirmed_filtered') # attach doubling rate of confirmed

    pd_result_larg.to_csv('../data/processed/COVID_final_set_.csv', sep=';', index=False)
    logging.warning("COVID_final_set_.csv saved")

