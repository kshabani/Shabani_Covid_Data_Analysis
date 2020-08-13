import subprocess
import os

import pandas as pd
import logging
import numpy as np

from datetime import datetime

import requests
import json


def get_johns_hopkins():
    ''' Get data by a git pull request, the source code has to be pulled first
        Result is stored in the predifined csv structure
    '''
    git_pull = subprocess.Popen("/usr/bin/git pull",
                                cwd=os.path.dirname(
                                    '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/raw/COVID-19/'),
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    (out, error) = git_pull.communicate()

    logging.info("Branch Paths : " + str(error))
    logging.info("File Update : " + str(out))
    logging.warning("Full Data Successfully downloaded!")


def COVID_SIR_DATA():
    data_path = '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/' \
                'raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw = pd.read_csv(data_path).copy()
    pd_raw.drop(["Province/State", "Lat", "Long"], axis=1, inplace=True)
    country_list = list(pd_raw['Country/Region'].unique())
    df = pd.DataFrame([])

    for each in country_list:
        series = pd.Series(pd_raw[pd_raw['Country/Region'] == each].sum(), name=each)
        df[each] = series
    df.drop('Country/Region', axis=0, inplace=True)
    pd.to_pickle(df, '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/processed/COVID_SIR.pkl')
    logging.warning("SIR Dataframe generated! ")

def main():
    get_johns_hopkins()
    COVID_SIR_DATA()
