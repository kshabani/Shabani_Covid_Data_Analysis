import pandas as pd
import numpy as np
import logging

from datetime import datetime

import pandas as pd
import numpy as np
import logging
from datetime import datetime


def store_relational_JH_data():
    ''' Transformes the COVID data in a relational data set

    '''

    data_path = '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/raw/COVID-19/' \
                'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw = pd.read_csv(data_path)

    pd_data_base = pd_raw.rename(columns={'Country/Region': 'country',
                                          'Province/State': 'state'})

    pd_data_base['state'] = pd_data_base['state'].fillna('no')

    pd_data_base = pd_data_base.drop(['Lat', 'Long'], axis=1)

    pd_relational_model = pd_data_base.set_index(['state', 'country']).T.stack(level=[0, 1]).reset_index().rename(
        columns={'level_0': 'date',
                 0: 'confirmed'},
        )

    pd_relational_model['date'] = pd_relational_model.date.astype('datetime64[ns]')
    logging.info("the dates are: {}".format(pd_relational_model.date))

    pd_relational_model.to_csv(
        '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/processed/COVID_relational_confirmed.csv',
        sep=';', index=False)

    logging.warning("Relational data structure generated !")
    logging.info(' Number of rows stored: ' + str(pd_relational_model.shape[0]))

def main():
    store_relational_JH_data()

