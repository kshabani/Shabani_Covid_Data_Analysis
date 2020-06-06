import pandas as pd
import numpy as np
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
from datetime import datetime
from scipy import optimize
from scipy import integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

df_input_large = pd.read_csv('/media/sem/HDD/Home_Programming/Git/ads_covid-19-sem/data/processed/COVID_final_set.csv',
                             sep=';')
df_analyse = pd.read_csv(
    '/media/sem/HDD/Home_Programming/Git/ads_covid-19-sem/data/processed/COVID_small_flat_table.csv', sep=';')

colors = {'background': '#111111', 'text': '#7FDBFF'}

N0 = 1000000  # max susceptible population
beta = 0.4  # infection spread dynamics
gamma = 0.1  # recovery rate


def SIR_model(SIR, beta, gamma):
    ''' Simple SIR model
        S: susceptible population
        I: infected people
        R: recovered people
        beta:

        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)

    '''

    S, I, R = SIR
    dS_dt = -beta * S * I / N0  # S*I is the
    dI_dt = beta * S * I / N0 - gamma * I
    dR_dt = gamma * I
    return ([dS_dt, dI_dt, dR_dt])


def SIR_model_t(SIR, t, beta, gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta:

        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)

    '''

    S, I, R = SIR
    dS_dt = -beta * S * I / N0  # S*I is the
    dI_dt = beta * S * I / N0 - gamma * I
    dR_dt = gamma * I
    return dS_dt, dI_dt, dR_dt


def fit_odeint(x, beta, gamma):
    '''
    helper function for the integration
    '''
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:, 1]


ydata = np.array(df_analyse.Germany[35:])
t = np.arange(len(ydata))

I0 = ydata[0]
S0 = N0 - I0
R0 = 0

fig = go.Figure()

app = dash.Dash()

tab_1 = dcc.Tab(label='Analysis of Rate of infection', value='tab_1', children=[

    dcc.Dropdown(
        id='country_drop_down',
        options=[{'label': each, 'value': each} for each in df_input_large['country'].unique()],
        value=['US', 'Germany', 'Italy'],  # which are pre-selected
        multi=True
    ),
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
    )
]
                )

tab_2 = dcc.Tab(label='SIR Model Demonstration For Germany', value='tab_2', children=[
    dcc.Dropdown(id='countries_2', options=['Germany'],
                 value='Germany', multi=False)
])

app.layout = html.Div(
    [html.Center(html.H1('Covid19 Data Analysis')), dcc.Tabs(id='my_tabs', value='tab_1', children=[tab_1, tab_2]),
     html.Div(html.Center([dcc.Graph(figure=fig, id='main_window_slope')]))])


@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('my_tabs', 'value'),
     Input('country_drop_down', 'value'),
     Input('doubling_time', 'value'),
     Input('countries_2', 'value')])
def update_figure(tab, country_list, show_doubling, country_1):
    if tab == 'tab_1':

        if 'doubling_rate' in show_doubling:
            my_yaxis = {'type': "log",
                        'title': 'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'}
        else:
            my_yaxis = {'type': "log",
                        'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                        }

        traces = []
        for each in country_list:

            df_plot = df_input_large[df_input_large['country'] == each]

            if show_doubling == 'doubling_rate_filtered':
                df_plot = df_plot[
                    ['state', 'country', 'confirmed', 'confirmed_filtered', 'doubling_rate', 'doubling_rate_filtered',
                     'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
            else:
                df_plot = df_plot[
                    ['state', 'country', 'confirmed', 'confirmed_filtered', 'doubling_rate', 'doubling_rate_filtered',
                     'date']].groupby(['country', 'date']).agg(np.sum).reset_index()

            traces.append(go.Scatter(x=df_plot.date,
                                     y=df_plot[show_doubling],
                                     mode='markers+lines',
                                     opacity=0.9,
                                     name=each)
                          )
            layout = go.Layout(
                width=1280,
                height=720,
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font={'color': colors['text']},
                xaxis={'title': 'Timeline',
                       'tickangle': -45,
                       'nticks': 20,
                       'tickfont': dict(size=14, color="#7f7f7f"),
                       },
                yaxis=my_yaxis
            )

        return dict(data=traces, layout=layout)

    else:

        ydata = np.array(df_analyse.Germany[35:])
        t = np.arange(len(ydata))
        I0 = ydata[0]
        S0 = N0 - I0
        R0 = 0
        popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
        fitted = fit_odeint(t, *popt)

        t_initial = 28
        t_intro_measures = 14
        t_hold = 21
        t_relax = 21

        beta_max = 0.4
        beta_min = 0.11
        gamma = 0.1
        pd_beta = np.concatenate((np.array(t_initial * [beta_max]),
                                  np.linspace(beta_max, beta_min, t_intro_measures),
                                  np.array(t_hold * [beta_min]),
                                  np.linspace(beta_min, beta_max, t_relax),
                                  ))
        SIR = np.array([S0, I0, R0])
        propagation_rates = pd.DataFrame(columns={'susceptible': S0,
                                                  'infected': I0,
                                                  'recoverd': R0})
        for each_beta in pd_beta:
            new_delta_vec = SIR_model(SIR, each_beta, gamma)

            SIR = SIR + new_delta_vec

            propagation_rates = propagation_rates.append({'susceptible': SIR[0],
                                                          'infected': SIR[1],
                                                          'recovered': SIR[2]}, ignore_index=True)
        t_phases = np.array([t_initial, t_intro_measures, t_hold, t_relax]).cumsum()
        fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{"colspan": 2}, None]], subplot_titles=(
            "Fit of SIR model for Germany cases",
            'Szenario SIR simulations')
                            )
        trace11 = go.Scatter(x=t, y=ydata, mode='markers')
        trace22 = go.Scatter(x=t, y=fitted, mode='lines')
        trace111 = go.Scatter(x=propagation_rates.index, y=propagation_rates.infected, name='infected', mode='lines',
                              line=dict(width=5))
        trace222 = go.Bar(x=np.arange(len(ydata)), y=ydata, name='current infected germany')

        fig.add_trace(trace11, row=1, col=1)
        fig.add_trace(trace22, row=1, col=1)
        fig.add_trace(trace111, row=2, col=1)
        fig.add_trace(trace222, row=2, col=1)

        fig.update_yaxes(type='log', row=1, col=1)
        fig.update_yaxes(type='log', row=2, col=1)
        fig.update_layout(plot_bgcolor=colors['background'],
                          paper_bgcolor=colors['background'],
                          font={'color': colors['text']})

        return fig

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)