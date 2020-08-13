import pandas as pd
import numpy as np
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from datetime import datetime
from scipy import optimize
from scipy import integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pickle
from plotly.subplots import make_subplots

def app_func():
    path1 = '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/processed/COVID_final_set_.csv'
    path2 = '/mnt/368AE7F88AE7B313/Files_Programming/Git/ads_covid-19-sem/data/processed/COVID_SIR.pkl'

    # df_input_large = pd.read_csv(path1,sep=';')
    df_input_large = pd.read_csv(path1, sep=';')
    df_input_large = df_input_large[df_input_large['date'] > '2020-03-01']
    df_analyse = pd.read_pickle(path2)

    colors = {'background': '#111111', 'text': '#7FDBFF'}

    fig = go.Figure()

    app = dash.Dash()

    tab_1 = dcc.Tab(label='Analysis of Rate of infection', value='tab_1', children=[
        html.H3("Countries(We have {} countries available)".format(len(df_input_large.country))),
        dcc.Dropdown(
            id='country_drop_down',
            options=[{'label': each, 'value': each} for each in df_input_large['country'].unique()],
            value=['US', 'Germany', 'Italy'],  # which are pre-selected
            multi=True
        ),
        html.H3("Type of Graph"),
        dcc.Dropdown(
            id='doubling_time',
            options=[
                {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
                {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
                {'label': 'Timeline Doubling Rate', 'value': 'confirmed_doubling_rate'},
                {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_doubling_rate'},
            ],
            value='confirmed',
            multi=False
        ),

        dcc.Markdown('''
                Regarding the filtration of data and doubling rate calculation, the following techniques are used.
                    * The savgol signal filtration was used to filter the data mainly to smoothen reporting delays and 
                     human errors in reporting.A window  size of five data points was used.
                    *  The doubling rate was calculated via rolling regression with a window size of 3 days back. 
        ''')
    ]
                    )

    tab_2 = dcc.Tab(label='SIR Model Demonstration', value='tab_2', children=[
        html.H3("N0: max susceptible population"),
        dcc.Slider(
            id='N0s',
            min=0,
            max=1000000,
            step=10000,
            value=100000,
            marks={i: '{}'.format(i) for i in list(range(0, 1000000, 100000))}
        ),

        html.H3("Gamma values: recovery rate"),
        dcc.Slider(
            id='gammar',
            min=0,
            max=1,
            step=0.1,
            value=0.1,
            marks={i: '{}'.format(i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        ), html.H3("Beta min-max :infection spread dynamics")
        ,
        dcc.RangeSlider(
            id='betas',
            min=0,
            max=1,
            step=0.001,
            value=[0.1, 0.4],
            marks={i: '{}'.format(i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        )
        ,
        html.H3("Countries")
        ,
        dcc.Dropdown(
            id='country_drop_down2',
            options=[{'label': each, 'value': each} for each in df_input_large['country'].unique()],
            value='Germany',  # which are pre-selected
            multi=False
        )
    ])

    app.layout = html.Div(
        [html.Center(html.H1('Covid19 Data Analysis')), dcc.Tabs(id='my_tabs', value='tab_1', children=[tab_1, tab_2]),
         html.Div(html.Center([dcc.Graph(figure=fig, id='main_window_slope')]))])

    @app.callback(
        Output('main_window_slope', 'figure'),
        [Input('my_tabs', 'value'),
         Input('gammar', 'value'),
         Input('betas', 'value'),
         Input('N0s', 'value'),
         Input('country_drop_down', 'value'),
         Input('country_drop_down2', 'value'),
         Input('doubling_time', 'value')])
    def update_figure(tab, gammas, betas, N0_val, country_list, sir_country_list, show_doubling):

        ind = df_analyse[df_analyse[sir_country_list] == 0].index.get_loc(
            df_analyse[df_analyse[sir_country_list] == 0].index.max()) + 1
        ydata = df_analyse[sir_country_list][ind + 1:]
        t = np.arange(len(ydata))
        N0 = N0_val  # max susceptible population
        I0 = ydata[0]
        S0 = N0 - I0
        R0 = 0
        gamma = gammas

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

        if tab == 'tab_1':

            if 'doubling_rate' in show_doubling:
                my_yaxis = {'type': "log",
                            'title': 'Approximated doubling rate over 3 days (larger numbers are better)'}
            else:
                my_yaxis = {'type': "log",
                            'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                            }

            traces = []
            for each in country_list:

                df_plot = df_input_large[df_input_large['country'] == each]

                if show_doubling == 'confirmed_filtered_doubling_rate':
                    df_plot = df_plot[
                        ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_doubling_rate',
                         'confirmed_filtered_doubling_rate',
                         'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
                else:
                    df_plot = df_plot[
                        ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_doubling_rate',
                         'confirmed_filtered_doubling_rate',
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

            ind = df_analyse[df_analyse[sir_country_list] == 0].index.get_loc(
                df_analyse[df_analyse[sir_country_list] == 0].index.max()) + 1

            ydata = df_analyse[sir_country_list][ind:]
            t = np.arange(len(ydata))

            I0 = ydata[0]
            S0 = N0 - I0
            R0 = 0

            t_initial = 28
            t_intro_measures = 14
            t_hold = 21
            t_relax = 21

            beta_max = betas[1]
            beta_min = betas[0]

            popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
            fitted = fit_odeint(t, *popt)

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
            fig = make_subplots(rows=2, cols=2,
                                specs=[[{"colspan": 2}, None], [{"colspan": 2}, None]], subplot_titles=(
                                "Fit of SIR model with fixed beta and gamma",
                                'Szenario SIR simulations with fixed gamma and dynamic beta')
                                )
            trace11 = go.Scatter(x=t, y=ydata, mode='markers', name='True infected number')
            trace22 = go.Scatter(x=t, y=fitted, mode='lines', name='fitted infected number')
            trace111 = go.Scatter(x=propagation_rates.index, y=propagation_rates.infected, name='simlated infected',
                                  mode='lines',
                                  line=dict(width=5))
            trace222 = go.Bar(x=np.arange(len(ydata)), y=ydata, name='current infected')

            fig.add_trace(trace11, row=1, col=1)
            fig.add_trace(trace22, row=1, col=1)
            fig.add_trace(trace111, row=2, col=1)
            fig.add_trace(trace222, row=2, col=1)

            fig.update_yaxes(type='log', row=1, col=1, title_text='population infected')
            fig.update_yaxes(type='log', row=2, col=1, title_text='population infected')

            fig.update_xaxes(row=1, col=1, title_text='time in days')
            fig.update_xaxes(row=2, col=1, title_text='time in days')

            fig.update_layout(plot_bgcolor=colors['background'],
                              paper_bgcolor=colors['background'],
                              font={'color': colors['text']})

            return fig
    return app

def main():
    app = app_func()
    app.run_server(debug=True, use_reloader=False)
