#!/usr/bin/env python

"""
  AO3_recommender - a recommendation system for fanfiction
  Copyright (C) 2020 - Marina Marcet-Houben
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State


## Functions

def generate_bar_plot(df, X, Y, color, hover, xlabel, ylabel):
    fig = px.bar(df, x=X, y=Y, hover_data=hover,color=color,labels={X:xlabel, Y:ylabel},color_continuous_scale="earth")
    fig.update_layout(plot_bgcolor='#ffffff')
    return fig

def generate_scatter_plot(df,Xname, Yname):
    if Xname and Yname:
        fig = px.scatter(df, x=Xname, y=Yname)
    else:
        df["factors"] = df["factors"].astype(str)
        fig = px.scatter(df, x="f1", y="map@k", color="factors", size = "alpha",symbol="iterations",color_discrete_sequence=px.colors.diverging.Earth )
        # ~ fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    fig.update_layout(plot_bgcolor='#ffffff')
    
    return fig

def get_recommendation(userName,kind,df_met):
    print("B",userName,kind)
    if not kind:
        df_final = get_popular_recommendation(df_met,userName)
    else:
        if kind == "ALS":
            df = dfR
            user = userName
        elif kind == "Content_based":
            df = dfC
            if userName:
                user = int(userName)
            else:
                user = userName
        if user in df[0].to_list():
            df_user = df[df[0] == user]
            #Get recommendations for user
            items = [int(df_user[x].to_string(index=False).split(".")[0].replace(" ","")) for x in range(1,11)]
            links = [html.A(html.P(str(x)),href="https://archiveofourown.org/works/"+str(x),target="_blank") for x in items]
            #Get the metadata belonging to those recommendations
            dfM = df_met[df_met["idName"].isin(items)].copy()
            #Add the order in which they were recommended
            dfM["order"] = dfM["idName"].apply(lambda x:items.index(x))
            #Reorder them
            dfM = dfM.sort_values("order").set_index("order").reset_index()
            #Add links
            dfM["links"] = links
            df_final = dfM[["links","author","title","date_update","numWords","numHits"]]
            df_final.date_update = pd.DatetimeIndex(df_final.date_update).strftime("%Y-%m-%d")
        else:
            df_final = get_popular_recommendation(df_met,userName)
    return df_final

def get_popular_recommendation(df_met,userName):
    df_popular = df_met.sort_values("numKudos",ascending=False)
    if userName:
        df_popular = df_popular[df_popular["author"] != userName]
    items = df_popular["idName"].to_list()[:10]
    links = [html.A(html.P(str(x)),href="https://archiveofourown.org/works/"+str(x),target="_blank") for x in items]
    dfM = df_popular[df_popular["idName"].isin(items)].copy()
    dfM["order"] = dfM["idName"].apply(lambda x:items.index(x))
    dfM = dfM.sort_values("order").set_index("order").reset_index()
    dfM["links"] = links
    df_final = dfM[["links","author","title","date_update","numWords","numHits"]]
    df_final.date_update = pd.DatetimeIndex(df_final.date_update).strftime("%Y-%m-%d")
    return df_final
    
def parse_date(x): 
    """ Parses a date to a datetime format.
    Input: string containing the date
    Output: datetime object
    """
    if "-" in x: 
        f = lambda x: pd.datetime.strptime(x, "%Y-%m-%d") 
    else: 
        f = lambda x: pd.datetime.strptime(x, "%d %b %Y") 
    return f(x) 

## Data

# ~ #Shows number of fics that are liked a certain number of times
df = pd.read_csv("info_fics.csv",sep=",",na_values="-")
dfU = pd.read_csv("info_user.csv",sep=",",na_values="-")
dfR = pd.read_csv("recom.u2i.csv",sep=",",header=None,na_values="-")
dfC = pd.read_csv("recom.i2i.csv",sep=",",header=None,na_values="-")
dfC = dfC.apply(pd.to_numeric)
df_met = pd.read_csv("../data/metadata_fics.cleaned.txt",sep="\t",na_values="-",parse_dates=["published_date","date_update"],date_parser=parse_date,usecols=["idName","author","title","published_date","date_update","numWords","numKudos"])
df_metrics = pd.read_csv("ALS_evaluation.txt",sep="\t")
df_metrics2 = pd.read_csv("implicit_evaluation.txt",sep="\t")
options = ["numHits","numKudos","count","numBookmarks","numComments"]
recommender_options = ["ALS","Content","Similarity"]
metric_options = ["precision","recall","f1","map@k"]
hyperparameter_options = ["alpha","regularization","factors","iterations"]
model_options = ["ALS","lmf","bpr"]


## Page layout elements

#Navigator bar

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("AO3 Marvel recommender", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://archiveofourown.org/tags/Marvel%20Cinematic%20Universe/works",
        )
    ],
    color="#edb879",
    # ~ dark=True,
    sticky="top",
)

# Fic basic information

FIC_BASICS = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Fic Basic Information")),
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading-fics-information",
                        children=[
                            dbc.Alert(
                                "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                                id="no-data-alert-fics",
                                color="warning",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(html.P(["Choose the popularity metric you want to see:"]), md=4),
                                    dbc.Col(
                                        [
                                            dcc.Dropdown(
                                                id="figure_fic_option",
                                                options=[{'label': i, 'value': i} for i in options],
                                                value='count',
                                            )
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(html.P(["Main popularity characteristics for selected fic:"]),md=4),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="figure_fics", clickData={'points': [{'customdata': [1867683]}]}), md=8),
                                    dbc.Col(dcc.Graph(id='summary_fic'), md=4),
                                ],
                                no_gutters=True,
                            ),
                        ],
                        type="default",
                    )
                ],
                style={"marginTop": 0, "marginBottom": 0},
            ),
        ],
        # ~ color="#dfd7c6",
    )
]

USER_BASICS = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("User Basic Information")),
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading-user-information",
                        children=[
                            dbc.Alert(
                                "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                                id="no-data-alert-users",
                                color="warning",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="figure_users", figure=generate_bar_plot(dfU,"index","count","count",["user"],"Users","Num. Fics read")))
                                ],
                                no_gutters=True,
                            ),
                        ]
                    )
                ]
            )
        ],
        # ~ color="#dfd7c6",
    )
] 

RECOM_CHOICES = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Recommender")),
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading-recommender-ALS",
                        children=[
                            dbc.Alert(
                                "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                                id="no-data-alert-recomALS",
                                color="warning",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [   
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose the kind of recommender"),
                                            dbc.RadioItems(
                                                id="backdrop-selector",
                                                options=[{"label": "User to item (default)", "value": True},{"label": "Item to item", "value": False}],
                                                value=True,
                                            ),
                                            dcc.Dropdown(id="kind_recommender"),
                                        ],
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Input user name"),
                                            dbc.Input(id="input_recom", placeholder="Input user/item name", type="text"),
                                            dbc.Button("Submit", id="input_recom_submit",color="primary"),
                                        ]
                                    ),
                                ],
                            ),
                        ]
                    )
                ]
            )
        ],
        color="info",
    )
]


RECOM_TABLE = dbc.Table.from_dataframe(get_recommendation("dls","ALS",df_met), id="recom_table", striped=True, bordered=True, responsive=True)

METRICS_ALS = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Metrics exploration")),
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading-metrics-information",
                        children=[
                            dbc.Alert(
                                "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                                id="no-data-alert-metrics",
                                color="warning",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose which metric you want to see:"),
                                            dcc.Dropdown(
                                                id="metric_option",
                                                options=[{'label': i, 'value': i} for i in metric_options],
                                            )
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose which hyperparameter you want to see:"),
                                            dcc.Dropdown(
                                                id="hyper_option",
                                                options=[{'label': i, 'value': i} for i in hyperparameter_options],
                                            )
                                        ],
                                        md=6,
                                    )
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="figure_metrics")),
                                ],
                                no_gutters=True,
                            ),
                        ],
                        type="default",
                    )
                ],
                style={"marginTop": 0, "marginBottom": 0},
            ),
        ],
        # ~ color = "#dfd7c6"
    )
]

METRICS_IMPL = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Metrics exploration")),
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading-metrics2-information",
                        children=[
                            dbc.Alert(
                                "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                                id="no-data-alert-metrics2",
                                color="warning",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose which metric you want to see:"),
                                            dcc.Dropdown(
                                                id="metric_option2",
                                                options=[{'label': i, 'value': i} for i in metric_options],
                                            )
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose which model you want to see:"),
                                            dcc.Dropdown(
                                                id="model_option",
                                                options=[{'label': i, 'value': i} for i in model_options],
                                            )
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose which hyperparameter you want to see:"),
                                            dcc.Dropdown(
                                                id="hyper_option2",
                                                options=[{'label': i, 'value': i} for i in hyperparameter_options],
                                            )
                                        ],
                                        md=4,
                                    )
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dcc.Graph(id="figure_metrics2")),
                                ],
                                no_gutters=True,
                            ),
                        ],
                        type="default",
                    )
                ],
                style={"marginTop": 0, "marginBottom": 0},
            ),
        ],
        # ~ color = "#dfd7c6"
    )
]


TABS = html.Div(
    dbc.Card(
        [
            dbc.Tabs(
                [
                    dbc.Tab(label="Fanfiction information", tab_id="tab-fic"),
                    dbc.Tab(label="Users information", tab_id="tab-user"),
                    dbc.Tab(label="Initial metrics ALS", tab_id="tab-metrics"),
                    dbc.Tab(label="Metrics Implicit", tab_id="tab-metrics2"),
                ],
                id="tabs",
                active_tab="tab-fic",
            ),
            html.Div(id="content"),
        ],
        color="info",
        # ~ inverse=True
    )
)



BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(RECOM_CHOICES)),], style={"marginTop": 50}),
        dbc.Row([dbc.Col(dbc.Card(RECOM_TABLE)),], style={"marginTop": 50}),
        dbc.Row([dbc.Col(dbc.Card(TABS)),], style={"marginTop": 50}),
    ],
    className="mt-12",
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div(children=[NAVBAR,BODY])


@app.callback(
    Output('figure_fics', 'figure'),
    [Input('figure_fic_option', 'value')])
def update_graph(kind):
    tags = {"count":"Traced likes","numKudos":"Total likes", "numHits":"Hits", "numBookmarks":"Bookmarks","numComments":"Comments"}
    ylabel = tags[kind]
    fig = generate_bar_plot(df, "index", kind, kind, ["idName","title","author"],"N. fics", ylabel)
    fig.update_layout(plot_bgcolor='#ffffff',showlegend=False)
    return fig
def create_summary_graph():
    df_user = pd.melt(df[df["idName"] == 1867683])
    df_user = df_user[df_user["variable"].isin(["count","numHits","numKudos","numBookmarks","numComments"])]
    fig = px.bar(df_user, x="variable", y="value", color="variable",labels={"variable":"Summary", "value":"Number"},color_continuous_scale="earth")
    fig.update_layout(showlegend=False)
    maxHits = df_met["numHits"].max()
    fig.update_yaxes(range=[0, maxHits])
    return fig

@app.callback(
    Output('summary_fic', 'figure'),
    [Input('figure_fics', 'clickData')])
def update_summary_graph(clickData):
    user = clickData['points'][0]["customdata"][0]
    df_user = pd.melt(df[df["idName"] == user])
    df_user = df_user[df_user["variable"].isin(["count","numHits","numKudos","numBookmarks","numComments"])]
    fig = px.bar(df_user, x="variable", y="value", color="variable",labels={"variable":"Summary", "value":"Number"},color_continuous_scale="earth")
    fig.update_layout(showlegend=False)
    maxHits = df_met["numHits"].max()
    fig.update_yaxes(range=[0, maxHits])
    return fig

@app.callback(
    Output('figure_metrics', "figure"),
    [Input("metric_option","value"),Input("hyper_option","value")]
)
def update_graph(metric,hyperparameter):
    fig = generate_scatter_plot(df_metrics,hyperparameter,metric)
    return fig

@app.callback(
    Output('figure_metrics2', "figure"),
    [Input("metric_option2","value"),Input("hyper_option2","value"),
    Input("model_option","value")]
)
def update_graph(metric,hyperparameter,model):
    if model:
        df1 = df_metrics2[df_metrics2["model"] == model]
    else:
        df1 = df_metrics2
    fig = generate_scatter_plot(df1,hyperparameter,metric)
    return fig
    

@app.callback(
    Output('recom_table', 'children'),
    [Input('input_recom_submit', 'n_clicks'),
    Input('kind_recommender', 'value')],
    [State('input_recom', 'value')])
def update_table(n_clicks,kind,value):
    print("A",n_clicks,kind,value)
    if n_clicks is None:
        if kind == "ALS":
            df = get_recommendation(value,kind,df_met)
        elif kind == "Content_based":
            df = get_recommendation(value,kind,df_met)
        else:
            df = get_popular_recommendation(df_met,value)
    else:
        df = get_recommendation(value,kind,df_met)
    table = dbc.Table.from_dataframe(df, striped=True, bordered=True)
    return table

@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-fic":
        return FIC_BASICS
    elif at == "tab-user":
        return USER_BASICS
    elif at == "tab-metrics":
        return METRICS_ALS
    elif at == "tab-metrics2":
        return METRICS_IMPL
    return html.P("This shouldn't ever be displayed...")

@app.callback(Output("kind_recommender", "options"), [Input("backdrop-selector", "value")])
def get_dowpdown(value):
    if value:
        recommender_options = ["ALS","Content_based","Similarity"]
    else:
        recommender_options = ["Content_based"]
    options=[{'label': i, 'value': i} for i in recommender_options]
    return options

if __name__ == '__main__':
    app.run_server(debug=True)
