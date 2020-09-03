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
import argparse


## Functions

def generate_bar_plot(df, X, Y, color, hover, xlabel, ylabel):
    """ Generates a barplot using plotly.
    Input:
    df -> dataframe containing the data
    X -> name of the column that will go to the X-axis
    Y -> name of the column that will go to the Y-axis
    color -> Name of the column that will go to the color
    hover -> Names of the columns to be included in the hover tags
    xlabel -> Name of the x-axis
    ylabel -> Name of the y-axis
    
    Output: returns the figure object """
    
    fig = px.bar(df, x=X, y=Y, hover_data=hover,color=color,labels={X:xlabel, Y:ylabel},color_continuous_scale="earth")
    fig.update_layout(plot_bgcolor='#ffffff')
    return fig

def generate_scatter_plot(df,Xname, Yname):
    """Generates a scatter plot for the exploratory data.
    Input:
    df -> dataframe containing the information
    X -> name of the column that will go to the X-axis
    Y -> name of the column that will go to the Y-axis
    
    Output: returns the figure object """
    
    if Xname and Yname:
        fig = px.scatter(df, x=Xname, y=Yname)
    else:
        df["factors"] = df["factors"].astype(str)
        fig = px.scatter(df, x="f1", y="map@k", color="factors", size = "alpha",symbol="iterations",color_discrete_sequence=px.colors.diverging.Earth )
        # ~ fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    fig.update_layout(plot_bgcolor='#ffffff')
    
    return fig

def generate_main_scatter_plot(df):
    """Generates a scatter plot for the comparison of the models.
    Input:
    df -> dataframe containing the information

    Output: returns the figure object """

    fig = px.scatter(df, "f1@k", y="map@k", hover_data=["Recom","Names"])
    fig.update_traces(textposition='top center')
    fig.update_layout(plot_bgcolor='#ffffff')
    
    return fig

def get_recommendation(userName,kind,df_met):
    """ Returns the recommendation for a selected kind of method and user/item
    Input
    userName: despite the name it can be either a user name, and item name or be empty
    kind: indicates wether the user to item or the item to item method has been selected
    df_met: the metadata dataframe
    
    Output:
    A dataframe containing the recommendations for the user / item
    """
    if not kind:
        df_final = get_popular_recommendation(df_met,userName)
    else:
        if kind == "user":
            df = dfR
            user = userName
        elif kind == "item":
            df = dfC
            if userName:
                try:
                    user = int(userName)
                except:
                    user = userName
            else:
                user = userName
        if user in df[0].to_list():
            df_user = df[df[0] == user]
            if type(user) == int:
                author = ""
            else:
                author = user
            print("1",author,"2",user)
            #Get recommendations for user
            items = [int(df_user[x].to_string(index=False).split(".")[0].replace(" ","")) for x in range(1,df_user.shape[1])]
            items = [x for x in items if df_met[df_met["idName"] == x]["author"].to_string(index=False)[1:] != author][:10]
            links = [html.A(html.P(str(x)),href="https://archiveofourown.org/works/"+str(x),target="_blank") for x in items]
            #Get the metadata belonging to those recommendations
            dfM = df_met[df_met["idName"].isin(items)].copy()
            #Add the order in which they were recommended
            dfM["order"] = dfM["idName"].apply(lambda x:items.index(x))
            #Reorder them
            dfM = dfM.sort_values("order").set_index("order").reset_index()
            #Add links
            dfM["links"] = links
            df_final = dfM[["links","author","title","date_update","numWords","numHits","numKudos"]]
            df_final.date_update = pd.DatetimeIndex(df_final.date_update).strftime("%Y-%m-%d")
        else:
            df_final = get_popular_recommendation(df_met,userName)
    return df_final

def get_popular_recommendation(df_met,userName):
    """ When for some reason a recommendation can not be given, then the 
    most popular items are recommended.
    
    Input:
    df_met -> metadata dataframe
    userName -> user name if given
    
    Output:
    A dataframe containing the recommendations for the user / item
    """
    
    df_popular = df_met.sort_values("numKudos",ascending=False)
    if userName:
        df_popular = df_popular[df_popular["author"] != userName]
    items = df_popular["idName"].to_list()[:10]
    links = [html.A(html.P(str(x)),href="https://archiveofourown.org/works/"+str(x),target="_blank") for x in items]
    dfM = df_popular[df_popular["idName"].isin(items)].copy()
    dfM["order"] = dfM["idName"].apply(lambda x:items.index(x))
    dfM = dfM.sort_values("order").set_index("order").reset_index()
    dfM["links"] = links
    df_final = dfM[["links","author","title","date_update","numWords","numHits","numKudos"]]
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
parser = argparse.ArgumentParser(description="dashboard")
parser.add_argument("-i",dest="metadataFile",action="store",required=True,\
    help="File containing the fics metadata")
parser.add_argument("-u",dest="infoU",action="store",default="info_user.csv",\
    help="File containing user info generated by format_files.py")
parser.add_argument("-t",dest="infoI",action="store",default="info_fics.csv",\
    help="File containing item info generated by format_files.py")
parser.add_argument("-rU",dest="recomU",action="store",default="recom.u2i.csv",\
    help="Formatted user to item recommendations")
parser.add_argument("-rI",dest="recomI",action="store",default="recom.i2i.csv",\
    help="Formatted item to item recommendations")
args = parser.parse_args()

# ~ #Shows number of fics that are liked a certain number of times
df = pd.read_csv(args.infoI,sep=",",na_values="-")
dfU = pd.read_csv(args.infoU,sep=",",na_values="-")
dfR = pd.read_csv(args.recomU,sep=",",header=None,na_values="-")
dfC = pd.read_csv(args.recomI,sep=",",header=None,na_values="-")
dfC = dfC.apply(pd.to_numeric)
df_met = pd.read_csv(args.metadataFile,sep="\t",na_values="-",\
    parse_dates=["published_date","date_update"],date_parser=parse_date,\
    usecols=["idName","author","title","published_date","date_update","numWords","numKudos","numHits"])
df_metrics = pd.read_csv("data/ALS_evaluation.txt",sep="\t")
df_metrics2 = pd.read_csv("data/implicit_evaluation.txt",sep="\t")
df_main_metrics = pd.read_csv("data/summary.txt",sep="\t")

# Options for dropdown menus
options = ["numHits","numKudos","count","numBookmarks","numComments"]
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

# Fic basic information - Makes the fanfiction information tab

FIC_BASICS = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Fic Basic Information")),
            dbc.CardBody(
                [   
                    html.P(
                            "This graph shows the number of fics that are in training dataset. Different measures can be found, the counts represent the number of times a fic has"
                            " been liked in our dataset whereas all the other metrics were taken from the metadata data. Click on a particular fic to see the number of likes (Kudos) "
                            " the number of hits, number of comments and number of bookmarks",
                            className="card-text",
                    ),
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

# User basic information - Makes the User information tab

USER_BASICS = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("User Basic Information")),
            dbc.CardBody(
                [
                    html.P(
                            "This graph shows for each user in the training dataset the number of fics they have read.",
                            className="card-text",
                    ),
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

# Makes the first tab containing an explanation and the graph comparing the different models

MAIN_METRICS = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Comparison of different models")),
            dbc.CardBody(
                [
                    dcc.Loading(
                        id="loading-metric-information",
                        children=[
                            dbc.Alert(
                                "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                                id="no-data-alert-users",
                                color="warning",
                                style={"display": "none"},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [ 
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H4("AO3 recommender system", className="card-title"),
                                                        html.P(
                                                            "This dashboard shows the results of the AO3 recommender system. In the upper part you can obtain recommendations "
                                                            "for users of the archive of our own system and for fics related to the Marvel universe. This graph summarized the "
                                                            "f1@k versus the map@k of the different recommenders applied to the user to item data. The best model, situated on the"
                                                            " upper right part of the graph belongs to a model based on matrix factorization using ALS. The recommendations shown"
                                                            " in this dashboard belong to this model. "
                                                            " In the next graphs you can explore the number of times a fic has been read and how many likes and other popularity"
                                                            " metrics it has by clicking on the graph. You can check how many fics a user has read and finally you can see the results of"
                                                            " the results obtained by each run of matrix factorization when searching for the best hyperparameters.",
                                                        className="card-text",
                                                        ),
                                                    ]
                                                ),
                                            )
                                        ],md=6
                                    ),
                                    dbc.Col(dcc.Graph(id="figure_main_metrics", figure=generate_main_scatter_plot(df_main_metrics)))
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

#Makes the upper part of the recommender where the kind of recommender can be chosen and the name of the user / item can be put

RECOM_CHOICES = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Recommender")),
            dbc.CardBody(
                [   html.P(
                            "To view the recommendations choose either user to item or item to item depending on your input. "
                            "Then write down either a user name for the user to item search or a fic id for the item to item search and press submit.",
                            className="card-text",
                    ),
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
                                                options=[{"label": "User to item (default)", "value": "user"},{"label": "Item to item", "value": "item"}],
                                                value="user",
                                            ),
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

#Prints the recommendation table

RECOM_TABLE = dbc.Table.from_dataframe(get_recommendation("dls","user",df_met), id="recom_table", striped=True, bordered=True, responsive=True)

#Prints the graph showing the different results for the ALS exploration of hyperparameters
METRICS_ALS = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Metrics exploration")),
            dbc.CardBody(
                [
                    html.P(
                            "This graph shows a summary of the results obtained when doing the initial hyperparameter exploration for a matrix factorization based on ALS recommender. "
                            "The metrics shown by default are f1@k versus map@k. The color is related to the number of factors used in the model, the size represents the alpha and the "
                            "different shapes the number of iterations. Selecting a metric and then one hyperparameter will show a more detailed graph of how a given hyperparameter has "
                            "affected the metric.",
                            className="card-text",
                    ),
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

#Prints the graph showing the different results for the implicit exploration of hyperparameters
METRICS_IMPL = [
    dbc.Card(
        [
            dbc.CardHeader(html.H5("Metrics exploration")),
            dbc.CardBody(
                [
                    html.P(
                            "This graph shows a summary of the results obtained when doing the initial hyperparameter exploration for the three models implemented in implicit. "
                            "The metrics shown by default are f1@k versus map@k. The color is related to the number of factors used in the model, the size represents the alpha and the "
                            "different shapes the number of iterations. Selecting a model will allow you to focus on the results of that model.  Selecting a metric and then one hyperparameter "
                            " will show a more detailed graph of how a given hyperparameter has affected the metric.",
                            className="card-text",
                    ),
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
                                            dbc.Label("Choose model:"),
                                            dcc.Dropdown(
                                                id="model_option",
                                                options=[{'label': i, 'value': i} for i in model_options],
                                            )
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose metric:"),
                                            dcc.Dropdown(
                                                id="metric_option2",
                                                options=[{'label': i, 'value': i} for i in metric_options],
                                            )
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Choose hyperparameter:"),
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

#Creates the tab bar 
TABS = html.Div(
    dbc.Card(
        [
            dbc.Tabs(
                [
                    dbc.Tab(label="Method comparison",tab_id="tab_compare"),
                    dbc.Tab(label="Fanfiction information", tab_id="tab-fic"),
                    dbc.Tab(label="Users information", tab_id="tab-user"),
                    dbc.Tab(label="Initial metrics ALS", tab_id="tab-metrics"),
                    dbc.Tab(label="Metrics Implicit", tab_id="tab-metrics2"),
                ],
                id="tabs",
                active_tab="tab_compare",
            ),
            html.Div(id="content"),
        ],
        color="info",
        # ~ inverse=True
    )
)


#Joins all options into the body
BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(RECOM_CHOICES)),], style={"marginTop": 50}),
        dbc.Row([dbc.Col(dbc.Card(RECOM_TABLE)),], style={"marginTop": 50}),
        dbc.Row([dbc.Col(dbc.Card(TABS)),], style={"marginTop": 50}),
    ],
    className="mt-12",
)

#Foot bar
FOOTER = dbc.Navbar(
    children=[
        html.A(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("Data science master Kschool, 2020", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
        )
    ],
    color="#edb879",
    # ~ dark=True,
    sticky="bottom",
)

#Main app layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div(children=[NAVBAR,BODY,FOOTER])

##CALLBACKS - they give the functionality

#Creates the fic information figure
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

# When clicking on a fic allows the data of the graph on the right to change 
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

#Generates the metrics for the initial ALS exploration
@app.callback(
    Output('figure_metrics', "figure"),
    [Input("metric_option","value"),Input("hyper_option","value")]
)
def update_graph(metric,hyperparameter):
    fig = generate_scatter_plot(df_metrics,hyperparameter,metric)
    return fig

#Generates the metrics for the implicit exploration
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
    
#Generates the recommendation table
@app.callback(
    Output('recom_table', 'children'),
    [Input('input_recom_submit', 'n_clicks'),
    Input("backdrop-selector", "value")],
    [State('input_recom', 'value')])
def update_table(n_clicks,kind,value):
    print("A",n_clicks,kind,value)
    if n_clicks is None:
        if kind == "user":
            df = get_recommendation(value,kind,df_met)
        elif kind == "item":
            df = get_recommendation(value,kind,df_met)
        else:
            df = get_popular_recommendation(df_met,value)
    else:
        df = get_recommendation(value,kind,df_met)
    table = dbc.Table.from_dataframe(df, striped=True, bordered=True)
    return table

#Gives tabs their functionality
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
    elif at == "tab_compare":
        return MAIN_METRICS
    return html.P("This shouldn't ever be displayed...")


if __name__ == '__main__':
    app.run_server(debug=True)
