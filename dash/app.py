# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State


## Functions

def generate_bar_plot(df, X, Y, color, hover, xlabel, ylabel):
    fig = px.bar(df, x=X, y=Y, hover_data=hover,color=color,labels={X:xlabel, Y:ylabel},color_continuous_scale="earth")
    fig.update_layout(plot_bgcolor='#ffffff')
    return fig

## Data

# ~ #Shows number of fics that are liked a certain number of times
df = pd.read_csv("info_fics.csv",sep=",")
dfU = pd.read_csv("info_user.csv",sep=",")
options = ["numHits","numKudos","count","numBookmarks","numComments"]

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
    color="dark",
    dark=True,
    sticky="top",
)

# Fic basic information

FIC_BASICS = [
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
]

USER_BASICS = [
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
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(FIC_BASICS)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(USER_BASICS)),], style={"marginTop": 30})
    ],
    className="mt-12",
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
    return fig

@app.callback(
    Output('summary_fic', 'figure'),
    [Input('figure_fics', 'clickData')])
def update_summary_graph(clickData):
    print(clickData["points"][0]["customdata"][0])
    user = clickData['points'][0]["customdata"][0]
    df_user = pd.melt(df[df["idName"] == user])
    df_user = df_user[df_user["variable"].isin(["count","numHits","numKudos","numBookmarks","numComments"])]
    fig = px.bar(df_user, x="variable", y="value", color="variable",labels={"variable":"Summary", "value":"Number"},color_continuous_scale="earth")
    fig.update_layout(showlegend=False)
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
