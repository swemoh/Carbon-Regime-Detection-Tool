import pandas as pd
import datetime
from tkinter.ttk import Style
# from pydoc import classname
# from msilib.schema import tables
# from optparse import Option
# from tkinter.ttk import Style
from scipy.cluster import hierarchy
import matplotlib

matplotlib.use('Agg') ## Adding to avoid assertion failed error. It would shut down the server.
import matplotlib.pyplot as plt
import colorcet as cc
import json
import time

# from lenspy import DynamicPlot


# Interactive UI
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, dash_table, State, ctx, dash_table  # pip install dash (version 2.0.0 or higher)
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html

import warnings
warnings.simplefilter('ignore')

# local functions
# import backend
from backend import get_ocean_data, build_grids, fit_multivariate_lin_regression, plot_slope_maps, run_hc #plot_slope_maps_plotly
from backend import plot_bic_score_cluster_no, detect_clusters_on_dendrogram, get_cluster_map
from backend import analyse_clusters_, get_random_forest_graphs, get_cluster_summary_details

# for static images
import io
import base64
# -------------------------------------- Dash App ----------------------------------------

app = Dash(__name__) # always pass __name__ -> It is connected with the asset folder.
print(f"---> Loading Dash App: {app}")

## Global Variables
year = None
month = None
year_2 = None
month_2 = None

drivers = None
target = None
drivers_2 = None
target_2 = None
data_df = None
data_df_2 = None
grids_df_lst = None
grids_df_lst_2 = None
reg_df_mvlr = None
reg_df_mvlr_2 = None

linkage_matrix = None
linkage_matrix_2 = None
norm_hc_df = None
norm_hc_df_2 = None

dendro_dict_lst = None
dendro_dict_lst_2 = None

del_var = None
del_var_2 = None
del_dist = None
del_dist_2 = None

hc_df = None #has cluster labels
hc_df_2 = None #has cluster labels

final_data = None
final_data_2 = None
cluster_colors = None
cluster_colors_2 = None


# Z_matrix = None
# dendrogram_plot = None
# grids_df_lst = None
# final_data = None
# grid_index_list = None
# dendrogram_plot = None


textvalue = {"Cluster_labels": {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0, "4": 5.0, "5": 6.0, "6": 7.0, "7": 8.0, "8": 9.0}, "Mean slope_SST": {"0": 11.617813892582348, "1": 411.2175598144531, "2": 79.00297546386719, "3": -12.861379809478816, "4": 18.110509959747606, "5": 27.648557662963867, "6": -30.071836471557617, "7": -85.80758562781936, "8": -228.46382796596473}, "Median slope_SST": {"0": 11.507404327392578, "1": 411.2175598144531, "2": 79.00297546386719, "3": -13.754770278930664, "4": 13.86928939819336, "5": 27.648557662963867, "6": -30.071836471557617, "7": -81.34871673583984, "8": -210.37831115722656}, "Std. Dev slope_SST": {"0": 4.964051403313165, "1": 0.0, "2": 0.0, "3": 23.785669956823245, "4": 19.654569090447925, "5": 0.0, "6": 0.0, "7": 28.640812695090776, "8": 79.17782721174007}, "Mean slope_DICP": {"0": 1.2953531831910219, "1": -24.67351722717285, "2": -14.651311874389648, "3": 2.299451827990995, "4": 1.8320281881492284, "5": 18.034093856811523, "6": 19.325162887573242, "7": 2.187595133282641, "8": 2.226955064725428}, "Median slope_DICP": {"0": 1.2720415592193604, "1": -24.67351722717285, "2": -14.651311874389648, "3": 2.260685682296753, "4": 1.7730202674865723, "5": 18.034093856811523, "6": 19.325162887573242, "7": 2.243786334991455, "8": 2.240457057952881}, "Std. Dev slope_DICP": {"0": 0.20948700977987195, "1": 0.0, "2": 0.0, "3": 0.6535840062718457, "4": 0.24247206498979432, "5": 0.0, "6": 0.0, "7": 0.48114367919936096, "8": 0.43048005765124575}, "Mean slope_ALK": {"0": -0.8988023009633921, "1": 24.868146896362305, "2": 14.95151138305664, "3": -1.9916649964457307, "4": -1.6161329827933397, "5": -21.93532943725586, "6": -11.006959915161133, "7": -1.8668428154577301, "8": -1.9967885072662237}, "Median slope_ALK": {"0": -0.8810020685195923, "1": 24.868146896362305, "2": 14.95151138305664, "3": -1.9725804328918457, "4": -1.531843662261963, "5": -21.93532943725586, "6": -11.006959915161133, "7": -1.9820913076400757, "8": -2.0467121601104736}, "Std. Dev slope_ALK": {"0": 0.20724851019648136, "1": 0.0, "2": 0.0, "3": 0.7179773367223017, "4": 0.3484541262405446, "5": 0.0, "6": 0.0, "7": 0.581153568964698, "8": 0.455063630616868}, "Area(in km. sq)": {"0": 264555.5, "1": 264555.5, "2": 264555.5, "3": 264555.5, "4": 264555.5, "5": 264555.5, "6": 264555.5, "7": 264555.5, "8": 264555.5}}






global COLOMN_LIST, CLUSTER_DETAILS
COLOMN_LIST = ['nav_lat', 'nav_lon', 'SST', 'DIC', 'ALK', 'fco2', 'cluster',
       'slope_sst', 'slope_dic', 'slope_alk']
CLUSTER_DETAILS = ['Cluster Label', 'Area (sq. Km)', 'Driver' ,'Mean Regression coeffcient', 'Median Regression coeffcient', 'Standard Deviation Regression coeffcient']
PAGE_SIZE = 20
LOGO_GEOMAR_PATH = 'assets/geomar-logo.png'
encoded_LOGO_GEOMAR_PATH = base64.b64encode(open(LOGO_GEOMAR_PATH, 'rb').read())

## Global CSS
green_button_style = {"margin": "15px",
                      "background-color": "#4CAF50", # /* Green */
                    #   "border": "none",
                      "color": "white",
                      "padding": "8px 12px",
                      "text-align": "center",
                    #   "text-decoration": "none",
                      "display": "inline-block",
                      "font-size": "15px",
                      "border-radius": "15px",
                      "width":"15%",
                      'align':'left',
                    #   ".hover": { "color": "pink", "background-color": "red"},
                      }

# Add the CSS stylesheet
# app.css.append_css({"external_url": "styles.css"})

# --------------------------------------- App layout ---------------------------------------

app.layout = html.Div([
    html.Div(className="row", children=[
        html.Img(src=dash.get_asset_url('geomar_logo_eng.jpg'), style={'width':'30%',}),  # for Dash version >= 2.2.0
        # html.Img(src=dash.get_asset_url('geomar-logo.png'), style={'width':'20%',}),  # for Dash version >= 2.2.0
        html.H1("A demonstration of Oceanic Carbon Regimes detection", style={'text-align': 'center', 'margin-top': '60px'}),
        html.Img(src=dash.get_asset_url('cau-logo.png'), style={'width':'30%',}),  # for Dash version >= 2.2.0
    ], style={'background-color': 'white', 'display':'flex'}),

    html.Br(),
    html.Hr(),
    # html.Br(),
    # html.P("Demo Introduction. A paragraph about the tool."),
    html.H2("Step 1: Select ocean model, required year and month. Step 2: Select required drivers of CO2.",),
    html.H2("Step 2: Press the button at the bottom to start grid-based multivariate linear regression between CO2 and its drivers."),

    html.Div(className="row", children=[
            dcc.Dropdown(id="input_select_ocean_model",
                options=[
                    {"label": "Select Ocean Model", "value": '00'},
                    {"label": "NEMO-ORCA05_1_month", "value": 'ORCA05'},
                    ],
                 multi=False,
                 value='00',
                 style={'width': "40%",}
                 ), # first DD ends
            dcc.Dropdown(id="input_select_year",
                options=[
                    {"label": str(year), "value": str(year)} for year in range(2009, 2019)
                    ],
                 multi=False,
                 value=2015,
                 style={'width': "28%",}
                 ), # first DD ends
            dcc.Dropdown(id="input_select_month",
                 options=[
                     {"label": month, "value": month[:3].lower()} for month in [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                    ],
                 multi=False,
                 value='jan',
                 style={'width': "30%"}
                 ), # second DD ends
                ],style=dict(display='flex')),
    
    ## Empty container
    # html.Div(id='empty_container_1', children=[]),
    ## Horizontal break
    html.Br(),
    html.Div(className="row", children=[
                html.Div(className='column', style={'margin-right': '20px'}, children=[
                html.H2('Select Drivers'),
                dcc.Checklist(
                id='checkboxes_driver',
                options=[
                    {'label': 'Sea Surface Temperature', 'value': 'SST'},
                    {'label': 'Dissolved Inorganic Carbon', 'value': 'DICP'},
                    {'label': 'Salinity', 'value': 'SAL'},
                    {'label': 'Alkalinity', 'value': 'ALK'},
                    {'label': 'Sea Ice Coverege', 'value': 'ICE'},
                ],
                # value=['sst'],  # Set the default selected values
                labelStyle={'display': 'block'},  # Display each label in a new line
                className='checkbox-label',  # Apply the custom CSS class for the checkboxes
                ), # end of checklist
                dcc.Slider(
                    id='slider',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.9,  # Set the initial value
                    marks={i/10: str(i/10) for i in range(11)},  # Add marks for each step
                ),
            ]),

            html.Div(className='column', children=[
                html.H2('Select Target'),
                dcc.Checklist(
                id='checkboxes_target',
                options=[
                    {'label': 'CO2 Natural', 'value': 'fco2_pre'},
                    {'label': 'CO2 Anthropogenic', 'value': 'fco2'},
                ],
                # value=['fco2_pre'],  # Set the default selected values
                labelStyle={'display': 'block'},  # Display the labels in a new line
                className='checkbox-label',  # Apply the custom CSS class for the checkboxes
                ),
            ]),
    ], style=dict(display='flex', width= '200 px')),

    html.Br(),
    html.Br(),
    html.H2("Add the next one."),

        html.Div(className="row", children=[
            dcc.Dropdown(id="input_select_ocean_model_2",
                options=[
                    {"label": "Select Ocean Model", "value": '00'},
                    {"label": "NEMO-ORCA05_1_month", "value": 'ORCA05'},
                    ],
                 multi=False,
                 value='00',
                 style={'width': "40%",}
                 ), # first DD ends
            dcc.Dropdown(id="input_select_year_2",
                options=[
                    {"label": str(year), "value": str(year)} for year in range(2009, 2019)
                    ],
                 multi=False,
                 value=2015,
                 style={'width': "28%",}
                 ), # first DD ends
            dcc.Dropdown(id="input_select_month_2",
                 options=[
                     {"label": month, "value": month[:3].lower()} for month in [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                    ],
                 multi=False,
                 value='jan',
                 style={'width': "30%"}
                 ), # second DD ends
                ],style=dict(display='flex')),
    
    ## Empty container
    # html.Div(id='empty_container_1', children=[]),

    ## Horizontal break
    html.Br(),
    html.Div(className="row", children=[
                html.Div(className='column', style={'margin-right': '20px'}, children=[
                html.H2('Select Drivers'),
                dcc.Checklist(
                id='checkboxes_driver_2',
                options=[
                    {'label': 'Sea Surface Temperature', 'value': 'SST'},
                    {'label': 'Dissolved Inorganic Carbon', 'value': 'DICP'},
                    {'label': 'Salinity', 'value': 'SAL'},
                    {'label': 'Alkalinity', 'value': 'ALK'},
                    {'label': 'Sea Ice Coverege', 'value': 'ICE'},
                ],
                # value=['sst'],  # Set the default selected values
                labelStyle={'display': 'block'},  # Display each label in a new line
                className='checkbox-label',  # Apply the custom CSS class for the checkboxes
                ), # end of checklist
                dcc.Slider(
                    id='slider_2',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.5,  # Set the initial value
                    marks={i/10: str(i/10) for i in range(11)},  # Add marks for each step
                ),
            ]),

            html.Div(className='column', children=[
                html.H2('Select Target'),
                dcc.Checklist(
                id='checkboxes_target_2',
                options=[
                    {'label': 'CO2 Natural', 'value': 'fco2_pre'},
                    {'label': 'CO2 Anthropogenic', 'value': 'fco2'},
                ],
                # value=['fco2_pre'],  # Set the default selected values
                labelStyle={'display': 'block'},  # Display the labels in a new line
                className='checkbox-label',  # Apply the custom CSS class for the checkboxes
                ),
            ]),
    ], style=dict(display='flex', width= '200 px')),

    html.Br(),

    html.Div(id='regression_button_container', children=[
        html.Button('Run analysis.', id='btn_regression', style= green_button_style),
    ], style=dict(display='flex')),

    html.Br(),

    html.Div(id='regression_output_container_1', children=[
        dcc.Loading(
            id="reg1_driver1_slopes_loading",
            type="default",
            # children=dcc.Graph(id='reg1_driver1_slopes_img', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            children=html.Img(id='reg1_driver1_slopes_img', src = '', style={'height':'450px', 'width':'60%','align':'center'}), 
            ),
        html.Br(),
        
        dcc.Loading(
            id="reg1_driver2_slopes_loading",
            type="default",
            # children=dcc.Graph(id='reg1_driver2_slopes_img', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            children=html.Img(id='reg1_driver2_slopes_img', src = '', style={'height':'450px', 'width':'60%','align':'center'}),
            ),
        html.Br(),
        
        dcc.Loading(
            id="reg1_driver3_slopes_loading",
            type="default",
            # children=dcc.Graph(id='reg1_driver3_slopes_img', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            children=html.Img(id='reg1_driver3_slopes_img', src = '', style={'height':'450px', 'width':'60%','align':'center'}),
            ),

    ], style={'text-align':'center'}),

    html.Br(),

    html.Div(id='regression_output_container_2', children=[
        dcc.Loading(
            id="reg2_driver1_slopes_loading",
            type="default",
            # children=dcc.Graph(id='reg2_driver1_slopes_img', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            children=html.Img(id='reg2_driver1_slopes_img', src = '', style={'height':'450px', 'width':'60%','align':'center'}),
            ),
        html.Br(),
        
        dcc.Loading(
            id="reg2_driver2_slopes_loading",
            type="default",
            # children=dcc.Graph(id='reg2_driver2_slopes_img', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            children=html.Img(id='reg2_driver2_slopes_img', src = '', style={'height':'450px', 'width':'60%','align':'center'}),
            ),
         html.Br(),
        
        dcc.Loading(
            id="reg2_driver3_slopes_loading",
            type="default",
            # children=dcc.Graph(id='reg2_driver3_slopes_img', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            children=html.Img(id='reg2_driver3_slopes_img', src = '', style={'height':'450px', 'width':'60%','align':'center'}),
            ),

    ], style={'text-align':'center'}),

    html.Br(),

    html.H2("Run the hierarchical clustering to group the grid boxes."),

    html.Div(id='clustering_button_container', children=[
        html.Button('Run hierarchical clustering.', id='btn_hc', style= green_button_style),
    ], style=dict(display='flex')),

    html.Div(id='clustering_output_container', children=[
        dcc.Loading(
            id="hc1_loading",
            type="default",
            children=html.Img(id='hc_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="hc2_loading",
            type="default",
            children=html.Img(id='hc_img_2', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
         html.Br(),
        
    ], style={'text-align':'center'}),

    html.Br(),

    html.H2("Get BIC scores and cluster numbers for different possible pairs of delta_var and delta_dist."),

    html.Div(id='bic_button_container', children=[
        html.Button('Plot graphs.', id='btn_bic', style= green_button_style),
    ], style=dict(display='flex')),

    html.Div(id='bic_graph_container', children=[
        dcc.Loading(
            id="bic1_loading",
            type="default",
            children=dcc.Graph(id='bic_graph', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            ),
        html.Br(),
        
        dcc.Loading(
            id="bic2_loading",
            type="default",
            children=dcc.Graph(id='bic_graph_2', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            ),
         html.Br(),
        
    ], style={'text-align':'center'}),
    html.H3("Note: Use these bubble graphs to make informed decision for the threshold selections in the next step.",
    style={'text-align':'center' , 'color':'gray'}),
    html.Br(),

    html.H2("Set the threshold parameters for change in distance (delta_dist) and change in variance (delta_var) for the first dendrogram.",),
    html.H2("Select desired values of thresholds from the list and then press the button at the right to begin the cluster selection procedure.",),
    html.Div(className="row", children=[
            dcc.Dropdown(id="input_select_delta_variance",
                options=[
                    {"label": "Select Delta Variance", "value": 0.0},
                    {"label": "0.1", "value": 0.1},
                    {"label": "0.2", "value": 0.2},
                    {"label": "0.3", "value": 0.3},
                    {"label": "0.4", "value": 0.4},
                    {"label": "0.5", "value": 0.5},
                    {"label": "0.6", "value": 0.6},
                    {"label": "0.7", "value": 0.7},
                    {"label": "0.8", "value": 0.8},
                    {"label": "0.9", "value": 0.9},
                    {"label": "1.0", "value": 1.0},
                    {"label": "1.2", "value": 1.2},
                    {"label": "1.5", "value": 1.5},
                    {"label": "1.7", "value": 1.7},
                    {"label": "2.0", "value": 2.0},],
                 multi=False,
                 value=0.0,
                 style={'width': "40%", "margin": "2px"}
                 ), # first DD ends
            dcc.Dropdown(id="input_select_delta_height",
                 options=[{"label": "Select Delta Distance", "value": 0.0},
                    {"label": "2.0", "value": 2.0},
                    {"label": "5.0", "value": 5.0},
                    {"label": "7.0", "value": 7.0},
                    {"label": "10.0", "value": 10.0},
                    {"label": "12.0", "value": 12.0},
                    {"label": "15.0", "value": 15.0},
                    {"label": "20.0", "value": 20.0},
                    {"label": "25.0", "value": 25.0},
                    {"label": "30.0", "value": 30.0},
                    {"label": "35.0", "value": 35.0},
                    {"label": "40.0", "value": 40.0},
                    {"label": "45.0", "value": 45.0},
                    {"label": "50.0", "value": 50.0},],
                 multi=False,
                 value=0.0,
                 style={'width': "40%",  "margin": "2px"}
                  # second DD ends, 
                 ), 
                html.Button('Get clusters.', id='btn_dendro', className="me-1", style= green_button_style),
                # html.Div([dbc.Button( "Primary", id="btn_dendro", color="primary", className="me-1")]),

                ], 

                style=dict(display='flex')),
    html.Br(),
    html.H2("or, input your own threshold values for change in variance and change in distance.",),
    html.Div(className="row", children=[
        dcc.Input(
            id="input_range_var", type="number", placeholder="add variance threshold e.g. 0.03",
            min=0.01, max=100.0, step=0.01, debounce=True, style={'margin':'10px', 'width':'20%'}
        ),
        dcc.Input(
            id="input_range_dist", type="number", placeholder="add distance threshold e.g. 0.6",
            min=0.1, max=100.0, step=0.1, debounce=True, style={'margin':'10px', 'width':'20%'}
        ),
    html.Button('Submit manual inputs.', id='btn_manual_dendro', style= green_button_style),],
        style=dict(display='flex')),
    
    html.Div(children=[
        html.H3("I. The red markers are the links identifed as clusters and are annotated with respective labels.",),
        html.H3("II. The dotted black lines highlight the distance on the dendrogram at which the clusters have been found.",),
        dcc.Loading(
            id="cluster-loading",
            type="circle",
            children=[html.Img(id='dendro_cut', src = '', style={'height':'800px', 'width':'80%','align':'center'}),]
        )], style={'text-align':'center'}),
    
    html.Br(),

    html.H2("Now repeat for the second dendrogram."),

    html.H2("Set the threshold parameters for change in distance (delta_dist) and change in variance (delta_var) for the first dendrogram.",),
    html.H2("Select desired values of thresholds from the list and then press the button at the right to begin the cluster selection procedure.",),
    html.Div(className="row", children=[
            dcc.Dropdown(id="input_select_delta_variance_2",
                options=[
                    {"label": "Select Delta Variance", "value": 0.0},
                    {"label": "0.1", "value": 0.1},
                    {"label": "0.2", "value": 0.2},
                    {"label": "0.3", "value": 0.3},
                    {"label": "0.4", "value": 0.4},
                    {"label": "0.5", "value": 0.5},
                    {"label": "0.6", "value": 0.6},
                    {"label": "0.7", "value": 0.7},
                    {"label": "0.8", "value": 0.8},
                    {"label": "0.9", "value": 0.9},
                    {"label": "1.0", "value": 1.0},
                    {"label": "1.2", "value": 1.2},
                    {"label": "1.5", "value": 1.5},
                    {"label": "1.7", "value": 1.7},
                    {"label": "2.0", "value": 2.0},],
                 multi=False,
                 value=0.0,
                 style={'width': "40%", "margin": "2px"}
                 ), # first DD ends
            dcc.Dropdown(id="input_select_delta_height_2",
                 options=[{"label": "Select Delta Distance", "value": 0.0},
                    {"label": "2.0", "value": 2.0},
                    {"label": "5.0", "value": 5.0},
                    {"label": "7.0", "value": 7.0},
                    {"label": "10.0", "value": 10.0},
                    {"label": "12.0", "value": 12.0},
                    {"label": "15.0", "value": 15.0},
                    {"label": "20.0", "value": 20.0},
                    {"label": "25.0", "value": 25.0},
                    {"label": "30.0", "value": 30.0},
                    {"label": "35.0", "value": 35.0},
                    {"label": "40.0", "value": 40.0},
                    {"label": "45.0", "value": 45.0},
                    {"label": "50.0", "value": 50.0},],
                 multi=False,
                 value=0.0,
                 style={'width': "40%",  "margin": "2px"}
                  # second DD ends, 
                 ), 
                html.Button('Get clusters.', id='btn_dendro_2', className="me-1", style= green_button_style),
                # html.Div([dbc.Button( "Primary", id="btn_dendro", color="primary", className="me-1")]),

                ], 

                style=dict(display='flex')),
    html.Br(),

    html.H2("or, input your own threshold values for change in variance and change in distance.",),
    html.Div(className="row", children=[
        dcc.Input(
            id="input_range_var_2", type="number", placeholder="add variance threshold e.g. 0.03",
            min=0.01, max=100.0, step=0.01, debounce=True, style={'margin':'10px', 'width':'20%'}
        ),
        dcc.Input(
            id="input_range_dist_2", type="number", placeholder="add distance threshold e.g. 0.6",
            min=0.1, max=100.0, step=0.1, debounce=True, style={'margin':'10px', 'width':'20%'}
        ),
    html.Button('Submit manual inputs.', id='btn_manual_dendro_2', style= green_button_style),],
        style=dict(display='flex')),

    html.Div(children=[
        html.H3("I. The red markers are the links identifed as clusters and are annotated with respective labels.",),
        html.H3("II. The dotted black lines highlight the distance on the dendrogram at which the clusters have been found.",),
        dcc.Loading(
            id="cluster-loading_2",
            type="circle",
            children=[html.Img(id='dendro_cut_2', src = '', style={'height':'800px', 'width':'80%','align':'center'}),]
        )], style={'text-align':'center'}),

    html.Br(),

    html.H2(" Step 3: Collect the clustering labels and study the output.",),
    html.Button("Extract clustering labels.", id="btn_get_cluster_maps", style= green_button_style),
    html.Br(),
    html.Div(children=[
         html.H3('Detected ocean carbon regimes are shown below on the map.'),
         dcc.Loading(id="loading-cluster-map",
                    type="circle",
                    children=[
                    html.Img(id='cluster_map_image', src = '', style={'height':'450px', 'width':'60%',}),
                    # dcc.Graph(id='cluster_map_graph', style={'height':'1200px', 'width':'100%','align':'center'}),
                    ]),
        dcc.Loading(id="loading-cluster-map_2",
                    type="circle",
                    children=[
                    html.Img(id='cluster_map_image_2', src = '', style={'height':'450px', 'width':'80%',}),
                    # dcc.Graph(id='cluster_map_graph', style={'height':'1200px', 'width':'100%','align':'center'}),
                    ]),

     ],  style={'text-align':'center'}),
    html.Br(),
    html.H2(" Step 4: Study the output.",),
    html.Button("Analyse Clusters.", id="btn_get_cluster_details", style= green_button_style),

    html.Div(children=[
                html.H3('A. Clustered multivariate linear regression coefficients distribution plot.'),
        dcc.Loading(id="loading-dist_image",
                    type="circle",
                    children=[
                    # html.Img(id='cluster_dist_image', src = '', style={'height':'700px', 'width':'80%',}),
                    html.H4("From 1st model:"),
                    html.Br(),

                    dcc.Graph(id='cluster_dist_image', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
                    dcc.Graph(id='cluster_dist_image_2', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
                    dcc.Graph(id='cluster_dist_image_3', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),

                    html.H4("From 2nd model:"),
                    html.Br(),

                    dcc.Graph(id='cluster_dist_image_4', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
                    dcc.Graph(id='cluster_dist_image_5', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
                    dcc.Graph(id='cluster_dist_image_6', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
                    ]),
    ]),
    html.Br(),
    html.Div(children=[
        html.H3('B. Explore Non-linear target-driver relationships in the detected carbon regimes.'),

        html.Div(className="row", children=[
        dcc.Input(
            id="input_decision_tree_nums", type="number", placeholder="Enter No. of Decision Trees",
            min=1, max=200, step=1, debounce=True, style={'margin':'10px', 'width':'20%'}
        ),
        html.Button("Run Random Forest.", id="btn_run_random_forest", style= green_button_style),
        ],
        style=dict(display='flex')),

        dcc.Loading(
            id = 'loading-random-forest',
            type='circle',
            children=[
                html.H4("From 1st model:"),
                dcc.Graph(id='rf_fig', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
                html.H4("From 2nd model:"),
                dcc.Graph(id='rf_fig_2', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            ],
        ),
    ]),
    html.Div(children=[
        html.H3('C. Clustering Overview.'),
        html.Button("Show", id="btn_show_cluster_summary", style= green_button_style),
        dcc.Loading(
            id="loading-datatable-summary-detail",
            type="circle",
            children=[
                    html.H4("From 1st model:"),
                    # Display the JSON data from the store
                    # dcc.Textarea(id='json-display',value = textvalue, readOnly=False, style={'width': '60%', 'height': 500}),
                    
                    # dcc.Textarea(id='json-display_2', readOnly=False, style={'width': '60%', 'height': 500}),

                    dash_table.DataTable(id='datatable-cluster-detail',
                    columns=[{'name': i, 'id': i} for i in CLUSTER_DETAILS],
                    page_current=0,
                    page_size=PAGE_SIZE,
                    page_action='custom',
                    style_table={'width':'50%','margin-left':'350px'},
                    style_header={'textAlign': 'center'},),

                    html.Br(),                 
                    html.H4("From 2nd model:"),

                    dash_table.DataTable(id='datatable-cluster-detail-2',
                    columns=[{'name': i, 'id': i} for i in CLUSTER_DETAILS],
                    page_current=0,
                    page_size=PAGE_SIZE,
                    page_action='custom',
                    style_table={'width':'50%','margin-left':'350px'},
                    style_header={'textAlign': 'center'},)
            ]),
        html.Br(),                 
    ]),
    html.H2(" Step 5: Download the output.",),
    # html.H4("Here is the clustering output in a tabular format. You can only see random 20 rows. To access the complete data with more than 200,000 datapoints, click on the download button below.",
    #                         style={'color':'blue'}),
    # dcc.Loading(id="loading-datatable-upload",type="circle",
    #                     children=[dash_table.DataTable(id='datatable-upload-container',
    #                         columns=[{'name': i, 'id': i} for i in COLOMN_LIST],
    #                         page_current=0,
    #                         page_size=PAGE_SIZE,
    #                         page_action='custom',
    #                         style_table={'width':'70%','margin-left':'200px'},
    #                         style_header={'textAlign': 'center'})]),

    html.Br(),
    html.Button("Download output.", id="btn_download", style= green_button_style),
    # html.Button("Download CSV", id="btn_csv", style= green_button_style),
    dcc.Download(id="download-dataframe-csv",),
    dcc.Download(id="download-dataframe-csv_2",),
    html.Br(),
    html.Br(),
])

# --------------------------- *************** -------------------------------
# ---------------------------- Callback functions ----------------------------
# --------------------------- *************** -------------------------------

#---------------------------- 1. Run regression. ----------------------------

## i/p - year, month, drivers and target
## o/p -  6 maps of the slopes in natural and anthropogenic scenario

@app.callback(
    [Output('reg1_driver1_slopes_img', 'src'),
    Output('reg1_driver2_slopes_img', 'src'),
    Output('reg1_driver3_slopes_img', 'src'),

    Output('reg2_driver1_slopes_img', 'src'),
    Output('reg2_driver2_slopes_img', 'src'),
    Output('reg2_driver3_slopes_img', 'src'),],

    [Input(component_id='input_select_year', component_property='value'),
    Input(component_id='input_select_month', component_property='value'),
    Input(component_id='checkboxes_driver', component_property='value'), 
    Input(component_id='slider', component_property='value'), 
    Input(component_id='checkboxes_target', component_property='value'), 

    Input(component_id='input_select_year_2', component_property='value'),
    Input(component_id='input_select_month_2', component_property='value'),
    Input(component_id='checkboxes_driver_2', component_property='value'), 
    Input(component_id='slider_2', component_property='value'), 
    Input(component_id='checkboxes_target_2', component_property='value'),  

    Input('btn_regression','n_clicks')])
# the order of parameter follows the order of input for callback.
def load_regression(input_select_year, input_select_month, checkboxes_driver, slider, checkboxes_target,
                   input_select_year_2, input_select_month_2, checkboxes_driver_2, slider_2, checkboxes_target_2,
                   btn_regression):
     if ctx.triggered_id == 'btn_regression':

        start_time = time.time()

        global drivers
        global target
        global drivers_2
        global target_2
        global data_df
        global data_df_2
        global grids_df_lst
        global grids_df_lst_2
        global reg_df_mvlr
        global reg_df_mvlr_2

        global year
        global month
        global year_2
        global month_2


        print("i/p:", input_select_year, input_select_month, checkboxes_driver, slider, checkboxes_target,
                   input_select_year_2, input_select_month_2, checkboxes_driver_2, slider_2, checkboxes_target_2)
        
        year = input_select_year
        year_2 = input_select_year_2
        month = input_select_month
        month_2 = input_select_month_2
        
        if checkboxes_target[0] == 'fco2_pre':
            is_natural = True
        else:
            is_natural = False
        
        if is_natural:
            drivers = checkboxes_driver
        else:
            drivers = [item.replace('P', '') for item in checkboxes_driver]
        target = checkboxes_target[0]
        
        print("---> Fetching Dataset 1.")
        data_df = get_ocean_data(input_select_year, input_select_month, is_natural, drivers, target, slider)
        print(data_df.head(10))
        

        if checkboxes_target_2[0] == 'fco2_pre':
            is_natural_2 = True
        else:
            is_natural_2 = False

        if is_natural_2:
            drivers_2 = checkboxes_driver_2
        else:
            drivers_2 = [item.replace('P', '') for item in checkboxes_driver_2]
        target_2 = checkboxes_target_2[0]
        

        print("---> Fetching Dataset 2.")
        data_df_2 = get_ocean_data(input_select_year_2, input_select_month_2, is_natural_2, drivers_2, target_2, slider_2)
        print(data_df_2.head(10))
        
        


        print("---> 2x2 gridding.")

        cell_width = 2
        grids_df_lst = build_grids(data_df,cell_width)
        grids_df_lst_2 = build_grids(data_df_2,cell_width)

        print("---> Running 1st MVLR.")
        reg_df_mvlr = fit_multivariate_lin_regression(grids_df_lst,drivers,target)
        
        print("---> Running 2nd MVLR.")
        reg_df_mvlr_2 = fit_multivariate_lin_regression(grids_df_lst_2,drivers_2,target_2)
        
        ## Remove grids with Zero or only one coordinate
        reg_df_mvlr = reg_df_mvlr[reg_df_mvlr.data_count != 0]
        reg_df_mvlr = reg_df_mvlr[reg_df_mvlr.data_count != 1]
        reg_df_mvlr_2 = reg_df_mvlr_2[reg_df_mvlr_2.data_count != 0]
        reg_df_mvlr_2 = reg_df_mvlr_2[reg_df_mvlr_2.data_count != 1]

        print(reg_df_mvlr.columns)
        print(len(reg_df_mvlr))

        print(reg_df_mvlr_2.columns)
        print(len(reg_df_mvlr_2))

        slopes_data = []
        for index, row in reg_df_mvlr.iterrows():
            _df_ = grids_df_lst[row['cell_id']]
            if row['reg_coef'] is None:
                continue
            for count, d in enumerate(drivers):
                lbl_name = f'slope_{d}'
                _df_[lbl_name] = row['reg_coef'][count]
            slopes_data.append(_df_)
        slopes_data = pd.concat(slopes_data)
        print("Slopes for MVLR1: \n", slopes_data)
        
        
        slopes_data_2 = []
        for index_, row_ in reg_df_mvlr_2.iterrows():
            _df_ = grids_df_lst_2[row_['cell_id']]
            if row_['reg_coef'] is None:
                continue
            for count, d in enumerate(drivers_2):
                lbl_name = f'slope_{d}'
                _df_[lbl_name] = row_['reg_coef'][count]
            slopes_data_2.append(_df_)
        slopes_data_2 = pd.concat(slopes_data_2)
        print("Slopes for MVLR2: \n", slopes_data_2)


        fig_paths = []
        print("---> Plotting slopes.")
        for driver in drivers:
            # plot_path = plot_slope_maps_plotly(slopes_data, input_select_year, input_select_month, driver)
            plot_path = plot_slope_maps(slopes_data, input_select_year, input_select_month, driver) #better outcome than plotly
            fig_paths.append(plot_path)
        
        print("---> Plotting a few more.")
        for driver_2 in drivers_2:
            # plot_path = plot_slope_maps_plotly(slopes_data_2, input_select_year_2, input_select_month_2, driver_2)
            plot_path = plot_slope_maps(slopes_data_2, input_select_year_2, input_select_month_2, driver_2) #better outcome than plotly
            fig_paths.append(plot_path)
        
        print("--> Finished plotting.")
        print("--- %s seconds ---" % (time.time() - start_time))

        # https://stackoverflow.com/questions/71642795/browser-is-crashing-when-using-dropdown-component-with-large-data-in-plotly-dash/71692056#71692056 
        # https://lenspy.readthedocs.io/en/latest/quickstart.html#using-with-ploty-dash-applications
        # https://github.com/telemetrydb/lenspy/blob/master/lenspy/__init__.py 
        # if len(fig_paths) == 6:
        #     plot1 = DynamicPlot(fig_paths[0])
        #     plot2 = DynamicPlot(fig_paths[1])
        #     plot3 = DynamicPlot(fig_paths[2])
        #     plot4 = DynamicPlot(fig_paths[3])
        #     plot5 = DynamicPlot(fig_paths[4])
        #     plot6 = DynamicPlot(fig_paths[5])
        # return plot1.refine_plot, plot2.refine_plot, plot3.refine_plot, plot4.refine_plot, plot5.refine_plot, plot6.refine_plot
        return fig_paths[0], fig_paths[1], fig_paths[2], fig_paths[3], fig_paths[4], fig_paths[5]

        if len(fig_paths) == 5:
            return fig_paths[0], fig_paths[1], fig_paths[2], fig_paths[3], fig_paths[4], dash.no_update
        
        if len(fig_paths) == 4:
            return fig_paths[0], fig_paths[1], fig_paths[2], fig_paths[3], dash.no_update, dash.no_update
        
        if len(fig_paths) == 3:
            return fig_paths[0], fig_paths[1], fig_paths[2], dash.no_update, dash.no_update, dash.no_update
        
        if len(fig_paths) == 2:
            return fig_paths[0], fig_paths[1], dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

     else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

#---------------------------- 2. Run clustering and BIC Scores. ----------------------------
## o/p - 2 Dendrogram plots for natural and anthropogenic scenario + 2 BIC plots for natural and anthropogenic scenario

@app.callback(
    [Output('hc_img', 'src'), Output('hc_img_2', 'src')],
    [Input('btn_hc','n_clicks')])
# the order of parameter follows the order of input for callback.
def load_dendrograms(btn_hc):
    if ctx.triggered_id == 'btn_hc':
        print("Fetching Dendrograms.")

        global linkage_matrix
        global linkage_matrix_2
        global norm_hc_df
        global norm_hc_df_2
        # global dendro_dict_lst
        # global dendro_dict_lst_2

        print(reg_df_mvlr.head(5))
        print(reg_df_mvlr_2.head(5))

        # reg_df_mvlr = reg_df_mvlr.dropna()
        # reg_df_mvlr_2 = reg_df_mvlr_2.dropna()

        ## Set up dataframe for Hierarchical clustering
        fn_0 = lambda row: row['reg_coef'][0]
        fn_1 = lambda row: row['reg_coef'][1]
        fn_2 = lambda row: row['reg_coef'][2]
        # fn_3 = lambda row: row['reg_coef'][3]
        # fn_4 = lambda row: row['reg_coef'][4]

        hc_df = pd.DataFrame()
        hc_df['cell_id'] = reg_df_mvlr['cell_id']

        for count, d in enumerate(drivers):
            lbl_name = f'slope_{d}'
            if count == 0:
                hc_df[lbl_name] = reg_df_mvlr.apply(fn_0,axis=1)
            elif count == 1:
                hc_df[lbl_name] = reg_df_mvlr.apply(fn_1,axis=1)
            elif count == 2:
                hc_df[lbl_name] = reg_df_mvlr.apply(fn_2,axis=1)
            elif count == 3:
                hc_df[lbl_name] = reg_df_mvlr.apply(fn_3,axis=1)
            elif count == 4:
                hc_df[lbl_name] = reg_df_mvlr.apply(fn_4,axis=1)

        hc_df_2 = pd.DataFrame()
        hc_df_2['cell_id'] = reg_df_mvlr_2['cell_id']

        for count, d in enumerate(drivers_2):
            lbl_name = f'slope_{d}'
            if count == 0:
                hc_df_2[lbl_name] = reg_df_mvlr_2.apply(fn_0,axis=1)
            elif count == 1:
                hc_df_2[lbl_name] = reg_df_mvlr_2.apply(fn_1,axis=1)
            elif count == 2:
                hc_df_2[lbl_name] = reg_df_mvlr_2.apply(fn_2,axis=1)
            elif count == 3:
                hc_df_2[lbl_name] = reg_df_mvlr_2.apply(fn_3,axis=1)
            elif count == 4:
                hc_df_2[lbl_name] = reg_df_mvlr_2.apply(fn_4,axis=1)
        
        print("Building Dendrogram for 1st model.")
        file_name = f'Dendrogram_{year}_{month}'
        norm_hc_df, linkage_matrix, dend_fig_path = run_hc(hc_df, drivers, file_name)
        
        print("Building Dendrogram for 2nd model.")
        file_name = f'Dendrogram_{year}_{month}_2'
        norm_hc_df_2, linkage_matrix_2, dend_fig_path_2 = run_hc(hc_df_2, drivers_2, file_name)

        return dend_fig_path, dend_fig_path_2

    else:
        return dash.no_update, dash.no_update


@app.callback(
    [Output('bic_graph', 'figure'), Output('bic_graph_2', 'figure')],
    [Input('btn_bic','n_clicks')])
# the order of parameter follows the order of input for callback.
def load_bic(btn_bic):
    if ctx.triggered_id == 'btn_bic':
        print("Building BIC Graph for 1st model.")
        global dendro_dict_lst
        global dendro_dict_lst_2
        dendro_dict_lst, bic_cluster_num_fig = plot_bic_score_cluster_no(Z = linkage_matrix, norm_hc_df = norm_hc_df, year = year, month = month)
        
        print("Building BIC Graph for 2nd model.")
        dendro_dict_lst_2, bic_cluster_num_fig_2 = plot_bic_score_cluster_no(Z = linkage_matrix_2, norm_hc_df = norm_hc_df_2, year = year_2, month = month_2)
        return bic_cluster_num_fig, bic_cluster_num_fig_2

    else:
        return dash.no_update, dash.no_update

#---------------------------- 3. Set the parameters. Detect Clusters on Dendrogram. ----------------------------

@app.callback(Output('dendro_cut', 'src'),
    [Input(component_id='input_select_delta_variance', component_property='value'),
    Input(component_id='input_select_delta_height', component_property='value'),
    Input('btn_dendro','n_clicks'),
    Input(component_id='input_range_var', component_property='value'),
    Input(component_id='input_range_dist', component_property='value'),
    Input('btn_manual_dendro','n_clicks')]
    )
# the order of parameter follows the order of input for callback.
def cut_dendrogram(input_select_delta_variance, input_select_delta_height,btn_dendro, 
                    input_range_var, input_range_dist,btn_manual_dendro):
    global grid_index_list
    global del_var
    global del_dist

    if ctx.triggered_id == 'btn_dendro':
        print(f'---> Button pressed = {ctx.triggered_id}')
        print(f'---> Delta Variance = {input_select_delta_variance}')
        print(f'---> Delta Distance = {input_select_delta_height}')
        del_var = input_select_delta_variance
        del_dist = input_select_delta_height
    elif ctx.triggered_id == 'btn_manual_dendro':
        print(f'---> Button pressed = {ctx.triggered_id}')
        print(f'---> Delta Variance = {input_range_var}')
        print(f'---> Delta Distance = {input_range_dist}')
        del_var = input_range_var
        del_dist = input_range_dist

    else:
        return dash.no_update

    grid_index_list, clustered_dendro_fig_path = detect_clusters_on_dendrogram(norm_hc_df, del_var, del_dist, dendro_dict_lst)
    return clustered_dendro_fig_path

@app.callback(Output('dendro_cut_2', 'src'),
    [Input(component_id='input_select_delta_variance_2', component_property='value'),
    Input(component_id='input_select_delta_height_2', component_property='value'),
    Input('btn_dendro_2','n_clicks'),
    Input(component_id='input_range_var_2', component_property='value'),
    Input(component_id='input_range_dist_2', component_property='value'),
    Input('btn_manual_dendro_2','n_clicks')]
    )
# the order of parameter follows the order of input for callback.
def cut_dendrogram_2(input_select_delta_variance_2, input_select_delta_height_2,btn_dendro_2, 
                    input_range_var_2, input_range_dist_2,btn_manual_dendro_2):
    global grid_index_list_2
    global del_var_2
    global del_dist_2

    if ctx.triggered_id == 'btn_dendro_2':
        print(f'---> Button pressed = {ctx.triggered_id}')
        print(f'---> Delta Variance = {input_select_delta_variance_2}')
        print(f'---> Delta Distance = {input_select_delta_height_2}')
        del_var_2 = input_select_delta_variance_2
        del_dist_2 = input_select_delta_height_2
    elif ctx.triggered_id == 'btn_manual_dendro_2':
        print(f'---> Button pressed = {ctx.triggered_id}')
        print(f'---> Delta Variance = {input_range_var_2}')
        print(f'---> Delta Distance = {input_range_dist_2}')
        del_var_2 = input_range_var_2
        del_dist_2 = input_range_dist_2

    else:
        return dash.no_update

    grid_index_list_2, clustered_dendro_fig_path_2 = detect_clusters_on_dendrogram(norm_hc_df_2, del_var_2, del_dist_2, dendro_dict_lst_2)

    return clustered_dendro_fig_path_2


#---------------------------- 4. Plot Clustered maps. ----------------------------
@app.callback(
    [Output('cluster_map_image', 'src'), Output('cluster_map_image_2', 'src')],
    [Input('btn_get_cluster_maps','n_clicks')])
# the order of parameter follows the order of input for callback.
def load_cluster_maps(btn_get_cluster_maps):
    if ctx.triggered_id == 'btn_get_cluster_maps':
        global final_data
        global final_data_2
        global cluster_colors
        global cluster_colors_2
        global hc_df
        global hc_df_2

        print("--> Cluster maps are being prepared.")


        hc_df, cluster_colors, final_data, cluster_map_path = get_cluster_map(grid_index_list, norm_hc_df, grids_df_lst, drivers, del_var, del_dist, year, month )
        hc_df_2, cluster_colors_2, final_data_2, cluster_map_path_2 = get_cluster_map(grid_index_list_2, norm_hc_df_2, grids_df_lst_2, drivers_2, del_var_2, del_dist_2, year_2, month_2)
        
        print("--> Cluster maps are generated.")

        return cluster_map_path, cluster_map_path_2

    else:
        return dash.no_update, dash.no_update

#---------------------------- 5. Analyse clusters. ----------------------------
#----------------------------  1) Random Forest. 2) Mean / Median/ Std Deviation Tables 3) Slope Scatter plots ----------------------------
# @app.callback(
#     [Output('cluster_dist_image', 'src'), Output('cluster_dist_image_2', 'src')],
#     [Input('btn_get_cluster_details','n_clicks')])
# # the order of parameter follows the order of input for callback.
# def load_cluster_details(btn_get_cluster_details):
#     if ctx.triggered_id == 'btn_get_cluster_details':

#         distribution_fig_path = analyse_clusters(drivers, hc_df, cluster_colors)
#         distribution_fig_path_2 = analyse_clusters(drivers_2, hc_df_2, cluster_colors_2)

#         return distribution_fig_path, distribution_fig_path_2

#     else:
#         return dash.no_update, dash.no_update

@app.callback(
    [Output('cluster_dist_image', 'figure'), 
     Output('cluster_dist_image_2', 'figure'),
     Output('cluster_dist_image_3', 'figure'),
     Output('cluster_dist_image_4', 'figure'),
     Output('cluster_dist_image_5', 'figure'),
     Output('cluster_dist_image_6', 'figure'),
     ],
    [Input('btn_get_cluster_details','n_clicks')])
# the order of parameter follows the order of input for callback.
def load_cluster_details(btn_get_cluster_details):
    if ctx.triggered_id == 'btn_get_cluster_details':
        print("--> Scatter plots among slopes are being prepared.")
        distribution_fig_path = analyse_clusters_(drivers, hc_df)
        distribution_fig_path_2 = analyse_clusters_(drivers_2, hc_df_2)
        print("--> Scatter plots are generated.")

        return distribution_fig_path[0], distribution_fig_path[1], distribution_fig_path[2], distribution_fig_path_2[0], distribution_fig_path_2[1], distribution_fig_path_2[2]

    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback([Output('rf_fig', 'figure'), Output('rf_fig_2', 'figure')],[Input('input_decision_tree_nums','value'),Input('btn_run_random_forest','n_clicks')])
def load_random_forest_output(input_decision_tree_nums,btn_run_random_forest):
    if ctx.triggered_id == 'btn_run_random_forest':
        print("--> Random forest is running.")
        rf = get_random_forest_graphs(final_data, drivers, target, month, year, input_decision_tree_nums)
        rf_2 = get_random_forest_graphs(final_data_2, drivers_2, target_2, month_2, year_2, input_decision_tree_nums)
        print("--> Feature importance has been calculated.")
        return rf, rf_2
    else:
        return dash.no_update, dash.no_update

@app.callback([ Output('datatable-cluster-detail', 'data'),  Output('datatable-cluster-detail-2', 'data')],
[Input('btn_show_cluster_summary','n_clicks')])
def load_model_summary(btn_show_cluster_summary):
    if ctx.triggered_id == 'btn_show_cluster_summary':
        detail_df = get_cluster_summary_details(final_data, drivers)
        detail_df = detail_df.round(2)

        detail_df_2 = get_cluster_summary_details(final_data_2, drivers_2)
        detail_df_2 = detail_df_2.round(2)

        print("--> Tables are ready!")
        return detail_df.to_dict('records'), detail_df_2.to_dict('records')
    else:
        return dash.no_update, dash.no_update

#---------------------------- 6. Download Clustered Data. ----------------------------

@app.callback(
   [Output("download-dataframe-csv", "data"),  Output("download-dataframe-csv_2", "data"),],
    Input("btn_download", "n_clicks"),
    prevent_initial_call=True,
)
def download_clustered_data(n_clicks):
    print('---> Downloading the data in CSV files.')
    return dcc.send_data_frame(final_data.to_csv, f"model_1_{str(datetime.datetime.now)}.csv"), dcc.send_data_frame(final_data_2.to_csv, f"model_2_{str(datetime.datetime.now)}.csv")



# ------------------------------- *************** ------------------------------------
# ------------------------------- Run Dash Server ------------------------------------
if __name__ == '__main__':

    """
    NOTE: With the debug=True parameter passed to app.run_server(), Dash app will automatically reload 
    and reflect the code changes when we save our file in the code editor. 
    This saves the trouble of manually restarting the server every time we make a change.

    When use_reloader is set to False, the server will not automatically reload when changes are made to the source code files. 
    This can be useful in certain scenarios, such as when you want to run the server in a production environment 
    where automatic reloading is not desirable.

    """
    app.run_server(debug=True, use_reloader=True)

    # After the server is closed...
    print("******* Shutting the server down! *******")

    # test_dict = {"Cluster_labels": {"0": 1.0, "1": 2.0, "2": 3.0, "3": 4.0, "4": 5.0, "5": 6.0, "6": 7.0, "7": 8.0, "8": 9.0}, 
    # "Mean slope_SST": {"0": 11.617813892582348, "1": 411.2175598144531, "2": 79.00297546386719, "3": -12.861379809478816, "4": 18.110509959747606, "5": 27.648557662963867, "6": -30.071836471557617, "7": -85.80758562781936, "8": -228.46382796596473}, 
    # "Median slope_SST": {"0": 11.507404327392578, "1": 411.2175598144531, "2": 79.00297546386719, "3": -13.754770278930664, "4": 13.86928939819336, "5": 27.648557662963867, "6": -30.071836471557617, "7": -81.34871673583984, "8": -210.37831115722656}, "Std. Dev slope_SST": {"0": 4.964051403313165, "1": 0.0, "2": 0.0, "3": 23.785669956823245, "4": 19.654569090447925, "5": 0.0, "6": 0.0, "7": 28.640812695090776, "8": 79.17782721174007}, "Mean slope_DICP": {"0": 1.2953531831910219, "1": -24.67351722717285, "2": -14.651311874389648, "3": 2.299451827990995, "4": 1.8320281881492284, "5": 18.034093856811523, "6": 19.325162887573242, "7": 2.187595133282641, "8": 2.226955064725428}, "Median slope_DICP": {"0": 1.2720415592193604, "1": -24.67351722717285, "2": -14.651311874389648, "3": 2.260685682296753, "4": 1.7730202674865723, "5": 18.034093856811523, "6": 19.325162887573242, "7": 2.243786334991455, "8": 2.240457057952881}, "Std. Dev slope_DICP": {"0": 0.20948700977987195, "1": 0.0, "2": 0.0, "3": 0.6535840062718457, "4": 0.24247206498979432, "5": 0.0, "6": 0.0, "7": 0.48114367919936096, "8": 0.43048005765124575}, "Mean slope_ALK": {"0": -0.8988023009633921, "1": 24.868146896362305, "2": 14.95151138305664, "3": -1.9916649964457307, "4": -1.6161329827933397, "5": -21.93532943725586, "6": -11.006959915161133, "7": -1.8668428154577301, "8": -1.9967885072662237}, "Median slope_ALK": {"0": -0.8810020685195923, "1": 24.868146896362305, "2": 14.95151138305664, "3": -1.9725804328918457, "4": -1.531843662261963, "5": -21.93532943725586, "6": -11.006959915161133, "7": -1.9820913076400757, "8": -2.0467121601104736}, "Std. Dev slope_ALK": {"0": 0.20724851019648136, "1": 0.0, "2": 0.0, "3": 0.7179773367223017, "4": 0.3484541262405446, "5": 0.0, "6": 0.0, "7": 0.581153568964698, "8": 0.455063630616868}, "Area(in km. sq)": {"0": 264555.5, "1": 264555.5, "2": 264555.5, "3": 264555.5, "4": 264555.5, "5": 264555.5, "6": 264555.5, "7": 264555.5, "8": 264555.5}}

    """
    TODO:
    - Calc total runtime for each function
    - Maybe add a notification sound when a function is finished/output is ready
    - implement sea ice slider functionalities
    - Add license to the github code
    """