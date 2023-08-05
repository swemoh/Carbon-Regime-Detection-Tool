# import pandas as pd
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

# Interactive UI
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, dash_table, State, ctx, dash_table  # pip install dash (version 2.0.0 or higher)
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html

# local functions
from load_dataset import get_ocean_data
from mult_lin_reg import run_model
from run_clustering import get_hierarchical_clustering_linkage_matrix, plot_dendrogram
from cut_dendrogram import build_cluster_points, plot_bic_score_cluster_no
from build_cluster_map import extract_clusters

# for static images
import io
import base64
# -------------------------------------- Dash App ----------------------------------------

app = Dash(__name__) # always pass __name__ -> It is connected with the asset folder.
print(f"---> Loading Dash App: {app}")

## Global Variables
Z_matrix = None
dendrogram_plot = None
grids_df_lst = None
norm_hc_df = None
final_data = None
grid_index_list = None
dendrogram_plot = None

global COLOMN_LIST, CLUSTER_DETAILS
COLOMN_LIST = ['nav_lat', 'nav_lon', 'SST', 'DIC', 'ALK', 'fco2', 'cluster',
       'slope_sst', 'slope_dic', 'slope_alk']
CLUSTER_DETAILS = ['Cluster Label','SST Slope', 'DIC Slope', 'ALK Slope']
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
                    {'label': 'Sea Surface Temperature', 'value': 'sst'},
                    {'label': 'Dissolved Inorganic Carbon', 'value': 'dicp'},
                    {'label': 'Salinity', 'value': 'sal'},
                    {'label': 'Alkalinity', 'value': 'alk'},
                    {'label': 'Sea Ice Coverege', 'value': 'seaice'},
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
                    {'label': 'Sea Surface Temperature', 'value': 'sst'},
                    {'label': 'Dissolved Inorganic Carbon', 'value': 'dicp'},
                    {'label': 'Salinity', 'value': 'sal'},
                    {'label': 'Alkalinity', 'value': 'alk'},
                    {'label': 'Sea Ice Coverege', 'value': 'seaice'},
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
            id="reg1_driver1_slopes",
            type="default",
            children=html.Img(id='reg1_driver1_slopes_img', src = '', style={'height':'500px', 'width':'60%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="reg1_driver2_slopes",
            type="default",
            children=html.Img(id='reg1_driver2_slopes_img', src = '', style={'height':'500px', 'width':'60%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="reg1_driver3_slopes",
            type="default",
            children=html.Img(id='reg1_driver3_slopes_img', src = '', style={'height':'500px', 'width':'60%','align':'center'})),

    ], style={'text-align':'center'}),

    html.Br(),

    html.Div(id='regression_output_container_2', children=[
        dcc.Loading(
            id="reg2_driver1_slopes",
            type="default",
            children=html.Img(id='reg2_driver1_slopes_img', src = '', style={'height':'500px', 'width':'60%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="reg2_driver2_slopes",
            type="default",
            children=html.Img(id='reg2_driver2_slopes_img', src = '', style={'height':'500px', 'width':'60%','align':'center'})),
         html.Br(),
        
        dcc.Loading(
            id="reg2_driver3_slopes",
            type="default",
            children=html.Img(id='reg2_driver3_slopes_img', src = '', style={'height':'500px', 'width':'60%','align':'center'})),

    ], style={'text-align':'center'}),

    html.Br(),
    html.Br(),

    html.H2("--> Run Hierarchical Clustering."),

    html.Div(id='clusetring_button_container', children=[
        html.Button('Run clustering.', id='btn_cluster', style= green_button_style),
    ], style=dict(display='flex')),

    ## Empty container
    html.Div(id='output_container', children=[]),
    ## Horizontal break
    html.Br(),
    # html.Button('Cluster', id='btn_cluster', style= green_button_style),
    html.Div(children=[dcc.Loading(
            id="loading-dend_1",
            type="default",
            children=html.Img(id='dendro_original_1', src = '', style={'height':'500px', 'width':'60%','align':'center'}),
        )], style={'text-align':'center'}),
    
    html.Br(),
    
    html.Div(children=[dcc.Loading(
            id="loading-dend_2",
            type="default",
            children=html.Img(id='dendro_original_2', src = '', style={'height':'500px', 'width':'60%','align':'center'}),
        )], style={'text-align':'center'}),

    ## Horizontal break
    html.Br(),
    html.H2(f"--> Check out BIC scores and no. of clusters for different values of change in distance and variance thresholds.",),
    html.Button('Get BIC scores.', id='get_bic_score', style= green_button_style),
    html.Br(),
    html.H3("The color of the bubble indicates the magnitude of the score. The bubble size corresponds to the number of clusters generated for a particular pair of threshold parameters."),
    html.Div(children=[
        dcc.Loading(
            id="loading-bubble",
            type="default",
            children=dcc.Graph(id='bic_graph', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            
        )], style={'text-align':'center'}),
    ## Horizontal break
    # html.Div(children=[dcc.Loading(
    #         id="loading-heatmap",
    #         type="default",
    #         children=dcc.Graph(id='heatmap_graph', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            
    #     )], style={'text-align':'center'}),
    ## Horizontal break
    html.H3("Note: Use this bubble graph to make informed decision for the threshold selections in next step.",
    style={'text-align':'center' , 'color':'gray'}),
    html.Br(),
    html.H2("Step 3: Set the threshold parameters for change in distance (delta_dist) and change in variance (delta_var).",),
    html.H2("Select desired values of thresholds from the list and then press the button at the right to begin the cluster selection procedure.",),
    html.Div(className="row", children=[
            dcc.Dropdown(id="input_select_delta_variance",
                options=[{"label": "Select Delta Variance", "value": 0.0},
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
    
    html.Br(),

    # html.Div([dbc.Button( "Primary", id="btn_dendro", color="primary", className="me-1")]),
    html.Div(children=[
        html.H3("I. The red markers are the links identifed as clusters and are annotated with respective labels.",),
        html.H3("II. The dotted black lines highlight the distance on the dendrogram at which the clusters have been found.",),
        dcc.Loading(
            id="loading-cluster",
            type="circle",
            children=[html.Img(id='dendro_cut', src = '', style={'height':'500px', 'width':'60%','align':'center'}),]
        )], style={'text-align':'center'}),

    html.H2(" Step 3: Collect the clustering labels and study the output.",),
    html.Button("Extract clustering labels.", id="btn_cluster_map", style= green_button_style),
    html.Br(),
    html.Div(children=[
                    html.H3('A. Detected ocean carbon regimes are shown below on the map.'),
                    dcc.Loading(
                        id="loading-cluster-map",
                        type="circle",
                        children=[
                        html.Img(id='cluster_map_image', src = '', style={'height':'800px', 'width':'80%',}),
                        # dcc.Graph(id='cluster_map_graph', style={'height':'1200px', 'width':'100%','align':'center'}),
                        ]),
                    html.Br(),
                    html.H3('B. Clustered multivariate linear regression coefficients distribution plot.'),
                    dcc.Loading(
                        id="loading-dist_image",
                        type="circle",
                        children=[
                        html.Img(id='cluster_dist_image', src = '', style={'height':'700px', 'width':'80%',}),
                        ]),
                    html.Br(),
                    html.H4('C. Clustering Overview: Averaged standardized values of oceanic CO2 drivers in each cluster.'),
                    dcc.Loading(
                        id="loading-datatable-cluster-detail",
                        type="circle",
                        children=[
                        dash_table.DataTable(
                        id='datatable-cluster-detail',
                        columns=[{'name': i, 'id': i} for i in CLUSTER_DETAILS],
                        page_current=0,
                        page_size=PAGE_SIZE,
                        page_action='custom',
                        style_table={'width':'50%','margin-left':'350px'},
                        style_header={'textAlign': 'center'},)
                        ]),
                    html.Br(),
                    html.H4("Here is the clustering output in a tabular format. You can only see random 20 rows. To access the complete data with more than 200,000 datapoints, click on the download button below.",
                            style={'color':'blue'}),
                    dcc.Loading(
                        id="loading-datatable-upload",
                        type="circle",
                        children=[
                            dash_table.DataTable(
                            id='datatable-upload-container',
                            columns=[{'name': i, 'id': i} for i in COLOMN_LIST],
                            page_current=0,
                            page_size=PAGE_SIZE,
                            page_action='custom',
                            style_table={'width':'70%','margin-left':'200px'},
                            style_header={'textAlign': 'center'})]),
                    html.Br(),], style={'text-align':'center'}),
    html.Button("Download CSV", id="btn_csv", style= green_button_style),
    dcc.Download(id="download-dataframe-csv",),
    html.Br(),
],) #'#EBEBEB'

# --------------------------- *************** -------------------------------
# ---------------------------- Callback functions ----------------------------
# --------------------------- *************** -------------------------------

#---------------------------- 1. Run regression. ----------------------------

## i/p - year, month, drivers and target
## o/p -  6 maps of the slopes in natural and anthropogenic scenario

@app.callback([Output('reg1_driver1_slopes', 'src'),
               Output('reg1_driver2_slopes', 'src'),
               Output('reg1_driver3_slopes', 'src'),

               Output('reg2_driver1_slopes', 'src'),
               Output('reg2_driver2_slopes', 'src'),
               Output('reg2_driver3_slopes', 'src'),

               ],
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

    Input('regression_button_container','n_clicks')])
# the order of parameter follows the order of input for callback.
def run_regression(input_select_year, input_select_month,btn_cluster):
     if ctx.triggered_id == 'regression_button_container':
        print("Do something.")
        return
     else:
        return dash.no_update

#---------------------------- 2. Run clustering. ----------------------------

## o/p - 2 Dendrogram plots for natural and anthropogenic scenario



#---------------------------- 3. Get BIC Scores. ----------------------------

## o/p - 2 BIC plots for natural and anthropogenic scenario




#---------------------------- Run regression and hierarchical clustering. ----------------------------
@app.callback([Output('dendro_original_1', 'src'),],
    [Input(component_id='input_select_year', component_property='value'),
    Input(component_id='input_select_month', component_property='value'),
    Input('btn_cluster','n_clicks')])
# the order of parameter follows the order of input for callback.
def get_dendrogram(input_select_year, input_select_month,btn_cluster):    
    if ctx.triggered_id == 'btn_cluster':
        print(f'---> Button pressed = {ctx.triggered_id}')
        print(f'---> Year = {input_select_year}')
        print(f'---> Month = {input_select_month}')

        ## Accessing global variables
        global Z_matrix
        global dend
        global grids_df_lst
        global norm_hc_df
        global dendrogram_plot

        # Load the dataset
        print('---> Loading data.')
        model_df = get_ocean_data(year = input_select_year, month = input_select_month)
        print('---> Running regression.')
        mult_linreg_df, grids_df_lst = run_model(model_df, feat_names=['sst', 'dic', 'alk'], target='fco2', cell_width=2)
        print('---> Clustering the ouput.')
        Z_matrix, norm_hc_df = get_hierarchical_clustering_linkage_matrix(mult_linreg_df)
        print('---> Building the dendrogram.')
        dendrogram_plot, dendro_fig_path = plot_dendrogram(Z_matrix, input_select_month, input_select_year)
        return dendro_fig_path
    else:
        return dash.no_update

#---------------------------- Get BIC score and cluster numbers. ----------------------------
@app.callback(Output('bic_graph', 'figure'),Input('get_bic_score','n_clicks'))
# the order of parameter follows the order of input for callback.
def get_bic_graph(n_clicks): 
    if ctx.triggered_id == 'get_bic_score':
        # bic_fig, cluster_num_fig = plot_bic_surface(Z = Z_matrix, norm_hc_df = norm_hc_df)
        # all_in_one_fig =  plot_bic_3d(Z = Z_matrix, norm_hc_df = norm_hc_df)
        bic_cluster_num_fig = plot_bic_score_cluster_no(Z = Z_matrix, norm_hc_df = norm_hc_df)
        return bic_cluster_num_fig
    else:
        return dash.no_update## specify no_update for as many number of times as the output parameters

#---------------------------- Cut the dendrogram. ----------------------------
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
    X = norm_hc_df.values
    clustered_dendro_fig_path, grid_index_list = build_cluster_points(delta_var = del_var, delta_dist = del_dist, 
                                        Z = Z_matrix, norm_hc_df = norm_hc_df, X= X,dend=dendrogram_plot)        
    return clustered_dendro_fig_path
    

# @app.callback(Output('dendro_cut', 'src'),
#     [Input(component_id='input_range_var', component_property='value'),
#     Input(component_id='input_range_dist', component_property='value'),
#     Input('btn_manual_dendro','n_clicks')]
#     )
# # the order of parameter follows the order of input for callback.
# def cut_dendrogram_manual_selection(input_range_var, input_range_dist,btn_manual_dendro):
#     global grid_index_list
#     if ctx.triggered_id == 'btn_manual_dendro':
#         print(f'---> Button pressed = {ctx.triggered_id}')
#         print(f'---> Delta Variance = {input_range_var}')
#         print(f'---> Delta Height = {input_range_dist}')
#         X = norm_hc_df.values
#         clustered_dendro_fig_path, grid_index_list = build_cluster_points(delta_var = input_range_var, 
#                                             delta_dist = input_range_dist, 
#                                             Z = Z_matrix, norm_hc_df = norm_hc_df, X= X)        
#         return clustered_dendro_fig_path
#     else:
#         return dash.no_update

#---------------------------- Collect the clusters. ----------------------------
@app.callback([Output('cluster_map_image', 'src'),# Output('cluster_map_graph', 'figure'), 
    Output('cluster_dist_image', 'src'),
    Output('datatable-cluster-detail', 'data'),
    Output('datatable-upload-container', 'data')
    ],
    Input('btn_cluster_map','n_clicks')
    )
# the order of parameter follows the order of input for callback.
def get_cluster_details(n_clicks):
    global final_data
    if ctx.triggered_id == 'btn_cluster_map':
        print(f'---> Button pressed = {ctx.triggered_id}')
        print('---> Extracting clustering details.')
        map_fig_path, data_dist_fig_path, cluster_info_df, final_data, bic = extract_clusters(grids_df_lst, grid_index_list, norm_hc_df)
        cluster_info_df = cluster_info_df.round(decimals = 2)

        ## Preparing final data for tabular visualization
        # print(f'---> Columns of the data table are: {final_data.columns}')
        final_data = final_data.drop(['sst','dic','alk','fco2'], axis=1).rename({'fco2_pre': 'fco2', 'sosstsst': 'SST'}, axis=1)
        # final_data = final_data.round(decimals = 2) # rounds entire dataframe
        final_data['SST']=final_data['SST'].apply(lambda x:round(x,2))
        final_data['DIC']=final_data['DIC'].apply(lambda x:round(x,2))
        final_data['ALK']=final_data['ALK'].apply(lambda x:round(x,2))
        final_data['fco2']=final_data['fco2'].apply(lambda x:round(x,2))
        final_data['slope_sst']=final_data['slope_sst'].apply(lambda x:round(x,2))
        final_data['slope_dic']=final_data['slope_dic'].apply(lambda x:round(x,2))
        final_data['slope_alk']=final_data['slope_alk'].apply(lambda x:round(x,2))
        
        print()
        print(f'---> Dataframe details:\n{final_data.info()}')
        print(final_data.head(5))
        df_sample = final_data.sample(n = PAGE_SIZE)
        print('---> Outputs are ready!')
        return map_fig_path, data_dist_fig_path,cluster_info_df.to_dict('records'), df_sample.to_dict('records')
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

#---------------------------- Download cluster details. ----------------------------
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def download_clustered_data(n_clicks):
    print('---> Downloading the data in a CSV file.')
    return dcc.send_data_frame(final_data.to_csv, f"clustered_ocean_data_{str(datetime.datetime.now)}.csv")


# ------------------------------- *************** ------------------------------------
# ------------------------------- Run Dash Server ------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
    print("******* Shutting the server down! *******")
