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
from backend import get_ocean_data, build_grids, fit_multivariate_lin_regression, plot_slope_maps

# for static images
import io
import base64
# -------------------------------------- Dash App ----------------------------------------

app = Dash(__name__) # always pass __name__ -> It is connected with the asset folder.
print(f"---> Loading Dash App: {app}")

## Global Variables
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
            children=html.Img(id='reg1_driver1_slopes_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="reg1_driver2_slopes_loading",
            type="default",
            children=html.Img(id='reg1_driver2_slopes_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="reg1_driver3_slopes_loading",
            type="default",
            children=html.Img(id='reg1_driver3_slopes_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),

    ], style={'text-align':'center'}),

    html.Br(),

    html.Div(id='regression_output_container_2', children=[
        dcc.Loading(
            id="reg2_driver1_slopes_loading",
            type="default",
            children=html.Img(id='reg2_driver1_slopes_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="reg2_driver2_slopes_loading",
            type="default",
            children=html.Img(id='reg2_driver2_slopes_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
         html.Br(),
        
        dcc.Loading(
            id="reg2_driver3_slopes_loading",
            type="default",
            children=html.Img(id='reg2_driver3_slopes_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),

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
            children=html.Img(id='hc1_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
        html.Br(),
        
        dcc.Loading(
            id="hc2_loading",
            type="default",
            children=html.Img(id='hc2_img', src = '', style={'height':'800px', 'width':'80%','align':'center'})),
         html.Br(),
        
    ], style={'text-align':'center'}),

    html.Br(),

    html.H2("Get BIC scores and cluster numbers for different possible pairs of delta_var and delta_dist."),

    html.Div(id='bic_button_container', children=[
        html.Button('Plot graphs.', id='btn_bic', style= green_button_style),
    ], style=dict(display='flex')),

    html.Div(id='clustering_output_container', children=[
        dcc.Loading(
            id="bic1_loading",
            type="default",
            children=dcc.Graph(id='bic_graph_1', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            ),
        html.Br(),
        
        dcc.Loading(
            id="bic2_loading",
            type="default",
            children=dcc.Graph(id='bic_graph_2', figure={}, style={'height':'500px', 'width':'100%','align':'center'}),
            ),
         html.Br(),
        
    ], style={'text-align':'center'}),

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

    Input('regression_button_container','n_clicks')])
# the order of parameter follows the order of input for callback.
def run_regression(input_select_year, input_select_month, checkboxes_driver, slider, checkboxes_target,
                   input_select_year_2, input_select_month_2, checkboxes_driver_2, slider_2, checkboxes_target_2,
                   regression_button_container):
     if ctx.triggered_id == 'regression_button_container':

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


        print("i/p:", input_select_year, input_select_month, checkboxes_driver, slider, checkboxes_target,
                   input_select_year_2, input_select_month_2, checkboxes_driver_2, slider_2, checkboxes_target_2)
        
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
        
        ## Remove grids with Zero number of coordinates
        reg_df_mvlr = reg_df_mvlr[reg_df_mvlr.data_count != 0]
        reg_df_mvlr_2 = reg_df_mvlr_2[reg_df_mvlr_2.data_count != 0]

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
            plot_path = plot_slope_maps(slopes_data, input_select_year, input_select_month, driver)
            fig_paths.append(plot_path)
        
        print("---> Plotting a few more.")
        for driver_2 in drivers_2:
            plot_path = plot_slope_maps(slopes_data_2, input_select_year_2, input_select_month_2, driver_2)
            fig_paths.append(plot_path)

        if len(fig_paths) == 6:
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


# ------------------------------- *************** ------------------------------------
# ------------------------------- Run Dash Server ------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
    print("******* Shutting the server down! *******")