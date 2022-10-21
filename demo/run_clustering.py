import pandas as pd
# import plotly.figure_factory as ff
from functools import partial
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.spatial.distance import pdist

from time import time
import datetime

# for static images
import io
import base64
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def get_hierarchical_clustering_linkage_matrix(reg_df):

    ## Remove grids with Zero number of coordinates
    reg_df = reg_df[reg_df.data_count != 0]

    ## Filter using P values
    ## null hypothesis: that the variable has no correlation with the dependent variable.
    ## significance level = 0.04 
    ## > significance level implies null hypothesis is accepted.
    reg_df_p = reg_df[(reg_df.p_sst <= 0.04) & (reg_df.p_dic <= 0.04)& (reg_df.p_alk <= 0.04)]

    ## Set up dataframe for Hierarchical clustering
    fn_sst = lambda row: row['reg_coef'][0]
    fn_dic = lambda row: row['reg_coef'][1]
    fn_alk = lambda row: row['reg_coef'][2]

    hc_df = pd.DataFrame()
    hc_df['cell_id'] = reg_df['cell_id']
    hc_df['slope_sst'] = reg_df_p.apply(fn_sst,axis=1)
    hc_df['slope_dic'] = reg_df_p.apply(fn_dic,axis=1)
    hc_df['slope_alk'] = reg_df_p.apply(fn_alk,axis=1)
    
    norm_hc_df = hc_df[['slope_sst', 'slope_dic', 'slope_alk']].dropna()
    
    # mean = 0, STD = 1
    norm_hc_df=(norm_hc_df-norm_hc_df.mean())/norm_hc_df.std()

    # Normalized between 0 and 1
    # norm_hc_df=(norm_hc_df-norm_hc_df.min())/(norm_hc_df.max()-norm_hc_df.min())

    X = norm_hc_df.values
    # x_ = X[0:5, :]
    Z = hierarchy.linkage(X, method='ward')
    return Z, norm_hc_df
    
    
    # pw_ed_func = partial(pdist, metric='euclidean')
    # print(datetime.datetime.now())
    # fig = ff.create_dendrogram(x_, distfun=pw_ed_func , linkagefun=lambda x: hierarchy.linkage(x, 'ward'))
    # print(datetime.datetime.now())
    # return fig

def plot_dendrogram(Z_matrix, input_select_month, input_select_year):
    ## Store Dendrogram
    buf = io.BytesIO() # in-memory files
    plt.figure(figsize=(20,12))
    plt.title(f"Dendrograms for month = {input_select_month}, year = {input_select_year}",
        fontsize=30)
    plt.xlabel('Individual grid cells', fontsize=30)
    plt.ylabel('Distance', fontsize=30)
    print(datetime.datetime.now())
    dendrogram_plot = hierarchy.dendrogram(Z_matrix, color_threshold=0)
    print(datetime.datetime.now())

    local_file_path = f'tmp/dend_original_{str(input_select_year)}_{input_select_month}.jpg'
    plt.savefig(local_file_path)
    plt.savefig(buf, format = "jpg") # save to the above file object
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.
    dendro_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    dendro_fig_path = f"data:image/jpg;base64,{dendro_data}"
    print('---> Returning the dendrogram path.')

    return dendrogram_plot, dendro_fig_path