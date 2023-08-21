
import pandas as pd
import numpy as np
from sklearn import linear_model
# for regression P values
import statsmodels.api as sm


def get_cell_range(start, end,cell_width =10):
    '''
    get the ranges of the cells
    '''
    num_iter = (abs(start) + abs(end))/cell_width
    range_lst = []
    for i in range(int(num_iter)+1):
        range_lst.append(start+i*cell_width)
#         print(start+i*cell_width)
    return range_lst


def fit_multivariate_lin_regression(grids_df_lst,feat_names,target):
    '''
    https://satishgunjal.com/multivariate_lr_scikit/ 
    '''
    # Fit Regression model
    grid_reg_score = []
    grid_reg_coef = []
    grid_reg_intercept = []
    p_values_intercept = []
    p_values_sst = []
    p_values_dic = []
    p_values_alk = []
    
    
    grid_reg_score_2 = []
    grid_reg_coef_2 = []
    grid_reg_intercept_2 = []
    
    grid_reg_score_3 = []
    grid_reg_coef_3 = []
    grid_reg_intercept_3 = []
    
    data_count = []
    
    count = 0
    
    for grid_i in grids_df_lst:
#         feat_lst=[]
#         for index, row in grid_i.iterrows():
#             row_feat_lst = [row[feat] for feat in feat_names]
#             feat_lst.append(np.array(row_feat_lst))
        
        X = grid_i.values[:,2:5]
        y = np.array(grid_i[target].values)
   
        if np.isnan(X).any():
            print(X)
            raise ValueError
        
        if (len(X) == 0) or (len(y) == 0) or(len(X) == 1) or (len(y) == 1) :
            grid_reg_score.append(None)
            grid_reg_coef.append(None)
            grid_reg_intercept.append(None)
            p_values_intercept.append(None)
            p_values_sst.append(None)
            p_values_dic.append(None)
            p_values_alk.append(None)

            grid_reg_score_2.append(None)
            grid_reg_coef_2.append(None)
            grid_reg_intercept_2.append(None)
            
            grid_reg_score_3.append(None)
            grid_reg_coef_3.append(None)
            grid_reg_intercept_3.append(None)
            
            data_count.append(0)
            
        else:
            data_count.append(len(X))
        
            # y = mX + c
            lin_reg = linear_model.LinearRegression().fit(X, y)
            
            X2 = sm.add_constant(X)
            est = sm.OLS(y, X2).fit()
            _p_ = est.pvalues
               
            p_values_intercept.append(_p_[0])
            p_values_sst.append(_p_[1])
            p_values_dic.append(_p_[2])
            p_values_alk.append(_p_[3])
            
            
            grid_reg_coef.append(lin_reg.coef_) #slope m
            grid_reg_intercept.append(lin_reg.intercept_) #intercept c
            grid_reg_score.append(lin_reg.score(X, y)) #quality or a confidence score
            
            
            count = count + 1
    
    
    save_df = pd.DataFrame(columns=['cell_id','lon_min','lon_max','lat_min','lat_max', 'data_count',
                                    'reg_score','reg_coef', 'reg_intercept',
                                    'p_intercept', 'p_sst', 'p_dic', 'p_alk',
                                   ])
    nav_lat_max_lst = []
    nav_lat_min_lst = []
    nav_lon_max_lst = []
    nav_lon_min_lst = []

    for grid_i in grids_df_lst:
        nav_lat_max_lst.append(grid_i['nav_lat'].max())
        nav_lat_min_lst.append(grid_i['nav_lat'].min())
        nav_lon_max_lst.append(grid_i['nav_lon'].max())
        nav_lon_min_lst.append(grid_i['nav_lon'].min())
    
    save_df['cell_id'] = range(0, len(grids_df_lst))
    save_df['lon_min'] = nav_lon_min_lst
    save_df['lon_max'] = nav_lon_max_lst
    save_df['lat_min'] = nav_lat_min_lst
    save_df['lat_max'] = nav_lat_max_lst
    save_df['data_count'] = data_count
    
    save_df['reg_score'] = grid_reg_score
    save_df['reg_coef'] = grid_reg_coef
    save_df['reg_intercept'] = grid_reg_intercept
    
    save_df['p_intercept'] = p_values_intercept
    save_df['p_sst'] = p_values_sst
    save_df['p_dic'] = p_values_dic
    save_df['p_alk'] = p_values_alk
    
#     save_df['reg_score_ridge'] = grid_reg_score_2
#     save_df['reg_coef_ridge'] = grid_reg_coef_2
#     save_df['reg_intercept_ridge'] = grid_reg_intercept_2
#     save_df['reg_score_lasso'] = grid_reg_score_3
#     save_df['reg_coef_lasso'] = grid_reg_coef_3
#     save_df['reg_intercept_lasso'] = grid_reg_intercept_3
        
    return save_df


def run_model(df_month, feat_names=['sst', 'dic', 'alk'], target='fco2', cell_width=2):
    
    # Prepare the cells
    nav_lat_grids = get_cell_range(start = -90, end = 90 ,cell_width = cell_width)
    nav_lon_grids = get_cell_range(start = -180, end = 180 ,cell_width = cell_width)
    
    if nav_lat_grids[-1] != 90:
        nav_lat_grids.append(90)
        
    if nav_lon_grids[-1] != 180:
        nav_lon_grids.append(180)
    
    
    # Build the grids. Store in a list.
    grids_df_lst=[]
    for lat_i in range(len(nav_lat_grids)):
        for lon_j in range(len(nav_lon_grids)):
            if((nav_lat_grids[lat_i] == 90) or (nav_lon_grids[lon_j] == 180)):
                break
            elif ((lat_i == len(nav_lat_grids) - 1) or (lon_j == len(nav_lon_grids) - 1)):
                break
            else:
                _df_ = df_month.loc[
                    (df_month['nav_lat'] >= nav_lat_grids[lat_i]) & 
                    (df_month['nav_lat'] <  nav_lat_grids[lat_i+1]) &
                    (df_month['nav_lon'] >= nav_lon_grids[lon_j]) & 
                    (df_month['nav_lon'] <  nav_lon_grids[lon_j+1])
                                ]
                grids_df_lst.append(_df_)
                
    
    print(f"\n Total no. of generated cells: {len(grids_df_lst)}")
                
    # fit linear regression
    save_df = fit_multivariate_lin_regression(grids_df_lst,feat_names,target)
    
    return save_df, grids_df_lst