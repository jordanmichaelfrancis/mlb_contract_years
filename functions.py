import os, pdb, re, warnings
import numpy as np
import pandas as pd
import seaborn as sns

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
FILE_LOCS = {'batting':'./data/baseballdatabank-2019.2/core/Batting.csv',
             'wOBA':'./data/constants/wOBA_constants.csv',
             'teams':'./data/baseballdatabank-2019.2/core/TeamsFranchises.csv',
             'people':'./data/baseballdatabank-2019.2/core/People.csv',
             'pitching':'./data/baseballdatabank-2019.2/core/Pitching.csv',
             'service_time':'./data/service_time/service_time.csv',
             'dataset': './data/dataset.csv'}
                 
DIRECTORIES = {'NL':'./data/league_adj/nl/', 
               'AL':'./data/league_adj/al/', 
               'signed':'./data/free_agents/signed/',
               'unsigned':'./data/free_agents/unsigned/',
               'park_factors':'./data/park_factors/'}

TEAM_ABV_DICT_FOR_REMAP = {'CHA':'CHW', 'CHN':'CHC', 'LAA':'ANA',
                           'LAN':'LAD', 'NYA':'NYY', 'NYN':'NYM', 
                           'SLN':'STL', 'SFN':'SFG', 'KCA':'KCR', 
                           'SDN':'SDP', 'MIA':'FLA', 'TBA':'TBD',
                           'WAS':'WSN', 'FLO':'FLA'}

def format_series(input_series, is_num=True):
    if is_num:
        input_series = input_series.astype(int).astype(str)
    return(input_series)
    
def get_combined_index(input_df, left_col, right_col,
                           left_is_num=False, right_is_num = False):
    
    left_series, right_series = input_df[left_col], input_df[right_col]
    left_series = format_series(left_series, left_is_num)
    right_series = format_series(right_series, right_is_num)
    
    return(left_series+'_'+right_series)

def get_df_from_dir(dir_loc, index_col=None, header=0):
    dir_files_list = os.listdir(dir_loc)
    output_df = pd.concat([pd.read_csv(dir_loc+file_str, 
                          index_col=index_col, header=header) 
                           for file_str in dir_files_list], sort=False)
    return(output_df)

def format_lg_df(input_df, lg_str):
    input_df['league'] = lg_str
    output_df_index = get_combined_index(input_df, 'Season', 'league', left_is_num=True)
    output_df = input_df.set_index(output_df_index)
    return(output_df)

def sum_player_year(player_df):
    cols_to_sum = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 
                   'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'uBB', '1B', 
                   'PA']
                   
    year_summed = player_df[cols_to_sum].sum()
    other_fields = player_df[player_df.columns.difference(cols_to_sum)].copy()
    if year_summed['PA'] != 0:
        other_fields['park_factor'] = sum(player_df['park_factor'] * player_df['PA']/year_summed['PA'])
        other_fields['league_scalar'] = sum(player_df['league_scalar'] * player_df['PA']/year_summed['PA'])
    else:
        other_fields['park_factor'] = 1
        other_fields['league_scalar'] = 1
    meta_data_dict = {'teamID': 'MULT', 'lgID': 'MULT'}
    for key, value in meta_data_dict.items(): other_fields[key] = value
    year_output = pd.concat([year_summed, other_fields.iloc[0]])
    return(year_output)
    
def divide_w_zeros(numerator_val, denominator_val):
    numerator_val = pd.to_numeric(numerator_val, errors='coerce')
    num_val_array = np.array(numerator_val, dtype=np.float64)
    den_val_array = np.array(denominator_val, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
         return(num_val_array / den_val_array)
    
def get_season_age(batting_df):
    born_after_7 = batting_df['birthMonth']<7
    born_on_first = batting_df['birthDay']!=1
    born_after_7_1 = ~((born_after_7) & (born_on_first))
    age = batting_df['yearID'] - batting_df['birthYear']
    age_on_7_1 = age - born_after_7_1.astype(float)
    batting_df['age'] = age_on_7_1.astype(int)
    return(batting_df)

def get_fa_df(free_agent_dir, col_str):
    """Reads in csvs and creates single dataframe based on the file name"""
    
    def get_ids_series(free_agent_df):
        return(free_agent_df['Name'].apply(lambda x: x.split('\\')[1]))
        
    def get_and_format_df(file_str):
        data_df = pd.read_csv(free_agent_dir+file_str) 
        ids = get_ids_series(data_df)
        years = pd.Series([file_str[0:4]] * len(ids))
        id_year = ids + '_' + years
        is_signed = pd.Series(np.full(len(ids), True, dtype=bool))
            
        output_df = pd.concat([id_year, is_signed], axis=1)
        output_df.columns = ['ids', col_str]
        return(output_df)
    
    files_str_list = os.listdir(free_agent_dir)
    free_agent_df_list = [get_and_format_df(file_str)
                          for file_str in files_str_list]
    free_agent_df = pd.concat(free_agent_df_list, ignore_index=True, sort=False)
    free_agent_df = free_agent_df.set_index('ids')
    return(free_agent_df)

def summarize_dfs(*args, col_name='wRC+', output_names=[]):
    data_list = [df.describe()[col_name] for df in args]
    output_df = pd.concat(data_list, axis=1)
    if output_names:
        output_df.columns = output_names
    return(output_df)
    
def plot_histogram(*args, col_name='wRC+'):
    plt = sns.distplot(args[0][col_name], color="blue", label="Free Agents")
    plt = sns.distplot(args[1][col_name], color="red", label="Everyone Else")
    plt.legend()

def get_t1_wRC_plus(input_df):
    
    def get_previous_wRC_plus(player_info):
        player_df = input_df[input_df['playerID']==player_info['playerID']]
        player_one_yr_ago = player_df[player_df['yearID']==player_info['yearID']-1]
        wRC_plus = player_one_yr_ago['wRC+']
        try:
            return(wRC_plus.iloc[0])
        except:
            return(np.nan)
            
    input_df['wRC+_t-1'] = input_df.apply(get_previous_wRC_plus, 
                                              axis=1, result_type='reduce')
    return(input_df)
    
    