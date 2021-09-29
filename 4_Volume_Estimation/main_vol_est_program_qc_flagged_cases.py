
import pandas as pd
import os
pd.set_option('mode.chained_assignment', None)

parent_directory = 'path/to/dataset/parent-directory/folder'
RESULT_FILE_NAME = 'Results_Est_Vol_LAV_withQC'
results_dir = os.path.join(parent_directory, RESULT_FILE_NAME)
current_version = '1'
log_save_dir = results_dir + '/' + current_version + '/'
data_file_name = log_save_dir + 'results.csv'

# Use pandas to read output measurement csv file
df_table = pd.read_csv(data_file_name)
df_table.drop('Unnamed: 0', axis=1, inplace=True)

# Conditions:
# ===========
# Apply Condition 1: 
mean_thresh_high = 0.65 
A = (df_table['QC_Score_mean'] <= mean_thresh_high)

# Apply Condition 2: 
perc_thresh_low = 0.2
B = (df_table['QC_Score_perc'] <= perc_thresh_low)

# Apply Condition 3: 
perc_no_thresh_low = 3
C = (df_table['QC_Score_perc_no'] <= perc_no_thresh_low)

# Apply Multiple Conditions: 
df_table_1 = df_table.loc[A | B | C]

# Apply Condition 4: - Remove rows that satisfy the following condition
mean_thresh_low = 0.60
perc_thresh_high = 0.60
perc_no_thresh_high = 4

D1 = (df_table_1['QC_Score_mean'] >= mean_thresh_low)
D2 = (df_table_1['QC_Score_perc'] >= perc_thresh_high)
D3 = (df_table_1['QC_Score_perc_no'] >= perc_no_thresh_high)

df_table_2 = df_table_1.loc[D1 & D2 & D3]
df_table_1.drop((df_table_2.index[:]), inplace=True)

df_table_final = df_table_1

# Save
data_file_name_check = log_save_dir + 'results_flagged'
df_table_final.to_csv(data_file_name_check + '.csv', encoding='utf-8', index=True)
df_table_final.to_excel(data_file_name_check + '.xlsx', index=True) 
