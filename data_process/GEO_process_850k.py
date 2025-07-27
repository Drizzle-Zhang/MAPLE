# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: heart_disease.py
# @time: 2023/10/12 21:43

import numpy as np
import pandas as pd
import pickle
import re
import os
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File

#%%
# AIRWAVE GSE147740
path_AIRWAVE = '/home/zhangyu/mnt_path/Data/AIRWAVE'
df_mat_GSE147740 = pd.read_csv(os.path.join(path_AIRWAVE, 'GSE147740_beta_QN_normalisation.txt'),
                               sep=' ', index_col=0)

file_series_matrix = os.path.join(path_AIRWAVE, 'GSE147740_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"") for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE147740 = pd.concat(list_series, axis=1)
df_meta_GSE147740['project_id'] = 'GSE147740'
df_meta_GSE147740['platform'] = '850k'
list_sample_GSE147740 = [idx for idx in df_mat_GSE147740.columns if idx in df_meta_GSE147740.index ]
df_meta_GSE147740 = df_meta_GSE147740.loc[list_sample_GSE147740, :]
df_meta_GSE147740.index = df_meta_GSE147740['sample_id'].tolist()
df_mat_GSE147740 = df_mat_GSE147740.loc[:, list_sample_GSE147740]
df_mat_GSE147740.columns = df_meta_GSE147740['sample_id'].tolist()
df_meta_GSE147740.columns = ['sample_id', 'age', 'sex', 'person id', 'disease state', 'project_id',
                            'platform']
df_meta_GSE147740.to_csv(os.path.join(path_AIRWAVE, 'GSE147740_meta.txt'))

# missing rate < 0.1
missing_rates = pd.Series(np.sum(np.isnan(df_mat_GSE147740), axis=1) / df_mat_GSE147740.shape[1],
                          index=df_mat_GSE147740.index)
cpgs_rm5_GSE147740 = list(missing_rates.loc[missing_rates < 0.05].index)


#%%
# GENOA GSE210255
path_GENOA = '/home/zhangyu/mnt_path/Data/GENOA'
df_mat_GSE210255 = pd.read_csv(os.path.join(path_GENOA, 'GSE210255_Beta_value_EPIC.txt'),
                               sep=' ', index_col=0)

file_series_matrix = os.path.join(path_GENOA, 'GSE210255_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"") for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE210255 = pd.concat(list_series, axis=1)
df_meta_GSE210255['project_id'] = 'GSE210255'
df_meta_GSE210255['platform'] = '850k'
df_meta_GSE210255.index = df_mat_GSE210255.columns
df_meta_GSE210255.index = df_meta_GSE210255['sample_id'].tolist()
df_mat_GSE210255.columns = df_meta_GSE210255['sample_id'].tolist()
df_meta_GSE210255.columns = ['sample_id', 'race', 'sex', 'age', 'plate', 'column', 'row',
                             'pedigreeid', 'subjectid', 'phase', 'project_id', 'platform']
df_meta_GSE210255.to_csv(os.path.join(path_GENOA, 'GSE210255_meta.txt'))

cpgs_GSE210255 = list(df_mat_GSE210255.index)

#%%
# SJLife GSE169156
path_SJLife = '/home/zhangyu/mnt_path/Data/SJLife'
df_mat_GSE169156 = pd.read_csv(
    os.path.join(path_SJLife, 'GSE169156_SJLIFE_IlluminaEPIC_2052samples_GEO_03162021_processed.txt'),
    sep='\t', index_col=0)
list_cols = [col for col in df_mat_GSE169156.columns if col[:9] != 'Detection']
df_mat_GSE169156 = df_mat_GSE169156.loc[:, list_cols]

file_series_matrix = os.path.join(path_SJLife, 'GSE169156_series_matrix.txt')
pattern = r'\[(.*?)\]'
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = []
            for one in list_line[1:]:
                matches = re.findall(pattern, one)
                result = matches[0]
                list_index.append(result.strip("\""))
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE169156 = pd.concat(list_series, axis=1)
df_meta_GSE169156['project_id'] = 'GSE169156'
df_meta_GSE169156['platform'] = '850k'
df_meta_GSE169156 = df_meta_GSE169156.loc[df_mat_GSE169156.columns, :]
df_meta_GSE169156.index = df_meta_GSE169156['sample_id'].tolist()
df_mat_GSE169156.columns = df_meta_GSE169156['sample_id'].tolist()
df_meta_GSE169156.to_csv(os.path.join(path_SJLife, 'GSE169156_meta.txt'))

cpgs_GSE169156 = list(df_mat_GSE169156.index)

#%%
# GSE196696
path_850k = '/home/zhangyu/mnt_path/Data/850k'
df_mat_GSE196696 = pd.read_csv(os.path.join(path_850k, 'GSE196696_processed_data.tsv'),
                               sep='\t', index_col=0)
list_cols = [col for col in df_mat_GSE196696.columns if col[:9] != 'Detection']
df_mat_GSE196696 = df_mat_GSE196696.loc[:, list_cols]

file_series_matrix = os.path.join(path_GENOA, 'GSE210255_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"") for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE210255 = pd.concat(list_series, axis=1)
df_meta_GSE210255['project_id'] = 'GSE147740'
df_meta_GSE210255['platform'] = '850k'

# %%
# GSE56046
path_HD = '/home/zhangyu/mnt_path/Data/heart_disease'
path_HD_process = os.path.join(path_HD, 'GSE56046_process')
df_mat_MESA = pd.read_csv(os.path.join(path_HD, 'GSE56046_methylome_normalized.txt'), sep='\t', index_col=0)

file_series_matrix = os.path.join(path_HD, 'GSE56046_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one[1:7] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            if list_line[1][1:4] == 'age':
                list_series.append(pd.Series(
                    [int(one.strip("\"").split(': ')[1]) for one in list_line[1:]],
                    index=list_index, name='age'))
            elif list_line[1][1:7] == 'plaque':
                list_series.append(pd.Series(
                    [int(one.strip("\"").split(': ')[1]) if one.strip("\"").split(': ')[1] != 'NA' else np.nan for one in list_line[1:]],
                    index=list_index, name='plaque'))
            elif list_line[1][1:4] == 'cac':
                list_series.append(pd.Series(
                    [float(one.strip("\"").split(': ')[1]) if one.strip("\"").split(': ')[1] != 'NA' else np.nan for one in list_line[1:]],
                    index=list_index, name='cac'))

df_meta = pd.concat(list_series, axis=1)
df_meta.to_csv(os.path.join(path_HD_process, 'GSE56046_meta.txt'))

# all index
new_train_index = df_meta['sample_id'].tolist()
with open(os.path.join(path_HD_process, "all_index.pkl"), 'wb') as f:
    pickle.dump(new_train_index, f)
    pickle.dump([], f)

# transform into dict
# all CpGs
file_all_HD_cpgs = os.path.join(path_HD_process, 'GSE56046_cpgs.txt')
write_stringList_2File(file_all_HD_cpgs, list(df_mat_MESA.index))
df_mat_all = df_mat_MESA.loc[:, [f"{idx}.Mvalue" for idx in df_meta.index]]
df_mat_all_beta = (1.0001*np.exp(df_mat_all)-0.0001)/(1+np.exp(df_mat_all))
df_mat_all_beta.columns = df_meta['sample_id'].tolist()
save_dict = {}
X_np = df_mat_all_beta.T
y_pd = df_meta

additional_list = ["sample_id", "age"]
for row_number in range(len(y_pd)):
    key_rename = y_pd.iloc[row_number].sample_id
    feature_np = np.array(X_np.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd.iloc[row_number]["age"].astype(np.float32)])
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

np.save(os.path.join(path_HD_process, 'Processed_all.npy'), save_dict)

# stroke
file_cpg_case = '/home/zhangyu/mnt_path/Data/EWAS_process/disease/cpgs_ewas_fudan_stroke_450k_1'
list_stroke_cpgs = read_stringList_FromFile(file_cpg_case)
df_mat_stroke = df_mat_MESA.loc[list_stroke_cpgs, [f"{idx}.Mvalue" for idx in df_meta.index]]
df_mat_stroke_beta = (1.0001*np.exp(df_mat_stroke)-0.0001)/(1+np.exp(df_mat_stroke))
df_mat_stroke_beta.columns = df_meta['sample_id'].tolist()

save_dict = {}
X_np = df_mat_stroke_beta.T
y_pd = df_meta

additional_list = ["sample_id", "age"]
for row_number in range(len(y_pd)):
    key_rename = y_pd.iloc[row_number].sample_id
    feature_np = np.array(X_np.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd.iloc[row_number]["age"].astype(np.float32)])
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

np.save(os.path.join(path_HD_process, 'Processed_stroke.npy'), save_dict)
