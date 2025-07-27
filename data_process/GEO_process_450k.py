# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: heart_disease.py
# @time: 2023/10/12 21:43

import numpy as np
import pandas as pd
import dask.dataframe as dd
import pickle
import re
import os
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File
from utils.common_utils import data_augmentation, impute_methy, read_single_csv

#%%
# LOLIPOP GSE55763
path_LOLIPOP = '/home/zhangyu/mnt_path/Data/LOLIPOP'
df_mat_GSE55763 = read_single_csv(os.path.join(path_LOLIPOP, 'GSE55763_beta_GMQN_BMIQ.txt'))
# df_mat_GSE55763 = \
#     dd.read_csv(os.path.join(path_LOLIPOP, 'GSE55763_beta_GMQN_BMIQ.txt'), index_col=0)
# df_mat_GSE55763 = pd.read_csv(os.path.join(path_LOLIPOP, 'GSE55763_normalized_betas.txt'),
#                                sep='\t', index_col=0)
# df_mat_GSE55763 = df_mat_GSE55763.loc[:,
#                         [one for one in df_mat_GSE55763.columns if one[:9] != 'Detection']]

file_series_matrix = os.path.join(path_LOLIPOP, 'GSE55763_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(", ")[1] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE55763 = pd.concat(list_series, axis=1)
df_meta_GSE55763['project_id'] = 'GSE55763'
df_meta_GSE55763['platform'] = '450k'
df_meta_GSE55763['ori_sample_id'] = df_meta_GSE55763.index
# list_sample_GSE55763 = [idx for idx in df_mat_GSE55763.columns if idx in df_meta_GSE55763.index ]
# df_meta_GSE55763 = df_meta_GSE55763.loc[list_sample_GSE55763, :]
df_meta_GSE55763.index = df_meta_GSE55763['sample_id'].tolist()
df_meta_GSE55763.columns = \
    ['sample_id', 'tissue', 'dataset', 'sex', 'age', 'project_id', 'platform', 'ori_sample_id']
df_meta_GSE55763.to_csv(os.path.join(path_LOLIPOP, 'GSE55763_meta.tsv'), sep='\t')
# df_mat_GSE55763 = df_mat_GSE55763.loc[:, list_sample_GSE55763]
df_mat_GSE55763.columns = df_meta_GSE55763['sample_id'].tolist()

# missing rate < 0.1
# missing_rates = pd.Series(np.sum(np.isnan(df_mat_GSE55763), axis=1) / df_mat_GSE55763.shape[1],
#                           index=df_mat_GSE55763.index)
# cpgs_rm10_GSE55763 = list(missing_rates.loc[missing_rates < 0.1].index)
cpgs_rm10_GSE55763 = \
    read_stringList_FromFile("/home/zhangyu/mnt_path/Data/EWAS_process/age/450k_rm10/cpgs_list.txt")
X_mat_imp_GSE55763, sample_GSE55763 = \
    impute_methy(df_mat_GSE55763.loc[cpgs_rm10_GSE55763, :], by_project=False)
df_mat_GSE55763_450k = pd.DataFrame(X_mat_imp_GSE55763,
                                    index=sample_GSE55763, columns=cpgs_rm10_GSE55763)

# all index
new_train_index = df_meta_GSE55763['sample_id'].tolist()
with open(os.path.join(path_LOLIPOP, "all_index.pkl"), 'wb') as f:
    pickle.dump(new_train_index, f)
    pickle.dump([], f)

# transform into dict
# all CpGs
file_GSE55763_cpgs = os.path.join(path_LOLIPOP, 'GSE55763_cpgs_GMQN.txt')
write_stringList_2File(file_GSE55763_cpgs, list(df_mat_GSE55763_450k.columns))
df_mat_GSE55763_450k.to_csv(os.path.join(path_LOLIPOP, 'GSE55763_mat_GMQN_fillna.txt'))

save_dict = {}
X_np = df_mat_GSE55763_450k
y_pd = df_meta_GSE55763

additional_list = df_meta_GSE55763.columns
for row_number in range(len(y_pd)):
    key_rename = y_pd.iloc[row_number].sample_id
    feature_np = np.array(X_np.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd.iloc[row_number]["age"]]).astype(np.float32)
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

np.save(os.path.join(path_LOLIPOP, 'Processed_all_GMQN.npy'), save_dict)

# %%
# GSE210254 450k
path_GENOA = '/home/zhangyu/mnt_path/Data/GENOA'
path_GSE210254_process = os.path.join(path_GENOA, 'GSE210254_process')
if not os.path.exists(path_GSE210254_process):
    os.mkdir(path_GSE210254_process)

file_series_matrix = os.path.join(path_GENOA, 'GSE210254_series_matrix.txt')
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

df_meta_GSE210254 = pd.concat(list_series, axis=1)
df_meta_GSE210254['ori_sample_id'] = [one.replace('sample', '') for one in df_meta_GSE210254.index]
df_meta_GSE210254.index = df_meta_GSE210254['sample_id'].tolist()
df_meta_GSE210254 = df_meta_GSE210254.loc[:, ['sample_id', 'age(yrs)', 'gender', 'ori_sample_id']]
df_meta_GSE210254.columns = ['sample_id', 'age', 'sex', 'ori_sample_id']
df_meta_GSE210254['project_id'] = 'GSE210254'
df_meta_GSE210254['platform'] = '450k'
df_meta_GSE210254['tissue'] = 'leukocyte'

df_meta_GSE210254.to_csv(os.path.join(path_GSE210254_process, 'GSE210254_meta.tsv'), sep='\t')


# %%
# GSE72680
path_450k = '/home/zhangyu/mnt_path/Data/450k'

file_series_matrix = os.path.join(path_450k, 'GSE72680_series_matrix.txt')
list_dict = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_geo_accession':
            list_gsm = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_gsm)):
                list_dict.append(dict(sample_id=list_gsm[one_idx]))
        elif list_line[0] == '!Sample_description':
            list_id = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_gsm)):
                list_dict[one_idx].update(dict(ori_sample_id=list_id[one_idx]))
        elif list_line[0] == '!Sample_characteristics_ch1':
            list_cha = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_gsm)):
                col_name = list_cha[one_idx].strip("\"").split(': ')[0]
                if col_name == '':
                    continue
                col_value = list_cha[one_idx].strip("\"").split(': ')[1]
                list_dict[one_idx].update({col_name: col_value})

df_meta_GSE72680 = pd.DataFrame(list_dict)
df_meta_GSE72680.index = df_meta_GSE72680['sample_id'].tolist()
df_meta_GSE72680['project_id'] = 'GSE72680'
df_meta_GSE72680['platform'] = '450k'
df_meta_GSE72680['tissue'] = 'whole blood'

df_meta_GSE72680.to_csv(os.path.join(path_450k, 'GSE72680_meta.tsv'), sep='\t')

# GSE40279
path_450k = '/home/zhangyu/mnt_path/Data/450k'

file_series_matrix = os.path.join(path_450k, 'GSE40279_series_matrix.txt')
list_dict = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_geo_accession':
            list_gsm = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_gsm)):
                list_dict.append(dict(sample_id=list_gsm[one_idx]))
        elif list_line[0] == '!Sample_characteristics_ch1':
            list_cha = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_gsm)):
                col_name = list_cha[one_idx].strip("\"").split(': ')[0]
                if col_name == '':
                    continue
                col_value = list_cha[one_idx].strip("\"").split(': ')[1]
                list_dict[one_idx].update({col_name: col_value})

df_meta_GSE40279 = pd.DataFrame(list_dict)
df_meta_GSE40279.index = df_meta_GSE40279['sample_id'].tolist()
df_meta_GSE40279 = df_meta_GSE40279.loc[:, ['sample_id', 'age (y)', 'gender', 'ethnicity']]
df_meta_GSE40279.columns = ['sample_id', 'age', 'sex', 'ethnicity']
df_meta_GSE40279['project_id'] = 'GSE40279'
df_meta_GSE40279['platform'] = '450k'
df_meta_GSE40279['tissue'] = 'whole blood'

df_meta_GSE40279.to_csv(os.path.join(path_450k, 'GSE40279_meta.tsv'), sep='\t')

# GSE87571
path_450k = '/home/zhangyu/mnt_path/Data/450k'

file_series_matrix = os.path.join(path_450k, 'GSE87571_series_matrix.txt')
list_dict = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_geo_accession':
            list_gsm = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_gsm)):
                list_dict.append(dict(sample_id=list_gsm[one_idx]))
        elif list_line[0] == '!Sample_characteristics_ch1':
            list_cha = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_gsm)):
                col_name = list_cha[one_idx].strip("\"").split(': ')[0]
                if col_name == '':
                    continue
                col_value = list_cha[one_idx].strip("\"").split(': ')[1]
                list_dict[one_idx].update({col_name: col_value})

df_meta_GSE87571 = pd.DataFrame(list_dict)
df_meta_GSE87571.index = df_meta_GSE87571['sample_id'].tolist()
df_meta_GSE87571['project_id'] = 'GSE87571'
df_meta_GSE87571['platform'] = '450k'

df_meta_GSE87571.to_csv(os.path.join(path_450k, 'GSE87571_meta.tsv'), sep='\t')
