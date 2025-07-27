# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: dict_850k_ctrl.py
# @time: 2023/11/23 16:39

import numpy as np
import pandas as pd
import datatable as dt
import pickle
import  os
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File
from utils.common_utils import read_single_csv, data_to_np_dict, impute_methy, order_cpg_to_ref


def save_mat_dict(df_mat_in, df_meta_in, file_cpgs, file_mat, file_npy, list_npy,
                  set_cpgs=None, missing_rate=0.2):
    missing_rates = \
        pd.Series(np.sum(np.isnan(df_mat_in), axis=1) / df_mat_in.shape[1],
                  index=df_mat_in.index)
    df_mat_in = df_mat_in.loc[missing_rates.loc[missing_rates <= missing_rate].index, :]
    if set_cpgs is None:
        df_mat = df_mat_in
    else:
        df_mat = df_mat_in.loc[[one for one in df_mat_in.index if one in set_cpgs], :]

    write_stringList_2File(file_cpgs, list(df_mat.index))

    X_mat_imp, samples = impute_methy(df_mat, by_project=False)
    df_mat_imp = pd.DataFrame(X_mat_imp, index=samples, columns=df_mat.index)
    df_mat_imp_T = df_mat_imp.T
    df_mat_imp_T.insert(loc=0, column='CpG', value=df_mat.index)
    dt_mat_imp = dt.Frame(df_mat_imp_T)
    dt_mat_imp.to_csv(file_mat)

    # save dict
    data_to_np_dict(df_mat_imp, df_meta_in, df_meta_in.index, list_npy, file_npy)

    return df_mat_imp


#%%
# Keep CpGs
cpg_list_disease = read_stringList_FromFile(
    "/home/zhangyu/mnt_path/Data/EWAS_process/disease/450k_rm10/cpgs_list.txt")
cpg_list_age = read_stringList_FromFile(
    "/home/zhangyu/mnt_path/Data/EWAS_process/age/450k_rm10/cpgs_list.txt")
cpg_list_850k = read_stringList_FromFile(
    "/home/zhangyu/mnt_path/Data/jizhuan/process/cpgs_jizhuan_850k_all.txt")
set_keep_cpgs = set(cpg_list_disease).intersection(set(cpg_list_850k)).intersection(
    set(cpg_list_age))

#%%
# GSE196696 850k
path_850k = '/home/zhangyu/mnt_path/Data/850k'
path_GSE196696_process = os.path.join(path_850k, 'GSE196696_process')
if not os.path.exists(path_GSE196696_process):
    os.mkdir(path_GSE196696_process)

file_series_matrix = os.path.join(path_850k, 'GSE196696_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' - ')[1] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE196696 = pd.concat(list_series, axis=1)
df_meta_GSE196696['project_id'] = 'GSE196696'
df_meta_GSE196696['platform'] = '850k'
df_meta_GSE196696.columns = ['sample_id', 'sex', 'age', 'tissue', 'batch', 'project_id', 'platform']

df_mat_GSE196696 = read_single_csv('/home/zhangyu/mnt_path/Data/850k/GSE196696_beta_GMQN_BMIQ_450k.txt')
# df_mat_GSE196696 = read_single_csv('/home/zhangyu/mnt_path/Data/850k/GSE196696_beta_GMQN_BMIQ.txt')
df_mat_GSE196696.columns = df_meta_GSE196696.loc[df_mat_GSE196696.columns, 'sample_id'].tolist()

df_meta_GSE196696['ori_sample_id'] = df_meta_GSE196696.index
df_meta_GSE196696.index = df_meta_GSE196696['sample_id'].tolist()
df_meta_GSE196696.to_csv(os.path.join(path_GSE196696_process, 'GSE196696_meta.tsv'), sep='\t')

missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE196696), axis=1) / df_mat_GSE196696.shape[1],
              index=df_mat_GSE196696.index)
df_mat_GSE196696 = df_mat_GSE196696.loc[
                            missing_rates.loc[missing_rates <= 0.2].index, :]
df_mat_GSE196696 = \
    df_mat_GSE196696.loc[[one for one in df_mat_GSE196696.index if one in set_keep_cpgs], :]

file_GSE196696_cpgs = os.path.join(path_GSE196696_process, 'GSE196696_cpgs_450k.txt')
write_stringList_2File(file_GSE196696_cpgs, list(df_mat_GSE196696.index))

X_mat_GSE196696_imp, sample_GSE196696 = \
    impute_methy(df_mat_GSE196696, by_project=False)
df_mat_GSE196696_imp = pd.DataFrame(
    X_mat_GSE196696_imp, index=sample_GSE196696, columns=df_mat_GSE196696.index)
df_mat_GSE196696_imp.T.to_csv(
    os.path.join(path_GSE196696_process, 'GSE196696_beta_GMQN_BMIQ_450k_impute.csv'))

# save dict
data_to_np_dict(df_mat_GSE196696_imp, df_meta_GSE196696, df_meta_GSE196696.index,
                ["sample_id", 'sex', "age"],
                path_GSE196696_process, 'Processed_all_450k.npy')

# 850k
df_mat_GSE196696 = dt.fread(os.path.join(path_850k, 'GSE196696_beta_GMQN_BMIQ.txt')).to_pandas()
df_mat_GSE196696.index = df_mat_GSE196696['C0'].tolist()
df_mat_GSE196696 = df_mat_GSE196696.drop(columns='C0')
df_meta_GSE196696 = \
    pd.read_csv(os.path.join(path_GSE196696_process, 'GSE196696_meta.tsv'), sep='\t', index_col=0)
df_meta_GSE196696.index = df_meta_GSE196696['ori_sample_id'].tolist()
df_mat_GSE196696.columns = df_meta_GSE196696.loc[df_mat_GSE196696.columns, 'sample_id'].tolist()

file_GSE196696_cpgs = os.path.join(path_GSE196696_process, 'GSE196696_cpgs_850k.txt')
file_GSE196696_mat = os.path.join(path_GSE196696_process, 'GSE196696_beta_GMQN_BMIQ_850k_impute.csv')
file_GSE196696_npy = os.path.join(path_GSE196696_process, 'Processed_all_850k.npy')
df_mat_GSE196696_imp = \
    save_mat_dict(df_mat_GSE196696, df_meta_GSE196696, file_GSE196696_cpgs, file_GSE196696_mat,
                  file_GSE196696_npy, ["sample_id", 'sex', "age"])


#%%
# GSE132203 850k
path_850k = '/home/zhangyu/mnt_path/Data/850k'
path_GSE132203_process = os.path.join(path_850k, 'GSE132203_process')
if not os.path.exists(path_GSE132203_process):
    os.mkdir(path_GSE132203_process)

file_series_matrix = os.path.join(path_850k, 'GSE132203_series_matrix.txt')
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

df_meta_GSE132203 = pd.concat(list_series, axis=1)
df_meta_GSE132203.index = df_meta_GSE132203['sample_id'].tolist()
df_meta_GSE132203 = df_meta_GSE132203.loc[:, ['sample_id', 'gender', 'age', 'age acceleration']]
df_meta_GSE132203.columns = ['sample_id', 'sex', 'age', 'age_acc']
df_meta_GSE132203['project_id'] = 'GSE132203'
df_meta_GSE132203['platform'] = '850k'
df_meta_GSE132203['tissue'] = 'whole blood'

df_mat_GSE132203 = read_single_csv(
    '/home/zhangyu/mnt_path/Data/850k/GSE132203_850k_450k_beta_GMQN_BMIQ.csv')
df_mat_GSE132203.columns = [one.split("_")[0] for one in df_mat_GSE132203.columns]

df_meta_GSE132203 = df_meta_GSE132203.loc[df_mat_GSE132203.columns, :]
df_meta_GSE132203.to_csv(os.path.join(path_GSE132203_process, 'GSE132203_meta.tsv'), sep='\t')

missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE132203), axis=1) / df_mat_GSE132203.shape[1],
              index=df_mat_GSE132203.index)
df_mat_GSE132203 = df_mat_GSE132203.loc[
                   missing_rates.loc[missing_rates <= 0.2].index, :]

file_GSE132203_cpgs = os.path.join(path_GSE132203_process, 'GSE132203_cpgs_450k.txt')
write_stringList_2File(file_GSE132203_cpgs, list(df_mat_GSE132203.index))

X_mat_GSE132203_imp, sample_GSE132203 = \
    impute_methy(df_mat_GSE132203, by_project=False)
df_mat_GSE132203_imp = pd.DataFrame(
    X_mat_GSE132203_imp, index=sample_GSE132203, columns=df_mat_GSE132203.index)
df_mat_GSE132203_imp.T.to_csv(
    os.path.join(path_GSE132203_process, 'GSE132203_beta_GMQN_BMIQ_450k_impute.csv'))

# save dict
data_to_np_dict(df_mat_GSE132203_imp, df_meta_GSE132203, df_meta_GSE132203.index,
                ["sample_id", 'sex', "age"],
                path_GSE132203_process, 'Processed_all_450k.npy')

# 850k
df_mat_GSE132203 = dt.fread(os.path.join(path_850k, 'GSE132203_beta_GMQN_BMIQ.csv')).to_pandas()
df_mat_GSE132203.index = df_mat_GSE132203['C0'].tolist()
df_mat_GSE132203 = df_mat_GSE132203.drop(columns='C0')
df_mat_GSE132203.columns = [one.split("_")[0] for one in df_mat_GSE132203.columns]
df_meta_GSE132203 = \
    pd.read_csv(os.path.join(path_GSE132203_process, 'GSE132203_meta.tsv'), sep='\t', index_col=0)

file_GSE132203_cpgs = os.path.join(path_GSE132203_process, 'GSE132203_cpgs_850k.txt')
file_GSE132203_mat = os.path.join(path_GSE132203_process, 'GSE132203_beta_GMQN_BMIQ_850k_impute.csv')
file_GSE132203_npy = os.path.join(path_GSE132203_process, 'Processed_all_850k.npy')
df_mat_GSE132203_imp = \
    save_mat_dict(df_mat_GSE132203, df_meta_GSE132203, file_GSE132203_cpgs, file_GSE132203_mat,
                  file_GSE132203_npy, ["sample_id", 'sex', "age"])


#%%
# GSE152026 850k
path_850k = '/home/zhangyu/mnt_path/Data/850k'
path_GSE152026_process = os.path.join(path_850k, 'GSE152026_process')
if not os.path.exists(path_GSE152026_process):
    os.mkdir(path_GSE152026_process)

file_series_matrix = os.path.join(path_850k, 'GSE152026_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[0] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE152026 = pd.concat(list_series, axis=1)
df_meta_GSE152026['project_id'] = 'GSE152026'
df_meta_GSE152026['platform'] = '850k'
df_meta_GSE152026['tissue'] = 'whole blood'
df_meta_GSE152026['disease'] = 'control'
df_meta_GSE152026.loc[df_meta_GSE152026['phenotype'] == 'Control', 'disease'] = 'first episode psychosis'
df_meta_GSE152026['sample_type'] = 'control'
df_meta_GSE152026.loc[df_meta_GSE152026['phenotype'] == 'Control', 'sample_type'] = 'disease tissue'
df_meta_GSE152026 = \
    df_meta_GSE152026.loc[:, ['sample_id', 'Sex', 'age', 'project_id', 'platform',
                              'tissue', 'disease', 'sample_type']]
df_meta_GSE152026.columns = ['sample_id', 'sex', 'age', 'project_id', 'platform',
                             'tissue', 'disease', 'sample_type']

df_mat_GSE152026 = read_single_csv(
    '/home/zhangyu/mnt_path/Data/850k/GSE152026_850k_450k_beta_GMQN_BMIQ_450k.csv')
df_mat_GSE152026.columns = df_meta_GSE152026.loc[df_mat_GSE152026.columns, 'sample_id'].tolist()

df_meta_GSE152026['ori_sample_id'] = df_meta_GSE152026.index
df_meta_GSE152026.index = df_meta_GSE152026['sample_id'].tolist()
# df_meta_GSE152026 = df_meta_GSE152026.loc[df_mat_GSE152026.columns, :]
df_meta_GSE152026.to_csv(os.path.join(path_GSE152026_process, 'GSE152026_meta.tsv'), sep='\t')

missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE152026), axis=1) / df_mat_GSE152026.shape[1],
              index=df_mat_GSE152026.index)
df_mat_GSE152026 = df_mat_GSE152026.loc[
                   missing_rates.loc[missing_rates <= 0.2].index, :]
df_mat_GSE152026 = \
    df_mat_GSE152026.loc[[one for one in df_mat_GSE152026.index if one in set_keep_cpgs], :]

file_GSE152026_cpgs = os.path.join(path_GSE152026_process, 'GSE152026_cpgs_450k.txt')
write_stringList_2File(file_GSE152026_cpgs, list(df_mat_GSE152026.index))

X_mat_GSE152026_imp, sample_GSE152026 = \
    impute_methy(df_mat_GSE152026, by_project=False)
df_mat_GSE152026_imp = pd.DataFrame(
    X_mat_GSE152026_imp, index=sample_GSE152026, columns=df_mat_GSE152026.index)
df_mat_GSE152026_imp.T.to_csv(
    os.path.join(path_GSE152026_process, 'GSE152026_beta_GMQN_BMIQ_450k_impute.csv'))

# save dict
data_to_np_dict(df_mat_GSE152026_imp, df_meta_GSE152026, df_meta_GSE152026.index,
                ["sample_id", 'sex', "age"],
                path_GSE152026_process, 'Processed_all_450k.npy')

# 850k
df_mat_GSE152026 = dt.fread(os.path.join(path_850k, 'GSE152026_beta_GMQN_BMIQ.csv')).to_pandas()
df_mat_GSE152026.index = df_mat_GSE152026['C0'].tolist()
df_mat_GSE152026 = df_mat_GSE152026.drop(columns='C0')
df_meta_GSE152026 = \
    pd.read_csv(os.path.join(path_GSE152026_process, 'GSE152026_meta.tsv'), sep='\t', index_col=0)
df_meta_GSE152026.index = df_meta_GSE152026['ori_sample_id'].tolist()
df_mat_GSE152026.columns = df_meta_GSE152026.loc[df_mat_GSE152026.columns, 'sample_id'].tolist()

file_GSE152026_cpgs = os.path.join(path_GSE152026_process, 'GSE152026_cpgs_850k.txt')
file_GSE152026_mat = os.path.join(path_GSE152026_process, 'GSE152026_beta_GMQN_BMIQ_850k_impute.csv')
file_GSE152026_npy = os.path.join(path_GSE152026_process, 'Processed_all_850k.npy')
df_meta_GSE152026.index = df_meta_GSE152026['sample_id'].tolist()
df_mat_GSE152026_imp = \
    save_mat_dict(df_mat_GSE152026, df_meta_GSE152026, file_GSE152026_cpgs, file_GSE152026_mat,
                  file_GSE152026_npy, ["sample_id", 'sex', "age"])

#%%
# AIRWAVE GSE147740 850k
path_AIRWAVE = '/home/zhangyu/mnt_path/Data/AIRWAVE'
path_GSE147740_process = os.path.join(path_AIRWAVE, 'GSE147740_process')
if not os.path.exists(path_GSE147740_process):
    os.mkdir(path_GSE147740_process)

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
df_meta_GSE147740.index = df_meta_GSE147740['sample_id'].tolist()
df_meta_GSE147740 = df_meta_GSE147740.loc[:, ['sample_id', 'age', 'Sex']]
df_meta_GSE147740.columns = ['sample_id', 'age', 'sex']
df_meta_GSE147740['project_id'] = 'GSE147740'
df_meta_GSE147740['platform'] = '850k'
df_meta_GSE147740['tissue'] = 'whole blood'

df_mat_GSE147740 = read_single_csv(
    os.path.join(path_AIRWAVE, 'GSE147740_850k_450k_beta_GMQN_BMIQ.csv'))
df_mat_GSE147740.columns = [one.split("_")[0] for one in df_mat_GSE147740.columns]

df_meta_GSE147740 = df_meta_GSE147740.loc[df_mat_GSE147740.columns, :]
df_meta_GSE147740.to_csv(os.path.join(path_GSE147740_process, 'GSE147740_meta.tsv'), sep='\t')

missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE147740), axis=1) / df_mat_GSE147740.shape[1],
              index=df_mat_GSE147740.index)
df_mat_GSE147740 = df_mat_GSE147740.loc[
                   missing_rates.loc[missing_rates <= 0.2].index, :]
df_mat_GSE147740 = \
    df_mat_GSE147740.loc[[one for one in df_mat_GSE147740.index if one in set_keep_cpgs], :]

file_GSE147740_cpgs = os.path.join(path_GSE147740_process, 'GSE147740_cpgs_450k.txt')
write_stringList_2File(file_GSE147740_cpgs, list(df_mat_GSE147740.index))

X_mat_GSE147740_imp, sample_GSE147740 = \
    impute_methy(df_mat_GSE147740, by_project=False)
df_mat_GSE147740_imp = pd.DataFrame(
    X_mat_GSE147740_imp, index=sample_GSE147740, columns=df_mat_GSE147740.index)
df_mat_GSE147740_imp.T.to_csv(
    os.path.join(path_GSE147740_process, 'GSE147740_beta_GMQN_BMIQ_450k_impute.csv'))

# save dict
data_to_np_dict(df_mat_GSE147740_imp, df_meta_GSE147740, df_meta_GSE147740.index,
                ["sample_id", 'sex', "age"],
                path_GSE147740_process, 'Processed_all_450k.npy')

# 850k
df_mat_GSE147740 = dt.fread(os.path.join(path_AIRWAVE, 'GSE147740_beta_GMQN_BMIQ.csv')).to_pandas()
df_mat_GSE147740.index = df_mat_GSE147740['C0'].tolist()
df_mat_GSE147740 = df_mat_GSE147740.drop(columns='C0')
df_mat_GSE147740.columns = [one.split("_")[0] for one in df_mat_GSE147740.columns]
df_meta_GSE147740 = \
    pd.read_csv(os.path.join(path_GSE147740_process, 'GSE147740_meta.tsv'), sep='\t', index_col=0)

file_GSE147740_cpgs = os.path.join(path_GSE147740_process, 'GSE147740_cpgs_850k.txt')
file_GSE147740_mat = os.path.join(path_GSE147740_process, 'GSE147740_beta_GMQN_BMIQ_850k_impute.csv')
file_GSE147740_npy = os.path.join(path_GSE147740_process, 'Processed_all_850k.npy')
df_meta_GSE147740.index = df_meta_GSE147740['sample_id'].tolist()
df_mat_GSE147740_imp = \
    save_mat_dict(df_mat_GSE147740, df_meta_GSE147740, file_GSE147740_cpgs, file_GSE147740_mat,
                  file_GSE147740_npy, ["sample_id", 'sex', "age"])


#%%
# GENOA GSE210255 850k
path_GENOA = '/home/zhangyu/mnt_path/Data/GENOA'
path_GSE210255_process = os.path.join(path_GENOA, 'GSE210255_process')
if not os.path.exists(path_GSE210255_process):
    os.mkdir(path_GSE210255_process)

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
df_meta_GSE210255['ori_sample_id'] = [one.replace('sample', '') for one in df_meta_GSE210255.index]
df_meta_GSE210255.index = df_meta_GSE210255['sample_id'].tolist()
df_meta_GSE210255 = df_meta_GSE210255.loc[:, ['sample_id', 'age(yrs)', 'gender', 'ori_sample_id']]
df_meta_GSE210255.columns = ['sample_id', 'age', 'sex', 'ori_sample_id']
df_meta_GSE210255['project_id'] = 'GSE210255'
df_meta_GSE210255['platform'] = '850k'
df_meta_GSE210255['tissue'] = 'leukocyte'

df_mat_GSE210255 = read_single_csv(
    os.path.join(path_GENOA, 'GSE210255_850k_450k_beta_GMQN_BMIQ.csv'))
df_mat_GSE210255.columns = [one.split("_")[0] for one in df_mat_GSE210255.columns]

df_meta_GSE210255 = df_meta_GSE210255.loc[df_mat_GSE210255.columns, :]
df_meta_GSE210255.to_csv(os.path.join(path_GSE210255_process, 'GSE210255_meta.tsv'), sep='\t')

missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE210255), axis=1) / df_mat_GSE210255.shape[1],
              index=df_mat_GSE210255.index)
df_mat_GSE210255 = df_mat_GSE210255.loc[
                   missing_rates.loc[missing_rates <= 0.2].index, :]
df_mat_GSE210255 = \
    df_mat_GSE210255.loc[[one for one in df_mat_GSE210255.index if one in set_keep_cpgs], :]

file_GSE210255_cpgs = os.path.join(path_GSE210255_process, 'GSE210255_cpgs_450k.txt')
write_stringList_2File(file_GSE210255_cpgs, list(df_mat_GSE210255.index))

X_mat_GSE210255_imp, sample_GSE210255 = \
    impute_methy(df_mat_GSE210255, by_project=False)
df_mat_GSE210255_imp = pd.DataFrame(
    X_mat_GSE210255_imp, index=sample_GSE210255, columns=df_mat_GSE210255.index)
df_mat_GSE210255_imp.T.to_csv(
    os.path.join(path_GSE210255_process, 'GSE210255_beta_GMQN_BMIQ_450k_impute.csv'))

# save dict
data_to_np_dict(df_mat_GSE210255_imp, df_meta_GSE210255, df_meta_GSE210255.index,
                ["sample_id", 'sex', "age"],
                path_GSE210255_process, 'Processed_all_450k.npy')

# 850k
df_mat_GSE210255 = dt.fread(os.path.join(path_GENOA, 'GSE210255_beta_GMQN_BMIQ.csv')).to_pandas()
df_mat_GSE210255.index = df_mat_GSE210255['C0'].tolist()
df_mat_GSE210255 = df_mat_GSE210255.drop(columns='C0')
df_mat_GSE210255.columns = [one.split("_")[0] for one in df_mat_GSE210255.columns]
df_meta_GSE210255 = \
    pd.read_csv(os.path.join(path_GSE210255_process, 'GSE210255_meta.tsv'), sep='\t', index_col=0)

file_GSE210255_cpgs = os.path.join(path_GSE210255_process, 'GSE210255_cpgs_850k.txt')
file_GSE210255_mat = os.path.join(path_GSE210255_process, 'GSE210255_beta_GMQN_BMIQ_850k_impute.csv')
file_GSE210255_npy = os.path.join(path_GSE210255_process, 'Processed_all_850k.npy')
df_meta_GSE210255.index = df_meta_GSE210255['sample_id'].tolist()
df_mat_GSE210255_imp = \
    save_mat_dict(df_mat_GSE210255, df_meta_GSE210255, file_GSE210255_cpgs, file_GSE210255_mat,
                  file_GSE210255_npy, ["sample_id", 'sex', "age"])


#%%
# GSE207927 850k
path_850k = '/home/zhangyu/mnt_path/Data/850k'
path_GSE207927_process = os.path.join(path_850k, 'GSE207927_process')
if not os.path.exists(path_GSE207927_process):
    os.mkdir(path_GSE207927_process)

file_series_matrix = os.path.join(path_850k, 'GSE207927_series_matrix.txt')
list_dict = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[1][1:-1] for one in list_line[1:]]
            for one_idx in range(len(list_index)):
                list_dict.append(dict(ori_sample_id=list_index[one_idx]))
        elif list_line[0] == '!Sample_geo_accession':
            list_gsm = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_index)):
                list_dict[one_idx].update(dict(sample_id=list_gsm[one_idx]))
        elif list_line[0] == '!Sample_characteristics_ch1':
            list_cha = [one.strip("\"") for one in list_line[1:]]
            for one_idx in range(len(list_index)):
                col_name = list_cha[one_idx].strip("\"").split(': ')[0]
                if col_name == '':
                    continue
                col_value = list_cha[one_idx].strip("\"").split(': ')[1]
                list_dict[one_idx].update({col_name: col_value})

df_meta_GSE207927 = pd.DataFrame(list_dict)
df_meta_GSE207927['project_id'] = 'GSE207927'
df_meta_GSE207927['platform'] = '850k'
df_meta_GSE207927.index = df_meta_GSE207927['ori_sample_id'].tolist()

df_mat_GSE207927 = dt.fread(
    os.path.join(path_850k, 'GSE207927_Processed_Beta_Matrix.txt'), sep='\t').to_pandas()
df_mat_GSE207927.index = df_mat_GSE207927['index'].tolist()
df_mat_GSE207927 = df_mat_GSE207927.drop(columns='index')
df_mat_GSE207927.columns = df_meta_GSE207927.loc[df_mat_GSE207927.columns, 'sample_id'].tolist()

df_meta_GSE207927.index = df_meta_GSE207927['sample_id'].tolist()
df_meta_GSE207927.columns = [one.replace('Sex', 'sex') for one in df_meta_GSE207927.columns]
df_meta_GSE207927.to_csv(os.path.join(path_GSE207927_process, 'GSE207927_meta.tsv'), sep='\t')

file_GSE207927_cpgs = os.path.join(path_GSE207927_process, 'GSE207927_cpgs_850k.txt')
file_GSE207927_mat = os.path.join(path_GSE207927_process, 'GSE207927_beta_GMQN_BMIQ_850k_impute.csv')
file_GSE207927_npy = os.path.join(path_GSE207927_process, 'Processed_all_850k.npy')
df_meta_GSE207927.index = df_meta_GSE207927['sample_id'].tolist()
df_mat_GSE207927_imp = \
    save_mat_dict(df_mat_GSE207927, df_meta_GSE207927, file_GSE207927_cpgs, file_GSE207927_mat,
                  file_GSE207927_npy, ["sample_id", 'sex', "age"])



#%%
# merge 850k data
path_GSE152026_process = '/home/zhangyu/mnt_path/Data/850k/GSE152026_process'
GSE152026_850k = np.load(os.path.join(path_GSE152026_process, 'Processed_all_450k.npy'),
                         allow_pickle = True).item()
df_meta_GSE152026 = pd.read_csv(
    os.path.join(path_GSE152026_process, 'GSE152026_meta.tsv'), sep='\t', index_col=0)
cpg_list_GSE152026 = read_stringList_FromFile(
    os.path.join(path_GSE152026_process, 'GSE152026_cpgs_450k.txt'))

path_GSE132203_process = '/home/zhangyu/mnt_path/Data/850k/GSE132203_process'
GSE132203_850k = np.load(os.path.join(path_GSE132203_process, 'Processed_all_450k.npy'),
                         allow_pickle = True).item()
df_meta_GSE132203 = pd.read_csv(
    os.path.join(path_GSE132203_process, 'GSE132203_meta.tsv'), sep='\t', index_col=0)
cpg_list_GSE132203 = read_stringList_FromFile(
    os.path.join(path_GSE132203_process, 'GSE132203_cpgs_450k.txt'))

path_GSE147740_process = '/home/zhangyu/mnt_path/Data/AIRWAVE/GSE147740_process'
GSE147740_850k = np.load(os.path.join(path_GSE147740_process, 'Processed_all_450k.npy'),
                         allow_pickle = True).item()
df_meta_GSE147740 = pd.read_csv(
    os.path.join(path_GSE147740_process, 'GSE147740_meta.tsv'), sep='\t', index_col=0)
cpg_list_GSE147740 = read_stringList_FromFile(
    os.path.join(path_GSE147740_process, 'GSE147740_cpgs_450k.txt'))

path_GSE196696_process = '/home/zhangyu/mnt_path/Data/850k/GSE196696_process'
GSE196696_850k = np.load(os.path.join(path_GSE196696_process, 'Processed_all_450k.npy'),
                         allow_pickle = True).item()
df_meta_GSE196696 = pd.read_csv(
    os.path.join(path_GSE196696_process, 'GSE196696_meta.tsv'), sep='\t', index_col=0)
cpg_list_GSE196696 = read_stringList_FromFile(
    os.path.join(path_GSE196696_process, 'GSE196696_cpgs_450k.txt'))

set_cpg_merge = set(cpg_list_GSE152026).intersection(set(cpg_list_GSE132203)).intersection(
    set(cpg_list_GSE147740))
# set_cpg_merge = set(cpg_list_GSE152026).intersection(set(cpg_list_GSE132203)).intersection(
#     set(cpg_list_GSE196696))
cpg_list_merge = [one for one in cpg_list_GSE152026 if one in set_cpg_merge]

GSE132203_np_dict = order_cpg_to_ref(
    cpg_list_GSE132203, cpg_list_merge, GSE132203_850k, df_meta_GSE132203['sample_id'].tolist())
list_sample_GSE152026 = \
    [one for one in df_meta_GSE152026['sample_id'].tolist() if one in GSE152026_850k.keys()]
GSE152026_np_dict = order_cpg_to_ref(
    cpg_list_GSE152026, cpg_list_merge, GSE152026_850k, list_sample_GSE152026)
list_sample_GSE147740 = \
    [one for one in df_meta_GSE147740['sample_id'].tolist() if one in GSE147740_850k.keys()]
GSE147740_np_dict = order_cpg_to_ref(
    cpg_list_GSE147740, cpg_list_merge, GSE147740_850k, list_sample_GSE147740)
# GSE196696_np_dict = order_cpg_to_ref(
#     cpg_list_GSE196696, cpg_list_merge, GSE196696_850k, df_meta_GSE196696['sample_id'].tolist())
dict_merge = GSE132203_np_dict
dict_merge.update(GSE152026_np_dict)
dict_merge.update(GSE147740_np_dict)

# df_meta_merge = pd.concat([df_meta_GSE132203, df_meta_GSE152026, df_meta_GSE196696])
df_meta_merge = pd.concat([df_meta_GSE132203, df_meta_GSE152026, df_meta_GSE147740])
df_meta_merge['tissue'] = 'whole blood'
df_meta_merge['sample_type'] = 'control'
df_meta_merge['disease'] = 'control'
df_meta_merge['bmi'] = -1
df_meta_merge = df_meta_merge.loc[df_meta_merge['sample_id'].apply(lambda x: x in dict_merge.keys()), :]

path_merge = '/home/zhangyu/mnt_path/Data/850k/merge'
write_stringList_2File(os.path.join(path_merge, 'cpgs_850k.txt'), cpg_list_merge)
np.save(os.path.join(path_merge, 'Processed_850k.npy'), dict_merge)
df_meta_merge.to_csv(os.path.join(path_merge, 'meta_850k.tsv'), sep='\t')

# %%
# merge 850k matrix
processData_path="/home/zhangyu/bioage/tyh/850k_model/850model_creat/process_data/"
GSE132203_np_dict = np.load(os.path.join(processData_path, "Processed_GSE132203_850k_rm10.npy"), allow_pickle = True).item()
GSE152026_np_dict = np.load(os.path.join(processData_path, "Processed_GSE152026_850k_rm10.npy"), allow_pickle = True).item()
GSE147740_np_dict = np.load(os.path.join(processData_path, "Processed_GSE147740_850k_rm10.npy"), allow_pickle = True).item()

path_GSE152026_process = '/home/zhangyu/mnt_path/Data/850k/GSE152026_process'
df_meta_GSE152026 = pd.read_csv(
    os.path.join(path_GSE152026_process, 'GSE152026_meta.tsv'), sep='\t', index_col=0)
path_GSE132203_process = '/home/zhangyu/mnt_path/Data/850k/GSE132203_process'
df_meta_GSE132203 = pd.read_csv(
    os.path.join(path_GSE132203_process, 'GSE132203_meta.tsv'), sep='\t', index_col=0)
path_GSE147740_process = '/home/zhangyu/mnt_path/Data/AIRWAVE/GSE147740_process'
df_meta_GSE147740 = pd.read_csv(
    os.path.join(path_GSE147740_process, 'GSE147740_meta.tsv'), sep='\t', index_col=0)

GSE132203_cpg_list = read_stringList_FromFile(os.path.join(processData_path, "GSE132203_cpgs_850k_rm10"))
GSE152026_cpg_list = read_stringList_FromFile(os.path.join(processData_path, "GSE152026_cpgs_850k_rm10"))
GSE147740_cpg_list = read_stringList_FromFile(os.path.join(processData_path, "GSE147740_cpgs_850k_rm10"))
set_cpg_merge_2 = set(GSE132203_cpg_list).intersection(set(GSE152026_cpg_list)).intersection(
    set(GSE147740_cpg_list))
cpg_list_merge_2 = [one for one in GSE132203_cpg_list if one in set_cpg_merge_2]

data_dict_GSE132203 = order_cpg_to_ref(
    GSE132203_cpg_list, cpg_list_merge_2, GSE132203_np_dict, df_meta_GSE132203['sample_id'].tolist())
list_sample_GSE147740 = \
    [one for one in df_meta_GSE147740['sample_id'].tolist() if one in GSE147740_np_dict.keys()]
data_dict_GSE147740 = order_cpg_to_ref(
    GSE147740_cpg_list, cpg_list_merge_2, GSE147740_np_dict, list_sample_GSE147740)
list_sample_GSE152026 = \
    [one for one in df_meta_GSE152026['sample_id'].tolist() if one in GSE152026_np_dict.keys()]
data_dict_GSE152026 = order_cpg_to_ref(
    GSE152026_cpg_list, cpg_list_merge_2, GSE152026_np_dict, list_sample_GSE152026)
dict_merge_2 = data_dict_GSE132203
dict_merge_2.update(data_dict_GSE147740)
dict_merge_2.update(data_dict_GSE152026)
path_merge = '/home/zhangyu/mnt_path/Data/850k/merge'
np.save(os.path.join(path_merge, 'Processed_850k_geo_mat.npy'), dict_merge_2)
write_stringList_2File(os.path.join(path_merge, 'cpgs_850k_geo_mat.txt'), cpg_list_merge_2)

