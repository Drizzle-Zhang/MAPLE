# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: diabetes.py
# @time: 2023/12/6 10:43


import numpy as np
import pandas as pd
import datatable as dt
import pickle
import os
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File
from utils.common_utils import data_augmentation, impute_methy, data_to_np_dict


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

# Keep CpGs
cpg_list_450k_disease = read_stringList_FromFile(
    "/home/zhangyu/mnt_path/Data/EWAS_process/disease/450k_rm10/cpgs_list.txt")
cpg_list_450k_age = read_stringList_FromFile(
    "/home/zhangyu/mnt_path/Data/EWAS_process/age/450k_rm10/cpgs_list.txt")
cpg_list_850k = read_stringList_FromFile(
    "/home/zhangyu/mnt_path/Data/jizhuan/process/cpgs_jizhuan_850k_all.txt")
set_keep_cpgs = set(cpg_list_450k_disease).intersection(set(cpg_list_850k)).intersection(
    set(cpg_list_450k_age))

#%%
# prediabetes GSE199700
path_prediabetes = '/home/zhangyu/mnt_path/Data/prediabetes'
path_GSE199700_process = os.path.join(path_prediabetes, 'GSE199700_process')
if not os.path.exists(path_GSE199700_process):
    os.mkdir(path_GSE199700_process)

file_series_matrix = os.path.join(path_prediabetes, 'GSE199700_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[-1] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one.strip("\"") for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] if one.strip("\"") != '' else '' for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE199700 = pd.concat(list_series, axis=1)
df_meta_GSE199700.index = df_meta_GSE199700['sample_id'].tolist()
col_1 = ['age', 'sbp (mmhg)', 'dbp (mmhg)', 'stature (m)']
col_2 = ['disease state', 'time', 'tissue', 'cell type']
for i in range(len(col_1)):
    df_meta_GSE199700.loc[df_meta_GSE199700['al'] == '', col_2[i]] = df_meta_GSE199700.loc[df_meta_GSE199700['al'] == '', col_1[i]]
    df_meta_GSE199700.loc[df_meta_GSE199700['al'] == '', col_1[i]] = ''
df_meta_GSE199700['sex'] = 'F'
df_meta_GSE199700.to_csv(os.path.join(path_GSE199700_process, 'GSE199700_meta.tsv'), sep='\t')

# 450k
df_mat_GSE199700_450k = pd.read_csv(
    os.path.join(path_prediabetes, 'GSE199700_850k_450k_beta_GMQN_BMIQ.csv'), index_col=0)
df_mat_GSE199700_450k.columns = [one.split('_')[0] for one in df_mat_GSE199700_450k.columns]
df_meta_GSE199700 = \
    pd.read_csv(os.path.join(path_GSE199700_process, 'GSE199700_meta.tsv'), sep='\t', index_col=0)

file_GSE199700_cpgs = os.path.join(path_GSE199700_process, 'GSE199700_cpgs_850k.txt')
file_GSE199700_mat = os.path.join(path_GSE199700_process, 'GSE199700_beta_GMQN_BMIQ_850k_impute.csv')
file_GSE199700_npy = os.path.join(path_GSE199700_process, 'Processed_all_850k.npy')
df_meta_GSE199700.index = df_meta_GSE199700['sample_id'].tolist()
df_meta_GSE199700 = df_meta_GSE199700.loc[df_meta_GSE199700["sample_id"] != 'GSM5981925', :]
df_meta_GSE199700 = df_meta_GSE199700.dropna(subset='age')
df_mat_GSE199700_imp = \
    save_mat_dict(df_mat_GSE199700_450k, df_meta_GSE199700, file_GSE199700_cpgs, file_GSE199700_mat,
                  file_GSE199700_npy, ["sample_id", 'sex', "age"])

# 850k
df_mat_GSE199700_850k = pd.read_csv(
    os.path.join(path_prediabetes, 'GSE199700_850k_beta_GMQN_BMIQ.csv'), index_col=0)
df_mat_GSE199700_850k.columns = [one.split('_')[0] for one in df_mat_GSE199700_850k.columns]
missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE199700_850k), axis=1) / df_mat_GSE199700_850k.shape[1],
              index=df_mat_GSE199700_850k.index)
df_mat_GSE199700_850k = df_mat_GSE199700_850k.loc[
                        missing_rates.loc[missing_rates <= 0.2].index, :]
X_mat_GSE199700_850k_imp, sample_GSE199700_850k = \
    impute_methy(df_mat_GSE199700_850k, by_project=False)
df_mat_GSE199700_850k_imp = pd.DataFrame(
    X_mat_GSE199700_850k_imp,
    index=sample_GSE199700_850k, columns=df_mat_GSE199700_850k.index)
df_mat_GSE199700_850k_imp.T.to_csv(os.path.join(path_GSE199700_process, 'GSE199700_850k_impute.csv'))

#%%
# T1D GSE76169
path_t1d = '/home/zhangyu/mnt_path/Data/T1D'
path_GSE76169_process = os.path.join(path_t1d, 'GSE76169_process')
if not os.path.exists(path_GSE76169_process):
    os.mkdir(path_GSE76169_process)

file_series_matrix = os.path.join(path_t1d, 'GSE76169_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[-1] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one.strip("\"") for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] if one.strip("\"") != '' else '' for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE76169 = pd.concat(list_series, axis=1)
df_meta_GSE76169.index = df_meta_GSE76169['sample_id'].tolist()
df_meta_GSE76169 = df_meta_GSE76169.loc[:, ['sample_id', 'age', 'gender', 'sample group']]
df_meta_GSE76169.columns = ['sample_id', 'age', 'sex', 'complication']
df_meta_GSE76169.loc[
    df_meta_GSE76169['complication'] == 'Case', 'complication'] = 'retinopathy or albuminuria'
df_meta_GSE76169['tissue'] = 'whole blood'
df_meta_GSE76169.to_csv(os.path.join(path_GSE76169_process, 'GSE76169_meta.tsv'), sep='\t')

# 450k
df_mat_GSE76169_450k = pd.read_csv(
    os.path.join(path_t1d, 'GSE76169_850k_450k_beta_GMQN_BMIQ.csv'), index_col=0)
df_mat_GSE76169_450k.columns = [one.split('_')[0] for one in df_mat_GSE76169_450k.columns]
df_mat_GSE76169_450k = \
    df_mat_GSE76169_450k.loc[
    [one for one in df_mat_GSE76169_450k.index if one in set_keep_cpgs], :]
missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE76169_450k), axis=1) / df_mat_GSE76169_450k.shape[1],
              index=df_mat_GSE76169_450k.index)
df_mat_GSE76169_450k = df_mat_GSE76169_450k.loc[
                       missing_rates.loc[missing_rates <= 0.2].index, :]
X_mat_GSE76169_450k_imp, sample_GSE76169_450k = \
    impute_methy(df_mat_GSE76169_450k, by_project=False)
df_mat_GSE76169_450k_imp = pd.DataFrame(
    X_mat_GSE76169_450k_imp,
    index=sample_GSE76169_450k, columns=df_mat_GSE76169_450k.index)
df_mat_GSE76169_450k_imp.T.to_csv(
    os.path.join(path_GSE76169_process, 'GSE76169_beta_GMQN_BMIQ_450k_impute.csv'))

# 850k
# df_mat_GSE76169_850k = pd.read_csv(
#     os.path.join(path_t1d, 'GSE76169_850k_beta_GMQN_BMIQ.csv'), index_col=0)
# df_mat_GSE76169_850k.columns = [one.split('_')[0] for one in df_mat_GSE76169_850k.columns]
# missing_rates = \
#     pd.Series(np.sum(np.isnan(df_mat_GSE76169_850k), axis=1) / df_mat_GSE76169_850k.shape[1],
#               index=df_mat_GSE76169_850k.index)
# df_mat_GSE76169_850k = df_mat_GSE76169_850k.loc[
#                         missing_rates.loc[missing_rates <= 0.2].index, :]
# X_mat_GSE76169_850k_imp, sample_GSE76169_850k = \
#     impute_methy(df_mat_GSE76169_850k, by_project=False)
# df_mat_GSE76169_850k_imp = pd.DataFrame(
#     X_mat_GSE76169_850k_imp,
#     index=sample_GSE76169_850k, columns=df_mat_GSE76169_850k.index)
# df_mat_GSE76169_850k_imp.T.to_csv(os.path.join(path_GSE76169_process, 'GSE76169_850k_impute.csv'))

#%%
# T1D GSE76170
path_t1d = '/home/zhangyu/mnt_path/Data/T1D'
path_GSE76170_process = os.path.join(path_t1d, 'GSE76170_process')
if not os.path.exists(path_GSE76170_process):
    os.mkdir(path_GSE76170_process)

file_series_matrix = os.path.join(path_t1d, 'GSE76170_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[-1] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one.strip("\"") for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] if one.strip("\"") != '' else '' for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE76170 = pd.concat(list_series, axis=1)
df_meta_GSE76170.index = df_meta_GSE76170['sample_id'].tolist()
df_meta_GSE76170 = df_meta_GSE76170.loc[:, ['sample_id', 'age', 'gender', 'sample group']]
df_meta_GSE76170.columns = ['sample_id', 'age', 'sex', 'complication']
df_meta_GSE76170['tissue'] = 'CD14+ monocyte'
df_meta_GSE76170.loc[
    df_meta_GSE76170['complication'] == 'Case', 'complication'] = 'retinopathy or albuminuria'
df_meta_GSE76170.to_csv(os.path.join(path_GSE76170_process, 'GSE76170_meta.tsv'), sep='\t')

# 450k
df_mat_GSE76170_450k = pd.read_csv(
    os.path.join(path_t1d, 'GSE76170_850k_450k_beta_GMQN_BMIQ.csv'), index_col=0)
df_mat_GSE76170_450k.columns = [one.split('_')[0] for one in df_mat_GSE76170_450k.columns]
df_mat_GSE76170_450k = \
    df_mat_GSE76170_450k.loc[
    [one for one in df_mat_GSE76170_450k.index if one in set_keep_cpgs], :]
missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE76170_450k), axis=1) / df_mat_GSE76170_450k.shape[1],
              index=df_mat_GSE76170_450k.index)
df_mat_GSE76170_450k = df_mat_GSE76170_450k.loc[
                       missing_rates.loc[missing_rates <= 0.2].index, :]
X_mat_GSE76170_450k_imp, sample_GSE76170_450k = \
    impute_methy(df_mat_GSE76170_450k, by_project=False)
df_mat_GSE76170_450k_imp = pd.DataFrame(
    X_mat_GSE76170_450k_imp,
    index=sample_GSE76170_450k, columns=df_mat_GSE76170_450k.index)
df_mat_GSE76170_450k_imp.T.to_csv(
    os.path.join(path_GSE76170_process, 'GSE76170_beta_GMQN_BMIQ_450k_impute.csv'))

# 850k
# df_mat_GSE76170_850k = pd.read_csv(
#     os.path.join(path_t1d, 'GSE76170_850k_beta_GMQN_BMIQ.csv'), index_col=0)
# df_mat_GSE76170_850k.columns = [one.split('_')[0] for one in df_mat_GSE76170_850k.columns]
# missing_rates = \
#     pd.Series(np.sum(np.isnan(df_mat_GSE76170_850k), axis=1) / df_mat_GSE76170_850k.shape[1],
#               index=df_mat_GSE76170_850k.index)
# df_mat_GSE76170_850k = df_mat_GSE76170_850k.loc[
#                         missing_rates.loc[missing_rates <= 0.2].index, :]
# X_mat_GSE76170_850k_imp, sample_GSE76170_850k = \
#     impute_methy(df_mat_GSE76170_850k, by_project=False)
# df_mat_GSE76170_850k_imp = pd.DataFrame(
#     X_mat_GSE76170_850k_imp,
#     index=sample_GSE76170_850k, columns=df_mat_GSE76170_850k.index)
# df_mat_GSE76170_850k_imp.T.to_csv(os.path.join(path_GSE76170_process, 'GSE76170_850k_impute.csv'))

#%%
# T2D GSE197881
path_t2d = '/home/zhangyu/mnt_path/Data/T2D'
path_GSE197881_process = os.path.join(path_t2d, 'GSE197881_process')
if not os.path.exists(path_GSE197881_process):
    os.mkdir(path_GSE197881_process)

file_series_matrix = os.path.join(path_t2d, 'GSE197881_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[-1] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one.strip("\"") for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] if one.strip("\"") != '' else '' for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta_GSE197881 = pd.concat(list_series, axis=1)
df_meta_GSE197881.index = df_meta_GSE197881['sample_id'].tolist()
df_meta_GSE197881 = df_meta_GSE197881.loc[:, ['sample_id', 'gender', 'disease status']]
df_meta_GSE197881.loc[
    df_meta_GSE197881['disease status'] == 'Diabetic', 'disease status'] = 'type 2 diabetes'
df_meta_GSE197881.loc[
    df_meta_GSE197881['disease status'] == 'Non-diabetic', 'disease status'] = 'control'
df_meta_GSE197881.columns = ['sample_id', 'sex', 'disease']
df_meta_GSE197881['sample_type'] = 'control'
df_meta_GSE197881.loc[
    df_meta_GSE197881['disease'] == 'type 2 diabetes', 'sample_type'] = 'disease tissue'
df_meta_GSE197881['tissue'] = 'CD14+ monocyte'
df_meta_GSE197881.to_csv(os.path.join(path_GSE197881_process, 'GSE197881_meta.tsv'), sep='\t')

# 450k
df_mat_GSE197881_450k = pd.read_csv(
    os.path.join(path_t2d, 'GSE197881_850k_450k_beta_GMQN_BMIQ.csv'), index_col=0)
df_mat_GSE197881_450k.columns = [one.split('_')[0] for one in df_mat_GSE197881_450k.columns]
df_mat_GSE197881_450k = \
    df_mat_GSE197881_450k.loc[
    [one for one in df_mat_GSE197881_450k.index if one in set_keep_cpgs], :]
missing_rates = \
    pd.Series(np.sum(np.isnan(df_mat_GSE197881_450k), axis=1) / df_mat_GSE197881_450k.shape[1],
              index=df_mat_GSE197881_450k.index)
df_mat_GSE197881_450k = df_mat_GSE197881_450k.loc[
                       missing_rates.loc[missing_rates <= 0.2].index, :]
X_mat_GSE197881_450k_imp, sample_GSE197881_450k = \
    impute_methy(df_mat_GSE197881_450k, by_project=False)
df_mat_GSE197881_450k_imp = pd.DataFrame(
    X_mat_GSE197881_450k_imp,
    index=sample_GSE197881_450k, columns=df_mat_GSE197881_450k.index)
df_mat_GSE197881_450k_imp.T.to_csv(
    os.path.join(path_GSE197881_process, 'GSE197881_beta_GMQN_BMIQ_450k_impute.csv'))

