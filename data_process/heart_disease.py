# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: heart_disease.py
# @time: 2023/10/12 21:43

import numpy as np
import pandas as pd
import datatable as dt
import pickle
import os
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File, FileUtils
from utils.common_utils import data_to_np_dict, impute_methy, read_single_csv


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
cpg_list_mrs_pretrain = read_stringList_FromFile(
    '/home/zhangyu/mnt_path/Data/EWAS_process/disease/cpgs_blood_450k_850k_stroke_pretrain_1')
set_keep_cpgs = set(cpg_list_mrs_pretrain)


# %%
# atherosclerosis
path_HD = '/home/zhangyu/mnt_path/Data/heart_disease'
path_HD_process = os.path.join(path_HD, 'GSE220622_process')
FileUtils.makedir(path_HD_process)
# df_mat_as = pd.read_csv(os.path.join(path_HD, 'GSE220622_processed_M-value.tsv'), sep='\t', index_col=0)
# df_mat_as = df_mat_as.loc[:, [one for one in df_mat_as.columns if one[:9] != 'Detection']]
# keep_cpgs = np.sum(pd.isna(df_mat_as), axis=1).loc[np.sum(pd.isna(df_mat_as), axis=1) < 10].index
# df_mat_as = df_mat_as.loc[keep_cpgs, :]

file_series_matrix = os.path.join(path_HD, 'GSE220622_series_matrix.txt')
list_series = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[-1].strip("[").strip("]") for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one[1:11] for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            if list_line[1][1:4] == 'age':
                list_series.append(pd.Series(
                    [int(one.strip("\"").split(': ')[1]) for one in list_line[1:]],
                    index=list_index, name='age'))
            elif list_line[1][1:7] == 'gender':
                list_series.append(pd.Series(
                    [one.strip("\"").split(': ')[1] if one.strip("\"").split(': ')[1] != 'NA' else np.nan for one in list_line[1:]],
                    index=list_index, name='sex'))
            elif list_line[1][1:11] == 'pesa score':
                list_series.append(pd.Series(
                    [one.strip("\"").split(': ')[1] if one.strip("\"").split(': ')[1] != 'NA' else np.nan for one in list_line[1:]],
                    index=list_index, name='pesa_score'))

df_meta = pd.concat(list_series, axis=1)
df_meta.index = df_meta['sample_id'].tolist()

df_mat_as = dt.fread(os.path.join(path_HD, 'GSE220622_850k_450k_beta_GMQN_BMIQ.txt')).to_pandas()
df_mat_as.index = df_mat_as['C0'].tolist()
df_mat_as = df_mat_as.drop(columns='C0')
df_mat_as.columns = [one.split('_')[0] for one in df_mat_as.columns]
# missing_rates = \
#     pd.Series(np.sum(np.isnan(df_mat_as), axis=1) / df_mat_as.shape[1],
#               index=df_mat_as.index)
# df_mat_as = df_mat_as.loc[missing_rates.loc[missing_rates <= 0.2].index, :]
# df_mat_as = \
#     df_mat_as.loc[[one for one in df_mat_as.index if one in set_keep_cpgs], :]

imp_mat, imp_samples = impute_methy(df_mat_as, by_project=False)
df_mat_as = pd.DataFrame(imp_mat, index=imp_samples, columns=df_mat_as.index)
df_meta = df_meta.loc[imp_samples, :]
df_meta.to_csv(os.path.join(path_HD_process, 'GSE220622_meta.txt'))

df_mat_imp_T = df_mat_as.T
df_mat_imp_T.insert(loc=0, column='CpG', value=df_mat_as.columns)
dt_mat_imp = dt.Frame(df_mat_imp_T)
dt_mat_imp.to_csv(os.path.join(path_HD_process, 'GSE220622_850k_450k_beta_GMQN_BMIQ_impute.csv'))

# transform into dict
# all CpGs
file_all_HD_cpgs = os.path.join(path_HD_process, 'GSE220622_cpgs_GMQN_450k.txt')
write_stringList_2File(file_all_HD_cpgs, list(df_mat_as.columns))
# df_mat_all = df_mat_as.loc[:, df_meta.index]
df_mat_all = df_mat_as
# df_mat_all_beta = (1.0001*np.exp(df_mat_all)-0.0001)/(1+np.exp(df_mat_all))
df_mat_all_beta = df_mat_all
df_mat_all_beta.index = df_meta['sample_id'].tolist()
df_meta.index = df_meta['sample_id'].tolist()
save_dict = {}
X_np = df_mat_all_beta
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

np.save(os.path.join(path_HD_process, 'Processed_all_GMQN_450k.npy'), save_dict)



#%%
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

# %%
# coronary artery ectasia
# GSE87016
path_HD = '/home/zhangyu/mnt_path/Data/heart_disease'
path_GSE87016_process = os.path.join(path_HD, 'GSE87016_process')
if not os.path.exists(path_GSE87016_process):
    os.mkdir(path_GSE87016_process)

file_series_matrix = os.path.join(path_HD, 'GSE87016_series_matrix.txt')
list_dict = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        list_line = line.strip().split('\t')
        if list_line[0] == '!Sample_title':
            list_index = ['_'.join(one.strip("\"").split(' ')[3:]) for one in list_line[1:]]
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

df_meta_GSE87016 = pd.DataFrame(list_dict)
df_meta_GSE87016['project_id'] = 'GSE87016'
df_meta_GSE87016['platform'] = '450k'
df_meta_GSE87016.index = df_meta_GSE87016['ori_sample_id'].tolist()
df_meta_GSE87016.loc[
    df_meta_GSE87016['disease state'] == 'healthy control', 'disease state'] = 'control'
df_meta_GSE87016['sample_type'] = df_meta_GSE87016['disease state']
df_meta_GSE87016.loc[
    df_meta_GSE87016['disease state'] != 'control', 'sample_type'] = 'disease tissue'
df_meta_GSE87016.columns = ['ori_sample_id', 'sample_id', 'disease', 'tissue', 'sex',
                            'creatnine', 'smoking', 'hyperlipidemia', 'hypertension', 'dm',
                            'project_id', 'platform', 'sample_type']

df_mat_GSE87016 = dt.fread(os.path.join(path_HD, 'GSE87016_beta_GMQN_BMIQ.csv')).to_pandas()
df_mat_GSE87016.index = df_mat_GSE87016['C0'].tolist()
df_mat_GSE87016 = df_mat_GSE87016.drop(columns='C0')
df_mat_GSE87016.columns = df_meta_GSE87016.loc[df_mat_GSE87016.columns, 'sample_id'].tolist()

df_meta_GSE87016.index = df_meta_GSE87016['sample_id'].tolist()
df_meta_GSE87016.to_csv(os.path.join(path_GSE87016_process, 'GSE87016_meta.tsv'), sep='\t')
df_meta_GSE87016['age'] = -10

file_GSE87016_cpgs = os.path.join(path_GSE87016_process, 'GSE87016_cpgs_450k.txt')
file_GSE87016_mat = os.path.join(path_GSE87016_process, 'GSE87016_beta_GMQN_BMIQ_450k_impute.csv')
file_GSE87016_npy = os.path.join(path_GSE87016_process, 'Processed_all_450k.npy')
df_mat_GSE87016_imp = \
    save_mat_dict(df_mat_GSE87016, df_meta_GSE87016, file_GSE87016_cpgs, file_GSE87016_mat,
                  file_GSE87016_npy, ["sample_id", 'sex', "age"])
