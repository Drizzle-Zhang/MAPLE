# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: heart_disease.py
# @time: 2023/10/12 21:43

import numpy as np
import pandas as pd
import pickle
import os
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File
from utils.common_utils import data_augmentation, impute_methy


#%%
# stroke
# GSE203399 850k
path_stroke = '/home/zhangyu/mnt_path/Data/stroke'
df_mat_GSE203399_850k = pd.read_csv(
    os.path.join(path_stroke, 'GSE203399_betas_normalized_pval_detection_replication.txt'),
    sep='\t', index_col=0)
df_mat_GSE203399_850k = df_mat_GSE203399_850k.loc[:,
                        [one for one in df_mat_GSE203399_850k.columns if one[:9] != 'Detection']]

# GSE203399
path_GSE203399_process = os.path.join(path_stroke, 'GSE203399_process')
if not os.path.exists(path_GSE203399_process):
    os.mkdir(path_GSE203399_process)

file_series_matrix = os.path.join(path_stroke, 'GSE203399-GPL29753_series_matrix.txt')
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
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta = pd.concat(list_series, axis=1)
df_meta.index = df_meta['sample_id'].tolist()
df_mat_GSE203399_850k.columns = df_meta['sample_id'].tolist()
df_meta.to_csv(os.path.join(path_GSE203399_process, 'GSE203399_850k_meta.txt'))

# all index
new_train_index = df_meta['sample_id'].tolist()
with open(os.path.join(path_GSE203399_process, "all_index.pkl"), 'wb') as f:
    pickle.dump(new_train_index, f)
    pickle.dump([], f)

# transform into dict
# all CpGs
file_GSE203399_850k_cpgs = os.path.join(path_GSE203399_process, 'GSE203399_850k_cpgs.txt')
write_stringList_2File(file_GSE203399_850k_cpgs, list(df_mat_GSE203399_850k.index))
df_mat_GSE203399_850k.to_csv(os.path.join(path_GSE203399_process, 'GSE203399_850k_mat.txt'))

save_dict = {}
X_np = df_mat_GSE203399_850k.T
y_pd = df_meta

additional_list = df_meta.columns
for row_number in range(len(y_pd)):
    key_rename = y_pd.iloc[row_number].sample_id
    feature_np = np.array(X_np.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([int(y_pd.iloc[row_number]["age"])]).astype(np.float32)
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

np.save(os.path.join(path_GSE203399_process, 'Processed_all_850k.npy'), save_dict)


# %%
# stroke
path_stroke = '/home/zhangyu/mnt_path/Data/stroke'
df_mat_GSE203399_450k = pd.read_csv(
    os.path.join(path_stroke, 'GSE203399_betas_normalized_pval_detection_discovery.txt'),
    sep='\t', index_col=0)
df_mat_GSE203399_450k = df_mat_GSE203399_450k.loc[:,
                        [one for one in df_mat_GSE203399_450k.columns if one[:9] != 'Detection']]

# GSE203399
path_GSE203399_process = os.path.join(path_stroke, 'GSE203399_process')
if not os.path.exists(path_GSE203399_process):
    os.mkdir(path_GSE203399_process)

file_series_matrix = os.path.join(path_stroke, 'GSE203399-GPL13534_series_matrix.txt')
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
                [one.strip("\"").split(': ')[1] for one in list_line[1:]],
                index=list_index, name=col_name))

df_meta = pd.concat(list_series, axis=1)
df_meta.index = df_meta['sample_id'].tolist()
df_mat_GSE203399_450k.columns = df_meta['sample_id'].tolist()
df_meta.to_csv(os.path.join(path_GSE203399_process, 'GSE203399-GPL13534_meta.txt'))

# all index
new_train_index = df_meta['sample_id'].tolist()
with open(os.path.join(path_GSE203399_process, "all_index.pkl"), 'wb') as f:
    pickle.dump(new_train_index, f)
    pickle.dump([], f)

# transform into dict
# all CpGs
file_GSE203399_450k_cpgs = os.path.join(path_GSE203399_process, 'GSE203399_450k_cpgs.txt')
write_stringList_2File(file_GSE203399_450k_cpgs, list(df_mat_GSE203399_450k.index))
df_mat_GSE203399_450k.to_csv(os.path.join(path_GSE203399_process, 'GSE203399_450k_mat.txt'))

save_dict = {}
X_np = df_mat_GSE203399_450k.T
y_pd = df_meta

additional_list = df_meta.columns
for row_number in range(len(y_pd)):
    key_rename = y_pd.iloc[row_number].sample_id
    feature_np = np.array(X_np.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([int(y_pd.iloc[row_number]["age"])]).astype(np.float32)
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

np.save(os.path.join(path_GSE203399_process, 'Processed_all_450k.npy'), save_dict)

# stroke
file_cpg_case = '/home/zhangyu/mnt_path/Data/EWAS_process/disease/cpgs_ewas_fudan_stroke_450k_1'
list_stroke_cpgs = read_stringList_FromFile(file_cpg_case)
list_cpgs_0 = [cpg for cpg in list_stroke_cpgs if cpg not in df_mat_GSE203399_450k.index]
df_mat_GSE203399_450k_plus = pd.concat(
    [df_mat_GSE203399_450k,
     pd.DataFrame(np.zeros((len(list_cpgs_0), df_mat_GSE203399_450k.shape[1])),
                  index=list_cpgs_0, columns=df_mat_GSE203399_450k.columns)], axis=0)
df_mat_GSE203399_stroke = df_mat_GSE203399_450k_plus.loc[list_stroke_cpgs, :]

save_dict = {}
X_np = df_mat_GSE203399_stroke.T
y_pd = df_meta

additional_list = ["sample_id", "age"]
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

np.save(os.path.join(path_GSE203399_process, 'Processed_stroke.npy'), save_dict)

# %%
# GSE69138
path_GSE69138_process = os.path.join(path_stroke, 'GSE69138_process')
if not os.path.exists(path_GSE69138_process):
    os.mkdir(path_GSE69138_process)

file_series_matrix = os.path.join(path_stroke, 'GSE69138_series_matrix.txt')
list_series = []
list_array = []
with open(file_series_matrix, 'r') as r_f:
    for line in r_f:
        # list_line = line.strip().split('\t')
        list_line = line.strip("\n").split('\t')
        if list_line[0] == '!Sample_title':
            list_index = [one.strip("\"").split(' ')[-1] for one in list_line[1:]]
        elif list_line[0] == '!Sample_geo_accession':
            list_series.append(pd.Series([one.strip("\"") for one in list_line[1:]],
                                         index=list_index, name='sample_id'))
        elif list_line[0] == '!Sample_characteristics_ch1':
            if list_line[1].strip("\"") == '':
                col_name = 'empty'
            else:
                col_name = list_line[1].strip("\"").split(': ')[0]
            list_series.append(pd.Series(
                [one.strip("\"").split(': ')[1] if one.strip("\"") != '' else '' for one in list_line[1:]],
                index=list_index, name=col_name))
        elif list_line[0] == '':
            continue
        elif list_line[0][0] == "\"" and list_line[0][1] == "I":
            id_ref = [one.strip("\"") for one in list_line[1:]]
        elif list_line[0][0] == "\"" and list_line[0][1] != "I":
            list_array.append(
                pd.Series([float(one) if one != '' else np.nan for one in list_line[1:]],
                          index=id_ref, name=list_line[0].strip("\"")))

df_mat = pd.concat(list_array, axis=1)
df_mat_notna = df_mat.loc[np.sum(pd.isna(df_mat), axis=1) != df_mat.shape[1], :]

df_meta = pd.concat(list_series, axis=1)
df_meta.index = df_meta['sample_id'].tolist()
df_meta_notna = df_meta.loc[
    df_mat_notna.index, ['sample_id', 'gender', 'stroke subtype', 'disease state', 'sample type']]
df_meta_notna.columns = ['sample_id', 'sex', 'stroke_subtype', 'disease', 'tissue']
df_meta_notna.to_csv(os.path.join(path_GSE69138_process, 'GSE69138_meta.txt'))

keep_cpgs = list(df_mat_notna.loc[:, np.sum(pd.isna(df_mat_notna), axis=0) < 40].columns)
df_mat_notna = df_mat_notna.loc[:, keep_cpgs]
X_mat_notna_imp, sample_stroke = impute_methy(df_mat_notna.T, df_meta_notna, by_project=False, use_col='')
df_mat_notna_imp = pd.DataFrame(X_mat_notna_imp, index=sample_stroke, columns=keep_cpgs)
df_mat_notna_imp.T.to_csv(os.path.join(path_GSE69138_process, 'GSE69138_mat.txt'),
                          index=True, header=True)
