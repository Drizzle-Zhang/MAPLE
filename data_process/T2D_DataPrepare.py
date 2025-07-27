# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: DA_PD.py
# @time: 2023/8/15 11:34

import numpy as np
import pandas as pd
import datatable as dt
import h5py
import random
import pickle
import os
os.chdir('/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/zhangyu001/bioage/code')
from sklearn.impute import SimpleImputer
from utils.file_utils import read_stringList_FromFile, write_stringList_2File, FileUtils
from fuzzywuzzy import process
from utils.common_utils import data_augmentation, read_single_csv, get_new_index


#%%
# keep samples
keep_tissue = \
    {'whole blood', 'peripheral blood mononuclear cell', 'CD4+ T cell', 'CD8+ T cell',
     'CD14+ monocyte', 'leukocyte', 'lymphocyte', 'cord blood'}

# %%
# load all ewas disease data
ewas_disease_data_450k = np.load(
    '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/disease/450k_rm10/Processed.npy',
    allow_pickle = True).item()
df_meta_disease = \
    pd.read_csv("/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS/disease/sample_disease.txt", sep = " ")
df_meta_disease = df_meta_disease.drop_duplicates('sample_id', keep='first')
df_meta_disease = df_meta_disease.loc[:, ["sample_id", "tissue", "age", "sex", "bmi", 'disease', 'sample_type', 'project_id']]
df_meta_disease['disease'] = df_meta_disease['disease'].fillna('control')
df_meta_disease['platform'] = '450k'
df_meta_disease['bmi'] = df_meta_disease['bmi'].fillna(-1)
df_meta_disease = df_meta_disease.loc[
                  df_meta_disease['tissue'].apply(lambda x: x in keep_tissue), :]
df_meta_disease_ctrl = df_meta_disease.loc[df_meta_disease['sample_type'] == 'control', :]
list_ex_projects = df_meta_disease_ctrl.value_counts('project_id').loc[
    df_meta_disease_ctrl.value_counts('project_id') < 50].index
list_disease_gse = np.unique(
    df_meta_disease.loc[df_meta_disease['sample_type'] != 'control', 'project_id'])
list_ex_projects = [one for one in list_ex_projects if one not in list_disease_gse]
df_meta_disease = df_meta_disease.loc[
                  df_meta_disease['project_id'].apply(lambda x: x not in list_ex_projects), :]
cpg_list_450k_disease = read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/disease/450k_rm10/cpgs_list.txt")


# load age dataset
path_out_age = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/age/450k_rm10"
file_meta_age = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS/age/sample_age.txt"
file_cpg_450k_age = os.path.join(path_out_age, 'cpgs_list.txt')
file_npz_age = os.path.join(path_out_age, "Processed.npy")
ewas_age_data_450k = np.load(file_npz_age, allow_pickle = True).item()
df_meta_age = pd.read_csv(file_meta_age, sep=' ', index_col=0)
df_meta_age = \
    df_meta_age.loc[:, ["sample_id", "tissue", "age", "sex", "bmi", 'disease', 'sample_type', 'project_id']]
df_meta_age['platform'] = '450k'
df_meta_age = df_meta_age.loc[
              df_meta_age['tissue'].apply(lambda x: x in keep_tissue), :]
df_meta_age['bmi'] = df_meta_age['bmi'].fillna(-1)
df_meta_age = df_meta_age.loc[
              df_meta_age['project_id'].apply(
                  lambda x: x not in np.unique(df_meta_disease['project_id'])), :]
df_meta_age['disease'] = df_meta_age['disease'].fillna('control')
cpg_list_450k_age = read_stringList_FromFile(file_cpg_450k_age)


# load 850k data
path_850k = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/850k/merge'
data_850k = np.load(os.path.join(path_850k, 'Processed_850k.npy'),
                    allow_pickle = True).item()

df_meta_850k = pd.read_csv(os.path.join(path_850k, 'meta_850k.tsv'), sep='\t', index_col=0)
df_meta_850k.loc[(df_meta_850k['project_id'] == 'GSE147740'), 'tissue'] = 'peripheral blood mononuclear cell'
cpg_list_850k = read_stringList_FromFile(os.path.join(path_850k, "cpgs_850k.txt"))


# 850k GSE196696
path_GSE196696_process = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/850k/GSE196696_process'
GSE196696_850k = np.load(os.path.join(path_GSE196696_process, 'Processed_all_450k.npy'),
    allow_pickle = True).item()
df_meta_GSE196696 = pd.read_csv(
    os.path.join(path_GSE196696_process, 'GSE196696_meta.tsv'), sep='\t', index_col=0)
df_meta_GSE196696 = df_meta_GSE196696.loc[:, ["sample_id", "age", "sex", 'project_id']]
df_meta_GSE196696["tissue"] = 'whole blood'
df_meta_GSE196696["bmi"] = -1
df_meta_GSE196696["disease"] = 'control'
df_meta_GSE196696["sample_type"] = 'control'
df_meta_GSE196696["platform"] = '850k'
# list_sample_GSE196696 = [one for one in df_meta_GSE196696.index if one in GSE196696_850k.keys()]
# df_meta_GSE196696 = df_meta_GSE196696.loc[list_sample_GSE196696, :]
cpg_list_GSE196696 = read_stringList_FromFile(
    os.path.join(path_GSE196696_process, 'GSE196696_cpgs_450k.txt'))

# jizhuan
data_jizhuan_850k = np.load(
    '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/jizhuan/process/Processed_jizhuan_850k_all.npy',
    allow_pickle = True).item()
cpg_list_jizhuan = read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/jizhuan/process/cpgs_jizhuan_850k_all.txt")
df_meta_jizhuan = pd.read_csv(
    '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/jizhuan/process/samples_jizhuan_850k.txt', sep='\t', index_col=0)
df_meta_jizhuan['bmi'] = df_meta_jizhuan['Weight']/((df_meta_jizhuan['身高']/1000)**2)
df_meta_jizhuan["disease"] = ''
df_meta_jizhuan["sample_type"] = 'control'
df_meta_jizhuan.loc[df_meta_jizhuan['GLU'] >= 11, "disease"] = 'type 2 diabetes'
df_meta_jizhuan.loc[df_meta_jizhuan['GLU'] <= 6, "disease"] = 'control'
df_meta_jizhuan = df_meta_jizhuan.loc[df_meta_jizhuan["disease"] != '', :]
df_meta_jizhuan.loc[df_meta_jizhuan["disease"] != 'control', "sample_type"] = 'disease tissue'
sel_samples = [one for one in df_meta_jizhuan.index if one in data_jizhuan_850k.keys()]
df_meta_jizhuan = \
    df_meta_jizhuan.loc[:, ["sample_id", "age", "sex", "bmi", "disease", "sample_type"]]
df_meta_jizhuan["tissue"] = 'whole blood'
df_meta_jizhuan["project_id"] = 'jizhuan'
df_meta_jizhuan["platform"] = '850k'

# T1D
# GSE76169
path_t1d = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/T1D'
path_GSE76169_process = os.path.join(path_t1d, 'GSE76169_process')
file_GSE76169_mat = os.path.join(path_GSE76169_process, 'GSE76169_beta_GMQN_BMIQ_450k_impute.csv')
file_GSE76169_meta = os.path.join(path_GSE76169_process, 'GSE76169_meta.tsv')
df_mat_GSE76169 = pd.read_csv(file_GSE76169_mat, index_col=0)
cpg_list_GSE76169 = df_mat_GSE76169.index
df_meta_GSE76169 = pd.read_csv(file_GSE76169_meta, index_col=0, sep='\t')
df_meta_GSE76169["bmi"] = -1
df_meta_GSE76169["disease"] = 'type 1 diabetes'
df_meta_GSE76169["sample_type"] = 'disease tissue'
df_meta_GSE76169["platform"] = '450k'
df_meta_GSE76169["project_id"] = 'GSE76169'

# GSE76170
path_t1d = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/T1D'
path_GSE76170_process = os.path.join(path_t1d, 'GSE76170_process')
file_GSE76170_mat = os.path.join(path_GSE76170_process, 'GSE76170_beta_GMQN_BMIQ_450k_impute.csv')
file_GSE76170_meta = os.path.join(path_GSE76170_process, 'GSE76170_meta.tsv')
df_mat_GSE76170 = pd.read_csv(file_GSE76170_mat, index_col=0)
cpg_list_GSE76170 = df_mat_GSE76170.index
df_meta_GSE76170 = pd.read_csv(file_GSE76170_meta, index_col=0, sep='\t')
df_meta_GSE76170["bmi"] = -1
df_meta_GSE76170["disease"] = 'type 1 diabetes'
df_meta_GSE76170["sample_type"] = 'disease tissue'
df_meta_GSE76170["platform"] = '450k'
df_meta_GSE76170["project_id"] = 'GSE76170'

# T2D
# GSE197881
path_t2d = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/T2D'
path_GSE197881_process = os.path.join(path_t2d, 'GSE197881_process')
file_GSE197881_mat = os.path.join(path_GSE197881_process, 'GSE197881_beta_GMQN_BMIQ_450k_impute.csv')
file_GSE197881_meta = os.path.join(path_GSE197881_process, 'GSE197881_meta.tsv')
df_mat_GSE197881 = pd.read_csv(file_GSE197881_mat, index_col=0)
cpg_list_GSE197881 = df_mat_GSE197881.index
df_meta_GSE197881 = pd.read_csv(file_GSE197881_meta, index_col=0, sep='\t')
df_meta_GSE197881["age"] = -1
df_meta_GSE197881["bmi"] = -1
df_meta_GSE197881["platform"] = '450k'
df_meta_GSE197881["project_id"] = 'GSE197881'

# prediabetes
path_prediabetes = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/prediabetes'
path_GSE199700_process = os.path.join(path_prediabetes, 'GSE199700_process')
file_GSE199700_mat = os.path.join(path_GSE199700_process, 'GSE199700_850k_impute.csv')
file_GSE199700_meta = os.path.join(path_GSE199700_process, 'GSE199700_meta.tsv')
df_mat_GSE199700 = pd.read_csv(file_GSE199700_mat, index_col=0)
cpg_list_GSE199700 = df_mat_GSE199700.index
df_meta_GSE199700 = pd.read_csv(file_GSE199700_meta, index_col=0, sep='\t')
df_meta_GSE199700 = df_meta_GSE199700.loc[df_mat_GSE199700.columns, :]
df_meta_GSE199700 = df_meta_GSE199700.dropna(subset='age')
df_meta_GSE199700['bmi'] = df_meta_GSE199700['bmi (kg/m2)']
df_meta_GSE199700['disease'] = df_meta_GSE199700['disease state']
df_meta_GSE199700.loc[df_meta_GSE199700['disease state'] == 'PG', 'disease'] = 'prediabetes'
df_meta_GSE199700.loc[df_meta_GSE199700['disease state'] == 'NGG', 'disease'] = 'control'
df_meta_GSE199700['sample_type'] = df_meta_GSE199700['disease']
df_meta_GSE199700.loc[df_meta_GSE199700['disease'] != 'control', 'sample_type'] = 'disease tissue'
df_meta_GSE199700 = df_meta_GSE199700.loc[:, ['sample_id', 'age', 'bmi', 'disease', 'sample_type']]
df_meta_GSE199700['sex'] = 'F'
df_meta_GSE199700["tissue"] = 'leukocyte'
df_meta_GSE199700["platform"] = '850k'
df_meta_GSE199700["project_id"] = 'GSE199700'


#%%
# parameters
path_out = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/intermediate_data/T2D'

disease_name = "type 2 diabetes"
# output_npz_file = os.path.join(path_out, "Processed_blood_450k_850k_DA_T2D_pretrain_1.npy")
# folder_out = os.path.join(path_out, 'index_blood_450k_850k_T2D_pretrain_1/')
# file_cpg_case = os.path.join(path_out, 'cpgs_blood_450k_850k_T2D_pretrain_1')
path_pretrain = os.path.join(path_out, 'pretrain_data_blood_450k_850k_T2D')
#
# if not os.path.exists(path_pretrain):
#     os.mkdir(path_pretrain)
# if not os.path.exists(folder_out):
#     os.mkdir(folder_out)


dict_1 = ewas_disease_data_450k
df_meta_1 = df_meta_disease
cpg_list_1 = cpg_list_450k_disease
dict_2 = data_850k
df_meta_2 = df_meta_850k
cpg_list_2 = cpg_list_850k
dict_3 = ewas_age_data_450k
df_meta_3 = df_meta_age
cpg_list_3 = cpg_list_450k_age
dict_4 = GSE196696_850k
df_meta_4 = df_meta_GSE196696
cpg_list_4 = cpg_list_GSE196696
dict_5 = data_jizhuan_850k
df_meta_5 = df_meta_jizhuan
cpg_list_5 = cpg_list_jizhuan

#%%
# T2D cpgs
file_GSE210255_cpgs = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/GENOA/GSE210255_process/GSE210255_cpgs_450k.txt'
set_GSE210255_cpgs = set(read_stringList_FromFile(file_GSE210255_cpgs))
file_GSE207927_cpgs = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/850k/GSE207927_process/GSE207927_cpgs_850k.txt'
set_GSE207927_cpgs = set(read_stringList_FromFile(file_GSE207927_cpgs))
# set_cpg_T2D = set(read_stringList_FromFile(
#     "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_T2D_plus.txt"))

# %%
overlap_cpgs = \
    set(cpg_list_2).intersection(set(cpg_list_1)).intersection(set(cpg_list_3)).intersection(
        cpg_list_5).intersection(cpg_list_4).intersection(cpg_list_GSE76169).intersection(
        cpg_list_GSE76170).intersection(cpg_list_GSE197881).intersection(
        set_GSE210255_cpgs).intersection(set_GSE207927_cpgs).intersection(cpg_list_GSE199700)
print(len(overlap_cpgs))
# list_index_1 = [cpg_list_1.index(cpg) for cpg in cpg_list_1 if cpg in overlap_cpgs]
list_overlap_1 = [cpg for cpg in cpg_list_1 if cpg in overlap_cpgs]
# write_stringList_2File(file_cpg_case, list_overlap_1)
list_index_1 = get_new_index(cpg_list_1, list_overlap_1)

# %%
# case and control samples
list_ctrl_1 = df_meta_1.loc[
    (df_meta_1['tissue'].apply(lambda x: x in keep_tissue)) &
    (df_meta_1['sample_type'] == 'control'), 'sample_id'].tolist()
list_case_1 = df_meta_1.loc[
    (df_meta_1['tissue'].apply(lambda x: x in keep_tissue)) &
    (df_meta_1['disease'] == disease_name), 'sample_id'].tolist()

list_feature_case_1 = []
for sample_id in list_case_1:
    one_sample = dict_1[sample_id]
    old_feature = one_sample["feature"]
    list_feature_case_1.append(old_feature[list_index_1][:, np.newaxis])
list_feature_ctrl_1 = []
for sample_id in list_ctrl_1:
    one_sample = dict_1[sample_id]
    old_feature = one_sample["feature"]
    list_feature_ctrl_1.append(old_feature[list_index_1][:, np.newaxis])

feature_case_1 = np.concatenate(list_feature_case_1, axis=1)
feature_case_1 = pd.DataFrame(feature_case_1, index=list_overlap_1, columns=list_case_1)
feature_case_1 = feature_case_1.loc[list_overlap_1, :]

feature_ctrl_1 = np.concatenate(list_feature_ctrl_1, axis=1)
feature_ctrl_1 = pd.DataFrame(feature_ctrl_1, index=list_overlap_1, columns=list_ctrl_1)
feature_ctrl_1 = feature_ctrl_1.loc[list_overlap_1, :]

#%%
# SIR
meta_sir_1 = df_meta_1.loc[
    (df_meta_1['tissue'].apply(lambda x: x in keep_tissue)) &
    (df_meta_1['disease'] == 'systemic insulin resistance'), :]
list_sir_1 = meta_sir_1.loc[:, 'sample_id'].tolist()

list_feature_sir_1 = []
for sample_id in list_sir_1:
    one_sample = dict_1[sample_id]
    old_feature = one_sample["feature"]
    list_feature_sir_1.append(old_feature[list_index_1][:, np.newaxis])

feature_sir_1 = np.concatenate(list_feature_sir_1, axis=1)
feature_sir_1 = pd.DataFrame(feature_sir_1, index=list_overlap_1, columns=list_sir_1)
feature_sir_1 = feature_sir_1.loc[list_overlap_1, :]

# %%
# data 2
list_index_2 = get_new_index(cpg_list_2, list_overlap_1)
list_overlap_2 = list_overlap_1

meta_ctrl_2 = df_meta_2.loc[df_meta_2['disease'] == "control", :]
list_ctrl_2 = meta_ctrl_2['sample_id'].tolist()

list_feature_ctrl_2 = []
for sample_id in list_ctrl_2:
    one_sample = dict_2[sample_id]
    old_feature = one_sample["feature"]
    list_feature_ctrl_2.append(old_feature[list_index_2][:, np.newaxis])

feature_ctrl_2 = np.concatenate(list_feature_ctrl_2, axis=1)
feature_ctrl_2 = pd.DataFrame(feature_ctrl_2, index=list_overlap_2, columns=list_ctrl_2)
feature_ctrl_2 = feature_ctrl_2.loc[list_overlap_1, :]

# %%
# data 3
list_index_3 = get_new_index(cpg_list_3, list_overlap_1)
list_overlap_3 = list_overlap_1

list_ctrl_3 = df_meta_3['sample_id'].tolist()
list_feature_ctrl_3 = []
for sample_id in list_ctrl_3:
    one_sample = dict_3[sample_id]
    old_feature = one_sample["feature"]
    list_feature_ctrl_3.append(old_feature[list_index_3][:, np.newaxis])

feature_ctrl_3 = np.concatenate(list_feature_ctrl_3, axis=1)
feature_ctrl_3 = pd.DataFrame(feature_ctrl_3, index=list_overlap_3, columns=list_ctrl_3)
feature_ctrl_3 = feature_ctrl_3.loc[list_overlap_1, :]

# %%
# data 4
list_index_4 = get_new_index(cpg_list_4, list_overlap_1)
list_overlap_4 = list_overlap_1

list_ctrl_4 = df_meta_4['sample_id'].tolist()
list_feature_ctrl_4 = []
for sample_id in list_ctrl_4:
    one_sample = dict_4[sample_id]
    old_feature = one_sample["feature"]
    list_feature_ctrl_4.append(old_feature[list_index_4][:, np.newaxis])

feature_ctrl_4 = np.concatenate(list_feature_ctrl_4, axis=1)
feature_ctrl_4 = pd.DataFrame(feature_ctrl_4, index=list_overlap_4, columns=list_ctrl_4)
feature_ctrl_4 = feature_ctrl_4.loc[list_overlap_1, :]

# %%
# data 5
list_index_5 = get_new_index(cpg_list_5, list_overlap_1)
list_overlap_5 = list_overlap_1

meta_case_5 = df_meta_5.loc[df_meta_5['disease'] == disease_name, :]
meta_ctrl_5 = df_meta_5.loc[df_meta_5['disease'] == "control", :]
list_case_5 = meta_case_5['sample_id'].tolist()
list_ctrl_5 = meta_ctrl_5['sample_id'].tolist()

list_feature_case_5 = []
for sample_id in list_case_5:
    one_sample = dict_5[sample_id]
    old_feature = one_sample["feature"]
    list_feature_case_5.append(old_feature[list_index_5][:, np.newaxis])
list_feature_ctrl_5 = []
for sample_id in list_ctrl_5:
    one_sample = dict_5[sample_id]
    old_feature = one_sample["feature"]
    list_feature_ctrl_5.append(old_feature[list_index_5][:, np.newaxis])

feature_case_5 = np.concatenate(list_feature_case_5, axis=1)
feature_case_5 = pd.DataFrame(feature_case_5, index=list_overlap_5, columns=list_case_5)
feature_case_5 = feature_case_5.loc[list_overlap_1, :]

feature_ctrl_5 = np.concatenate(list_feature_ctrl_5, axis=1)
feature_ctrl_5 = pd.DataFrame(feature_ctrl_5, index=list_overlap_5, columns=list_ctrl_5)
feature_ctrl_5 = feature_ctrl_5.loc[list_overlap_1, :]

# %%
# diabetes
df_mat_t1d_wb = df_mat_GSE76169.T.loc[:, list_overlap_1]
df_mat_t1d_mon = df_mat_GSE76170.T.loc[:, list_overlap_1]
df_mat_t2d = df_mat_GSE197881.T.loc[:, list_overlap_1]
df_mat_pd = df_mat_GSE199700.T.loc[:, list_overlap_1]

#%%
# pretrain data
# X_df_merge = pd.concat([feature_case_1.T, feature_case_5.T,
#                         feature_ctrl_1.T, feature_ctrl_2.T, feature_ctrl_3.T, feature_ctrl_4.T,
#                         feature_ctrl_5.T,
#                         df_mat_t1d_wb, df_mat_t1d_mon, df_mat_t2d, df_mat_pd])
# y_df_merge = pd.concat([df_meta_1.loc[list_case_1, :], meta_case_5,
#                         df_meta_1.loc[list_ctrl_1, :], df_meta_2, df_meta_3, df_meta_4, meta_ctrl_5,
#                         df_meta_GSE76169, df_meta_GSE76170, df_meta_GSE197881, df_meta_GSE199700])
# file_hdf_pretrain = os.path.join(path_pretrain, 'mat_methy.h5')
# X_df_merge.to_hdf(file_hdf_pretrain, key='data', mode='w')
# file_df_meta = os.path.join(path_pretrain, 'meta.tsv')
# y_df_merge.to_csv(file_df_meta, sep='\t')

# read T2D score from Nature aging
df_mrs_correct = \
    pd.read_csv(os.path.join(path_pretrain, 'meta_corrected.tsv'), sep='\t', index_col=0)
df_mrs_correct["age"] = df_mrs_correct["age"].astype(np.float32)
df_mrs_correct["T2D_risk_norm"] = df_mrs_correct["T2D_risk_norm"].astype(np.float32)
df_mrs_correct = df_mrs_correct.loc[df_mrs_correct['project_id'] != 'GSE62219', :]


# %%
# data augmentation
# case
# new_df_meta_1 = df_mrs_correct.loc[list_case_1, :]
# aug_case_1, meta_aug_case_1 = \
#     data_augmentation(feature_case_1, new_df_meta_1, 50, by_project=False)
# new_df_meta_stroke_1 = df_mrs_correct.loc[df_meta_stroke_1.index, :]
# aug_case_1_2, meta_aug_case_1_2 = \
#     data_augmentation(df_mat_stroke_1.T, new_df_meta_stroke_1, 20, by_project=False)

df_meta_t1d_wb = df_mrs_correct.loc[df_meta_GSE76169.index, :]
aug_case_t1d_wb, meta_aug_case_t1d_wb = \
    data_augmentation(df_mat_t1d_wb.T, df_meta_t1d_wb, 10, by_project=False)
# df_meta_t1d_mon = df_mrs_correct.loc[df_meta_GSE76170.index, :]
# aug_case_t1d_mon, meta_aug_case_t1d_mon = \
#     data_augmentation(df_mat_t1d_mon.T, df_meta_t1d_mon, 10, by_project=False)
#
new_df_meta_5 = df_mrs_correct.loc[meta_case_5.index, :]
aug_case_5, meta_aug_case_5 = \
    data_augmentation(feature_case_5, new_df_meta_5, 50, by_project=False)


# %%
# transform into dict
save_dict = {}

df_mrs_correct['T2D_risk_scale'] = 0.5 + (df_mrs_correct['T2D_risk_norm'] - np.mean(df_mrs_correct['T2D_risk_norm'])) / np.std(df_mrs_correct['T2D_risk_norm'])
p5, p95 = np.quantile(df_mrs_correct['T2D_risk_scale'], [0.05, 0.95])
data_scaled = (df_mrs_correct['T2D_risk_scale'] - p5) / (p95 - p5)
df_mrs_correct['T2D_risk_scale'] = data_scaled

# val data
X_np_val = pd.concat([feature_ctrl_1.T, feature_ctrl_4.T])
y_pd_val = pd.concat([
    df_mrs_correct.loc[[one for one in feature_ctrl_1.columns if one in df_mrs_correct.index], :],
    df_mrs_correct.loc[feature_ctrl_4.columns, :],
])
sel_val_samples = random.sample(y_pd_val['sample_id'].tolist(), 1791)
X_np_val = X_np_val.loc[sel_val_samples, :]
y_pd_val = y_pd_val.loc[sel_val_samples, :]
X_np_val = pd.concat([X_np_val, feature_case_1.T, df_mat_t1d_mon, df_mat_t2d])
y_pd_val = pd.concat([y_pd_val, df_mrs_correct.loc[feature_case_1.columns, :],
                      df_mrs_correct.loc[df_meta_GSE76170.index, :],
                      df_mrs_correct.loc[df_meta_GSE197881.index, :]])
y_pd_val["sex"] = y_pd_val["sex"].fillna("other")
y_pd_val["T2D_risk_scale"] = y_pd_val["T2D_risk_scale"].fillna(-10000)
set_val_sample = set(y_pd_val.loc[y_pd_val['T2D_risk_scale'] > -10000, :].index)
X_np_val.index = [f"val_{i}" for i in range(len(y_pd_val.index))]
y_pd_val.index = [f"val_{i}" for i in range(len(y_pd_val.index))]

additional_list = ["age", "sex", 'sample_type', 'project_id']
type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
for row_number in range(len(y_pd_val)):
    key_rename = y_pd_val.index[row_number]
    feature_np = np.array(X_np_val.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    mrs = np.asarray([np.array(y_pd_val.iloc[row_number]["T2D_risk_scale"]).astype(np.float16)])
    save_dict[key_rename]["target"] = mrs

    additional = {}
    for additional_name in additional_list:
        value = y_pd_val.iloc[row_number][additional_name]
        additional[additional_name] = value

    additional["type_index"] = type2index[additional["sample_type"]]
    additional["project_id"] = "project_val"
    save_dict[key_rename]["additional"] = additional

# train data
X_np_train = pd.concat([aug_case_t1d_wb, aug_case_5,
                        feature_ctrl_2.T, feature_ctrl_3.T, feature_ctrl_5.T])
y_pd_train = pd.concat([meta_aug_case_t1d_wb, meta_aug_case_5,
                        df_mrs_correct.loc[feature_ctrl_2.columns, :],
                        df_mrs_correct.loc[[one for one in feature_ctrl_3.columns if one in df_mrs_correct.index], :],
                        df_mrs_correct.loc[feature_ctrl_5.columns, :]])
sel_val_samples = random.sample(y_pd_train['sample_id'].tolist(), 6830)
X_np_train = X_np_train.loc[sel_val_samples, :]
y_pd_train = y_pd_train.loc[sel_val_samples, :]
y_pd_train["sex"] = y_pd_train["sex"].fillna("other")
y_pd_train["age"] = y_pd_train["age"].astype(np.float32)
y_pd_train["T2D_risk_scale"] = y_pd_train["T2D_risk_scale"].fillna(-10000)
set_sample = set(y_pd_train.index).intersection(df_mrs_correct.index)
X_np_train.index = [f"train_{i}" for i in range(len(y_pd_train.index))]
y_pd_train.index = [f"train_{i}" for i in range(len(y_pd_train.index))]
df_project = pd.DataFrame(
    {'new_id': [f"project_{i}" for i in range(len(np.unique(y_pd_train['project_id'])))]},
    index=np.unique(y_pd_train['project_id'])
)

additional_list = ["age", "sex", 'sample_type', 'project_id']
type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
for row_number in range(len(y_pd_train)):
    key_rename = y_pd_train.index[row_number]
    feature_np = np.array(X_np_train.loc[key_rename, :]).astype(np.float16)
    mrs = np.asarray([np.array(y_pd_train.iloc[row_number]['T2D_risk_scale']).astype(np.float32)])

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    save_dict[key_rename]["target"] = mrs

    additional = {}
    for additional_name in additional_list:
        value = y_pd_train.iloc[row_number][additional_name]
        if additional_name == 'project_id':
            value = df_project.loc[y_pd_train.iloc[row_number][additional_name], 'new_id']
        additional[additional_name] = value

    additional["type_index"] = type2index[additional["sample_type"]]

    save_dict[key_rename]["additional"] = additional

y_pd_merge = pd.concat([y_pd_train, y_pd_val])
set_pretrain_index = set_sample.union(set_val_sample)
pretrain_index = y_pd_merge.loc[y_pd_merge['sample_id'].apply(lambda x: x in set_pretrain_index), :].index

# %%
# save data
# output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/models/train_data/epiAge_traindata.npz"
# np.savez(output_npz_file,
#          data=save_dict, cpgs=list_overlap_1,
#          train_index=list(y_pd_train.index), val_index=list(y_pd_val.index))

output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/models/train_data/T2D_traindata.npz"
np.savez_compressed(output_npz_file,
                    data=save_dict, cpgs=list_overlap_1,
                    train_index=list(y_pd_train.index), val_index=list(y_pd_val.index),
                    pretrain_index=pretrain_index)
