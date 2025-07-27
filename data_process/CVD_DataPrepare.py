# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: DA_PD.py
# @time: 2023/8/15 11:34

import numpy as np
import pandas as pd
import random
import pickle
import os
import datatable as dt
os.chdir('/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/zhangyu001/bioage/code')
from sklearn.impute import SimpleImputer
from utils.file_utils import read_stringList_FromFile, write_stringList_2File
from fuzzywuzzy import process
from utils.common_utils import data_augmentation, read_single_csv, get_new_index


#%%
# keep samples
path_save = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/bioage/anno_fudan/blood_saliva'
set_ewas_disease_fudan_samples = \
    set(read_stringList_FromFile(os.path.join(path_save, 'keep_sample_ewas_disease_fudan')))
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


# stroke
path_stroke = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/stroke'
path_GSE203399_process = os.path.join(path_stroke, 'GSE203399_process')
df_stroke_1 = pd.read_csv(os.path.join(path_GSE203399_process, 'GSE203399_disc_mat.csv'),
                          index_col=0).T
df_meta_stroke_1 = pd.read_csv(
    os.path.join(path_GSE203399_process, 'GSE203399_disc_meta.txt'), index_col=0, sep='\t')

df_stroke_850k = pd.read_csv(os.path.join(path_GSE203399_process, 'GSE203399_rep_mat_1.csv'),
                             index_col=0).T
df_meta_stroke_850k = pd.read_csv(
    os.path.join(path_GSE203399_process, 'GSE203399_rep_meta.txt'), index_col=0, sep='\t')
df_stroke_850k_1 = df_stroke_850k.iloc[:30, :]
df_meta_stroke_850k_1 = df_meta_stroke_850k.iloc[:30, :]
df_stroke_850k_2 = df_stroke_850k.iloc[30:, :]
df_meta_stroke_850k_2 = df_meta_stroke_850k.iloc[30:, :]

df_stroke_1 = pd.concat([df_stroke_1, df_stroke_850k_1])
df_meta_stroke_1 = pd.concat([df_meta_stroke_1, df_meta_stroke_850k_1])
df_meta_stroke_1 = df_meta_stroke_1.loc[:, ['sample_id', 'age', 'gender']]
df_meta_stroke_1.columns = ['sample_id', 'age', 'sex']
df_meta_stroke_1['tissue'] = 'whole blood'
df_meta_stroke_1['sample_type'] = 'disease tissue'
df_meta_stroke_1['disease'] = 'stroke'
df_meta_stroke_1['project_id'] = 'GSE203399'
df_meta_stroke_1['bmi'] = -1
df_meta_stroke_1['platform'] = '450k'

df_stroke_850k = df_stroke_850k_2
df_meta_stroke_850k = df_meta_stroke_850k_2
df_meta_stroke_850k = df_meta_stroke_850k.loc[:, ['sample_id', 'age', 'gender']]
df_meta_stroke_850k.columns = ['sample_id', 'age', 'sex']
df_meta_stroke_850k['tissue'] = 'whole blood'
df_meta_stroke_850k['sample_type'] = 'disease tissue'
df_meta_stroke_850k['disease'] = 'stroke'
df_meta_stroke_850k['project_id'] = 'GSE203399'
df_meta_stroke_850k['bmi'] = -1
df_meta_stroke_850k['platform'] = '850k'

path_GSE69138_process = os.path.join(path_stroke, 'GSE69138_process')
df_stroke_2 = pd.read_csv(os.path.join(path_GSE69138_process, 'GSE69138_disc_mat.csv'), index_col=0).T
df_meta_stroke_2 = pd.read_csv(
    os.path.join(path_GSE69138_process, 'GSE69138_meta.txt'), index_col=0)
df_meta_stroke_2['age'] = -10
df_meta_stroke_2['tissue'] = 'whole blood'
df_meta_stroke_2['sample_type'] = 'disease tissue'
df_meta_stroke_2['disease'] = 'stroke'
df_meta_stroke_2['project_id'] = 'GSE69138'
df_meta_stroke_2['bmi'] = -1
df_meta_stroke_2['platform'] = '450k'

path_GSE197080_process = os.path.join(path_stroke, 'GSE197080_process')
df_stroke_3 = pd.read_csv(
    os.path.join(path_GSE197080_process, 'GSE197080_beta_GMQN_BMIQ_450k_impute.csv'), index_col=0).T
df_meta_stroke_3 = pd.read_csv(
    os.path.join(path_GSE197080_process, 'GSE197080_meta.tsv'), index_col=0, sep='\t')
df_meta_stroke_3['tissue'] = 'whole blood'
df_meta_stroke_3['bmi'] = -1
df_meta_stroke_3['platform'] = '850k'


#%%
# parameters
path_out = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/disease'
# disease_name = "type 2 diabetes"
# output_npz_file = os.path.join(path_out, "Processed_ewas_fudan_27k_DA_T2D.npy")
# folder_out = os.path.join(path_out, 'T2D_blood/')
# file_cpg_case = os.path.join(path_out, 'cpgs_ewas_fudan_T2D_27k')

disease_name = "stroke"
# output_npz_file = os.path.join(path_out, "Processed_blood_450k_850k_DA_stroke_pretrain_1.npy")
# folder_out = os.path.join(path_out, 'index_blood_450k_850k_stroke_pretrain_1/')
# file_cpg_case = os.path.join(path_out, 'cpgs_blood_450k_850k_stroke_pretrain_1')

dict_1 = ewas_disease_data_450k
df_meta_1 = df_meta_disease
cpg_list_1 = cpg_list_450k_disease
m2beta_1=False
dict_2 = data_850k
df_meta_2 = df_meta_850k
cpg_list_2 = cpg_list_850k
m2beta_2=False
dict_3 = ewas_age_data_450k
df_meta_3 = df_meta_age
m2beta_3=False
cpg_list_3 = cpg_list_450k_age
dict_4 = GSE196696_850k
df_meta_4 = df_meta_GSE196696
m2beta_4=False
cpg_list_4 = cpg_list_GSE196696

#%%
# T2D cpgs
cpg_list_jizhuan = read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/jizhuan/process/cpgs_jizhuan_850k_all.txt")
set_cpg_AS_850k = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/heart_disease/GSE220622_process/GSE220622_cpgs.txt"))
# set_cpg_T2D = set(read_stringList_FromFile(
#     "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_T2D_plus.txt"))
set_cpg_CCD = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_CCD.txt"))
# set_cpg_T2D = set(read_stringList_FromFile(
#     "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/disease/cpgs_ewas_fudan_T2D_Glu_450k"))

# %%
# overlap_cpgs = set(cpg_list_2).intersection(set(cpg_list_1)).intersection(
#     set_cpg_CCD).intersection(set(cpg_list_3)).intersection(
#     df_stroke_1.columns).intersection(df_stroke_2.columns).intersection(
#     set(set_cpg_AS_850k)).intersection(set(cpg_list_jizhuan)).intersection(
#     cpg_list_4).intersection(df_stroke_850k.columns).intersection(df_stroke_3.columns)
overlap_cpgs = \
    set(cpg_list_2).intersection(set(cpg_list_1)).intersection(set(cpg_list_3)).intersection(
    df_stroke_1.columns).intersection(df_stroke_2.columns).intersection(
    set(set_cpg_AS_850k)).intersection(set(cpg_list_jizhuan)).intersection(
    cpg_list_4).intersection(df_stroke_850k.columns).intersection(df_stroke_3.columns)
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
# if m2beta_1:
#     feature_case_1 = (1.0001*np.exp(feature_case_1)-0.0001)/(1+np.exp(feature_case_1))
feature_case_1 = pd.DataFrame(feature_case_1, index=list_overlap_1, columns=list_case_1)
feature_case_1 = feature_case_1.loc[list_overlap_1, :]

feature_ctrl_1 = np.concatenate(list_feature_ctrl_1, axis=1)
# if m2beta_1:
#     feature_ctrl_1 = (1.0001*np.exp(feature_ctrl_1)-0.0001)/(1+np.exp(feature_ctrl_1))
feature_ctrl_1 = pd.DataFrame(feature_ctrl_1, index=list_overlap_1, columns=list_ctrl_1)
feature_ctrl_1 = feature_ctrl_1.loc[list_overlap_1, :]

df_mat_stroke_1 = df_stroke_1.loc[:, list_overlap_1]
df_mat_stroke_2 = df_stroke_2.loc[:, list_overlap_1]
df_mat_stroke_3 = df_stroke_3.loc[:, list_overlap_1]
df_mat_stroke_850k = df_stroke_850k.loc[:, list_overlap_1]

# %%
# data 2
list_index_2 = get_new_index(cpg_list_2, list_overlap_1)
list_overlap_2 = list_overlap_1

# meta_case_2 = df_meta_2.loc[df_meta_2['disease'] == disease_name, :]
meta_ctrl_2 = df_meta_2.loc[df_meta_2['disease'] == "control", :]
# list_case_2 = meta_case_2['sample_id'].tolist()
list_ctrl_2 = meta_ctrl_2['sample_id'].tolist()

# list_feature_case_2 = []
# for sample_id in list_case_2:
#     one_sample = dict_2[sample_id]
#     old_feature = one_sample["feature"]
#     list_feature_case_2.append(old_feature[list_index_2][:, np.newaxis])
list_feature_ctrl_2 = []
for sample_id in list_ctrl_2:
    one_sample = dict_2[sample_id]
    old_feature = one_sample["feature"]
    list_feature_ctrl_2.append(old_feature[list_index_2][:, np.newaxis])

# feature_case_2 = np.concatenate(list_feature_case_2, axis=1)
# if m2beta_2:
#     feature_case_2 = (1.0001*np.exp(feature_case_2)-0.0001)/(1+np.exp(feature_case_2))
# feature_case_2 = pd.DataFrame(feature_case_2, index=list_overlap_2, columns=list_case_2)
# feature_case_2 = feature_case_2.loc[list_overlap_1, :]

feature_ctrl_2 = np.concatenate(list_feature_ctrl_2, axis=1)
# if m2beta_2:
#     feature_ctrl_2 = (1.0001*np.exp(feature_ctrl_2)-0.0001)/(1+np.exp(feature_ctrl_2))
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
# list_index_3 = [cpg_list_3.index(cpg) for cpg in cpg_list_3 if cpg in overlap_cpgs]
# list_overlap_3 = [cpg for cpg in cpg_list_3 if cpg in overlap_cpgs]
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

#%%
# pretrain data
path_pretrain = os.path.join(path_out, 'pretrain_blood_stroke_450k_850k')
if not os.path.exists(path_pretrain):
    os.mkdir(path_pretrain)
#
# X_df_merge = pd.concat([feature_case_1.T, df_mat_stroke_1, df_mat_stroke_850k,
#                         feature_ctrl_1.T, feature_ctrl_2.T, feature_ctrl_3.T, feature_ctrl_4.T])
# y_df_merge = pd.concat([df_meta_1.loc[list_case_1, :], df_meta_stroke_1, df_meta_stroke_850k,
#                         df_meta_1.loc[list_ctrl_1, :], df_meta_2, df_meta_3, df_meta_4])
# file_hdf_pretrain = os.path.join(path_pretrain, 'mat_methy.h5')
# X_df_merge.to_hdf(file_hdf_pretrain, key='data', mode='w')
# file_df_meta = os.path.join(path_pretrain, 'meta.tsv')
# y_df_merge.to_csv(file_df_meta, sep='\t')

# read mrs score
df_mrs_correct = \
    pd.read_csv(os.path.join(path_pretrain, 'meta_mrs_corrected.tsv'), sep='\t', index_col=0)
df_mrs_correct["age"] = df_mrs_correct["age"].astype(np.float32)
df_mrs_correct["mrs_combined"] = df_mrs_correct["mrs_combined"].astype(np.float32)
df_mrs_correct = df_mrs_correct.loc[df_mrs_correct['project_id'] != 'GSE62219', :]


# %%
# data augmentation
# case
new_df_meta_1 = df_mrs_correct.loc[list_case_1, :]
aug_case_1, meta_aug_case_1 = \
    data_augmentation(feature_case_1, new_df_meta_1, 7, by_project=False)
# new_df_meta_stroke_1 = df_mrs_correct.loc[df_meta_stroke_1.index, :]
# aug_case_1_2, meta_aug_case_1_2 = \
#     data_augmentation(df_mat_stroke_1.T, new_df_meta_stroke_1, 20, by_project=False)

#
# aug_case_stroke_2, meta_aug_case_stroke_2 = \
#     data_augmentation(df_mat_stroke_2.T, df_meta_stroke_2, 0, by_project=False)
new_df_meta_stroke_850k = df_mrs_correct.loc[df_meta_stroke_850k.index, :]
aug_case_stroke_850k, meta_aug_case_stroke_850k = \
    data_augmentation(df_mat_stroke_850k.T, new_df_meta_stroke_850k, 20, by_project=False)
#
# # ctrl
# aug_ctrl_1, meta_aug_ctrl_1 = \
#     data_augmentation(feature_ctrl_1, df_meta_1.loc[list_ctrl_1, :], 0, by_project=False)


# %%
# transform into dict
save_dict = {}

# validation data
X_np_val = pd.concat([df_mat_stroke_1, feature_ctrl_1.T, feature_ctrl_4.T])
y_pd_val = df_mrs_correct.loc[X_np_val.index, :]
X_np_val = pd.concat([X_np_val, df_mat_stroke_2])
y_pd_val = pd.concat([y_pd_val, df_meta_stroke_2])
y_pd_val["sex"] = y_pd_val["sex"].fillna("other")
y_pd_val["mrs_combined"] = y_pd_val["mrs_combined"].fillna(-10000)
sel_val_samples = random.sample(y_pd_val['sample_id'].tolist(), 2658)
X_np_val = X_np_val.loc[sel_val_samples, :]
y_pd_val = y_pd_val.loc[sel_val_samples, :]
set_val_sample = set(y_pd_val.loc[y_pd_val['mrs_combined'] > -10000, :].index)
X_np_val.index = [f"val_{i}" for i in range(len(sel_val_samples))]
y_pd_val.index = [f"val_{i}" for i in range(len(sel_val_samples))]

additional_list = ["age", "sex", 'sample_type', 'project_id']
type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
for row_number in range(len(y_pd_val)):
    key_rename = y_pd_val.index[row_number]
    feature_np = np.array(X_np_val.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    mrs = np.asarray([np.array(y_pd_val.iloc[row_number]['mrs_combined']).astype(np.float16)])
    save_dict[key_rename]["target"] = mrs

    additional = {}
    for additional_name in additional_list:
        value = y_pd_val.iloc[row_number][additional_name]
        additional[additional_name] = value

    additional["type_index"] = type2index[additional["sample_type"]]
    additional["project_id"] = "project_val"

    save_dict[key_rename]["additional"] = additional


# train data
X_np_train = pd.concat([aug_case_1, aug_case_stroke_850k, feature_ctrl_2.T, feature_ctrl_3.T])
y_pd_train = pd.concat([meta_aug_case_1, meta_aug_case_stroke_850k,
                        df_mrs_correct.loc[feature_ctrl_2.columns, :],
                        df_mrs_correct.loc[[one for one in feature_ctrl_3.columns if one in df_mrs_correct.index], :]])
y_pd_train["sex"] = y_pd_train["sex"].fillna("other")
y_pd_train["age"] = y_pd_train["age"].astype(np.float32)
# y_pd_train["mrs_combined"] = -10000
set_sample = set(y_pd_train.index).intersection(df_mrs_correct.index)
# y_pd_train.loc[[one for one in y_pd_train.index if one in set_sample], "mrs_combined"] = \
#     df_mrs_correct.loc[[one for one in y_pd_train.index if one in set_sample], ["mrs_combined"]]
y_pd_train["mrs_combined"] = y_pd_train["mrs_combined"].fillna(-10000)
df_project = pd.DataFrame(
    {'new_id': [f"project_{i}" for i in range(len(np.unique(y_pd_train['project_id'])))]},
    index=np.unique(y_pd_train['project_id'])
)
X_np_train = X_np_train.loc[y_pd_train.index, :]
X_np_train.index = [f"train_{i}" for i in range(len(y_pd_train.index))]
y_pd_train.index = [f"train_{i}" for i in range(len(y_pd_train.index))]

additional_list = ["age", "sex", 'sample_type', 'project_id']
type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
for row_number in range(len(y_pd_train)):
    key_rename = y_pd_train.index[row_number]
    feature_np = np.array(X_np_train.loc[key_rename, :]).astype(np.float16)
    mrs = np.asarray([np.array(y_pd_train.iloc[row_number]['mrs_combined']).astype(np.float32)])

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
# y_pd_train["age"] = y_pd_train["age"].astype(np.float32)
set_pretrain_index = set_sample.union(set_val_sample)
pretrain_index = y_pd_merge.loc[y_pd_merge['sample_id'].apply(lambda x: x in set_pretrain_index), :].index


# %%
# save data
# output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/models/train_data/epiAge_traindata.npz"
# np.savez(output_npz_file,
#          data=save_dict, cpgs=list_overlap_1,
#          train_index=list(y_pd_train.index), val_index=list(y_pd_val.index))

output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/models/train_data/CVD_traindata.npz"
np.savez_compressed(output_npz_file,
                    data=save_dict, cpgs=list_overlap_1,
                    train_index=list(y_pd_train.index), val_index=list(y_pd_val.index),
                    pretrain_index=pretrain_index)
