# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @time: 2024/3/21 11:34

import numpy as np
import pandas as pd
import datatable as dt
import random
import pickle
import os
os.chdir('/home/zhangyu/bioage/code')
from sklearn.impute import SimpleImputer
from utils.file_utils import read_stringList_FromFile, write_stringList_2File
from fuzzywuzzy import process
from utils.common_utils import data_augmentation, get_new_index, order_cpg_to_ref, order_cpg_to_ref_fill0_2


# %%
# load all ewas disease data
ewas_disease_data_450k = np.load(
    '/home/zhangyu/mnt_path/Data/EWAS_process/disease/450k_rm10/Processed.npy',
    allow_pickle = True).item()
df_meta_disease = \
    pd.read_csv("/home/zhangyu/mnt_path/Data/EWAS/disease/sample_disease.txt", sep = " ")
df_meta_disease = df_meta_disease.loc[:, ["sample_id", "tissue", "age", "sex", "bmi", 'disease', 'sample_type', 'project_id']]
df_meta_disease = df_meta_disease.drop_duplicates('sample_id', keep='first')
cpg_list_450k_disease = read_stringList_FromFile(
    "/home/zhangyu/mnt_path/Data/EWAS_process/disease/450k_rm10/cpgs_list.txt")
set_ewas = set(df_meta_disease.index)

# bmi data
path_out_bmi = "/home/zhangyu/mnt_path/Data/EWAS_process/bmi/450k_rm10"
file_meta_bmi = "/home/zhangyu/mnt_path/Data/EWAS/bmi/sample_bmi.txt"
file_cpg_450k_bmi = os.path.join(path_out_bmi, 'cpgs_list.txt')
file_npz_bmi = os.path.join(path_out_bmi, "Processed.npy")
ewas_bmi_data_450k = np.load(file_npz_bmi, allow_pickle = True).item()
df_meta_bmi = pd.read_csv(file_meta_bmi, sep=' ', index_col=0)
df_meta_bmi = df_meta_bmi.loc[
              df_meta_bmi['sample_id'].apply(lambda x: x not in set_ewas), :]
cpg_list_450k_bmi = read_stringList_FromFile(file_cpg_450k_bmi)
set_ewas = set_ewas.union(set(df_meta_bmi.index))

# load age dataset
path_out_age = "/home/zhangyu/mnt_path/Data/EWAS_process/age/450k_rm10"
file_meta_age = "/home/zhangyu/mnt_path/Data/EWAS/age/sample_age.txt"
file_cpg_450k_age = os.path.join(path_out_age, 'cpgs_list.txt')
file_npz_age = os.path.join(path_out_age, "Processed.npy")
ewas_age_data_450k = np.load(file_npz_age, allow_pickle = True).item()
df_meta_age = pd.read_csv(file_meta_age, sep=' ', index_col=0)
cpg_list_450k_age = read_stringList_FromFile(file_cpg_450k_age)
df_meta_age = df_meta_age.loc[
              df_meta_age['sample_id'].apply(lambda x: x not in set_ewas), :]
set_ewas = set_ewas.union(set(df_meta_age.index))

# tissue data
file_meta_tissue = "/home/zhangyu/mnt_path/Data/EWAS/tissue/sample_tissue.txt"
path_out_tissue = "/home/zhangyu/mnt_path/Data/EWAS_process/tissue/450k_rm10"
file_cpg_450k_tissue = os.path.join(path_out_tissue, 'cpgs_list.txt')
file_npz_tissue = os.path.join(path_out_tissue, "Processed.npy")
ewas_tissue_data_450k = np.load(file_npz_tissue, allow_pickle = True).item()
df_meta_tissue = pd.read_csv(file_meta_tissue, sep=' ', index_col=0)
cpg_list_450k_tissue = read_stringList_FromFile(file_cpg_450k_tissue)
df_meta_tissue = df_meta_tissue.loc[
                 df_meta_tissue['sample_id'].apply(lambda x: x not in set_ewas), :]
set_ewas = set_ewas.union(set(df_meta_tissue.index))

# blood data
path_out_blood = "/home/zhangyu/mnt_path/Data/EWAS_process/blood/450k_rm10"
file_meta_blood = "/home/zhangyu/mnt_path/Data/EWAS/blood/sample_blood.txt"
file_cpg_450k_blood = os.path.join(path_out_blood, 'cpgs_list.txt')
file_npz_blood = os.path.join(path_out_blood, "Processed.npy")
ewas_blood_data_450k = np.load(file_npz_blood, allow_pickle = True).item()
df_meta_blood = pd.read_csv(file_meta_blood, sep=' ', index_col=0)
cpg_list_450k_blood = read_stringList_FromFile(file_cpg_450k_blood)
df_meta_blood = df_meta_blood.loc[
              df_meta_blood['sample_id'].apply(lambda x: x not in set_ewas), :]
set_ewas = set_ewas.union(set(df_meta_blood.index))

# brain data
path_out_brain = "/home/zhangyu/mnt_path/Data/EWAS_process/brain/450k_rm10"
file_meta_brain = "/home/zhangyu/mnt_path/Data/EWAS/brain/sample_brain.txt"
file_cpg_450k_brain = os.path.join(path_out_brain, 'cpgs_list.txt')
file_npz_brain = os.path.join(path_out_brain, "Processed.npy")
ewas_brain_data_450k = np.load(file_npz_brain, allow_pickle = True).item()
df_meta_brain = pd.read_csv(file_meta_brain, sep=' ', index_col=0)
cpg_list_450k_brain = read_stringList_FromFile(file_cpg_450k_brain)
df_meta_brain = df_meta_brain.loc[
                df_meta_brain['sample_id'].apply(lambda x: x not in set_ewas), :]
set_ewas = set_ewas.union(set(df_meta_brain.index))

# sex data
path_out_sex = "/home/zhangyu/mnt_path/Data/EWAS_process/sex/450k_rm10"
file_meta_sex = "/home/zhangyu/mnt_path/Data/EWAS/sex/sample_sex.txt"
file_cpg_450k_sex = os.path.join(path_out_sex, 'cpgs_list.txt')
file_npz_sex = os.path.join(path_out_sex, "Processed.npy")
ewas_sex_data_450k = np.load(file_npz_sex, allow_pickle = True).item()
df_meta_sex = pd.read_csv(file_meta_sex, sep=' ', index_col=0)
df_meta_sex.loc[(df_meta_sex['project_id'] == 'GSE60655') & (df_meta_sex['sex'] == 'M'), 'age'] = 27.5
df_meta_sex.loc[(df_meta_sex['project_id'] == 'GSE60655') & (df_meta_sex['sex'] == 'F'), 'age'] = 26.4
cpg_list_450k_sex = read_stringList_FromFile(file_cpg_450k_sex)
df_meta_sex = df_meta_sex.loc[
              df_meta_sex['sample_id'].apply(lambda x: x not in set_ewas), :]


#%%
# parameters
path_out = '/home/zhangyu/mnt_path/intermediate_data/merge_ewas_datahub'

output_npz_file = os.path.join(path_out, "meth_beta_all_450k_rm10.npy")
file_cpg_case = os.path.join(path_out, 'cpgs_all_450k_rm10.txt')
file_mat = os.path.join(path_out, 'mat_beta_all_450k_rm10.csv')
file_meta = os.path.join(path_out, 'meta_all_450k.tsv')

# dict_1 = ewas_disease_data_27k
dict_1 = ewas_disease_data_450k
df_meta_1 = df_meta_disease
m2beta_1=False
cpg_list_1 = cpg_list_450k_disease
dict_3 = ewas_age_data_450k
df_meta_3 = df_meta_age
m2beta_3=False
cpg_list_3 = cpg_list_450k_age
dict_7 = ewas_bmi_data_450k
df_meta_7 = df_meta_bmi
cpg_list_7 = cpg_list_450k_bmi
dict_8 = ewas_tissue_data_450k
df_meta_8 = df_meta_tissue
cpg_list_8 = cpg_list_450k_tissue
dict_9 = ewas_blood_data_450k
df_meta_9 = df_meta_blood
cpg_list_9 = cpg_list_450k_blood
dict_10 = ewas_brain_data_450k
df_meta_10 = df_meta_brain
cpg_list_10 = cpg_list_450k_brain
dict_11 = ewas_sex_data_450k
df_meta_11 = df_meta_sex
cpg_list_11 = cpg_list_450k_sex


# %%
# train data cpgs
overlap_cpgs = set(cpg_list_1).intersection(cpg_list_3).intersection(
    cpg_list_7).intersection(cpg_list_8).intersection(cpg_list_9).intersection(
    cpg_list_10).intersection(cpg_list_11)
print(len(overlap_cpgs))
# list_index_1 = [cpg_list_1.index(cpg) for cpg in cpg_list_1 if cpg in overlap_cpgs]
list_overlap_1 = [cpg for cpg in cpg_list_1 if cpg in overlap_cpgs]
write_stringList_2File(file_cpg_case, list_overlap_1)


# %%
# train data
def process_train_data(dict_one, df_meta_one, cpg_list_one, list_overlap_ref, m2beta_one=False):
    list_index_one = get_new_index(cpg_list_one, list_overlap_ref)
    list_overlap_one = list_overlap_ref

    list_ctrl_one = df_meta_one['sample_id'].tolist()

    list_feature_ctrl_one = []
    for sample_id_one in list_ctrl_one:
        sample_one = dict_one[sample_id_one]
        old_feature_one = sample_one["feature"]
        list_feature_ctrl_one.append(old_feature_one[list_index_one][:, np.newaxis])

    feature_ctrl_one = np.concatenate(list_feature_ctrl_one, axis=1)
    if m2beta_one:
        feature_ctrl_one = (1.0001*np.exp(feature_ctrl_one)-0.0001)/(1+np.exp(feature_ctrl_one))
    feature_ctrl_one = pd.DataFrame(feature_ctrl_one, index=list_overlap_one, columns=list_ctrl_one)
    feature_ctrl_one = feature_ctrl_one.loc[list_overlap_ref, :]

    return feature_ctrl_one


# %%
feature_ctrl_1 = process_train_data(dict_1, df_meta_1, cpg_list_1, list_overlap_1)

feature_ctrl_3 = process_train_data(dict_3, df_meta_3, cpg_list_3, list_overlap_1)

feature_ctrl_7 = process_train_data(dict_7, df_meta_7, cpg_list_7, list_overlap_1)

feature_ctrl_8 = process_train_data(dict_8, df_meta_8, cpg_list_8, list_overlap_1)

feature_ctrl_9 = process_train_data(dict_9, df_meta_9, cpg_list_9, list_overlap_1)

feature_ctrl_10 = process_train_data(dict_10, df_meta_10, cpg_list_10, list_overlap_1)

feature_ctrl_11 = process_train_data(dict_11, df_meta_11, cpg_list_11, list_overlap_1)


# %%
# transform into dict
save_dict = {}

# fudan data
X_np_fudan = pd.concat([feature_ctrl_1.T, feature_ctrl_3.T, feature_ctrl_7.T,
                        feature_ctrl_8.T, feature_ctrl_9.T, feature_ctrl_10.T, feature_ctrl_11.T])
y_pd_fudan = pd.concat(
    [df_meta_1, df_meta_3, df_meta_7, df_meta_8, df_meta_9, df_meta_10, df_meta_11])
y_pd_fudan['disease'] = y_pd_fudan['disease'].fillna('control')
y_pd_fudan["sex"] = y_pd_fudan["sex"].fillna("other")
y_pd_fudan['bmi'] = y_pd_fudan['bmi'].fillna(-1)
y_pd_fudan['age'] = y_pd_fudan['age'].fillna(-1)

additional_list = ["sample_id", "tissue", "age", "sex", "bmi", 'disease', 'sample_type', 'project_id']
type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
for row_number in range(len(y_pd_fudan)):
    key_rename = y_pd_fudan.iloc[row_number].sample_id
    feature_np = np.array(X_np_fudan.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd_fudan.iloc[row_number]["age"].astype(np.float32)])
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd_fudan.iloc[row_number][additional_name]
        additional[additional_name] = value

    additional["type_index"] = type2index[additional["sample_type"]]

    save_dict[key_rename]["additional"] = additional

# %%
# save data
np.save(output_npz_file, save_dict)

# save training matrix
df_mat_train = X_np_fudan.T
df_mat_train.insert(loc=0, column='CpG', value=list_overlap_1)
df_mat_train = dt.Frame(df_mat_train)
df_mat_train.to_csv(file_mat)

# save sample info
y_pd_fudan.to_csv(file_meta, sep='\t')
