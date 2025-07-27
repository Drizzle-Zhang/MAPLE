# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @time: 2024/3/21 11:34

import numpy as np
import pandas as pd
import datatable as dt
import random
import pickle
import os
os.chdir('/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/zhangyu001/bioage/code')
from sklearn.impute import SimpleImputer
from utils.file_utils import read_stringList_FromFile, write_stringList_2File
from fuzzywuzzy import process
from utils.common_utils import data_augmentation, get_new_index, order_cpg_to_ref, order_cpg_to_ref_fill0_2


# %%
# load all ewas datahub data
path_ewas = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/intermediate_data/merge_ewas_datahub'
npz_file_ewas = os.path.join(path_ewas, "meth_beta_all_450k_rm10.npy")
file_cpg_ewas = os.path.join(path_ewas, 'cpgs_all_450k_rm10.txt')
file_meta_ewas = os.path.join(path_ewas, 'meta_all_450k.tsv')

ewas_data_450k = np.load(npz_file_ewas, allow_pickle = True).item()
df_meta_ewas = pd.read_csv(file_meta_ewas, sep = "\t", index_col=0)
print(df_meta_ewas.shape)
df_meta_ewas.loc[(df_meta_ewas['project_id'] == 'GSE60655') & (df_meta_ewas['sex'] == 'M'), 'age'] = 27.5
df_meta_ewas.loc[(df_meta_ewas['project_id'] == 'GSE60655') & (df_meta_ewas['sex'] == 'F'), 'age'] = 26.4
df_meta_ewas = df_meta_ewas.drop_duplicates('sample_id', keep='first')
print(df_meta_ewas.shape)

cpg_list_ewas = read_stringList_FromFile(file_cpg_ewas)

# list_test_gse = ['GSE210255', 'GSE196696', 'GSE234461',
#                  'GSE210254', 'GSE55763', 'GSE72680', 'GSE87571',
#                  'GSE61259', 'GSE61452',
#                  'GSE61450', 'GSE67024', 'GSE61257', 'GSE61453',
#                  'GSE78743', 'GSE74193', 'GSE64509', 'GSE111223', 'GSE109042']
list_test_gse = ['GSE61259', 'GSE61452',
                 'GSE61450', 'GSE67024', 'GSE61257', 'GSE61453',
                 'GSE78743', 'GSE74193', 'GSE64509', 'GSE111223', 'GSE109042',
                 'GSE50498', 'GSE48325', 'GSE61258', 'GSE72680', 'GSE105123']
df_meta_ewas_train = \
    df_meta_ewas.loc[df_meta_ewas['project_id'].apply(lambda x: x not in list_test_gse), :]
print(df_meta_ewas_train.shape)
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['obese'] != 'obese', :]
df_meta_ewas_train = df_meta_ewas_train.loc[pd.isna(df_meta_ewas_train['infection']), :]
df_meta_ewas_train = df_meta_ewas_train.loc[pd.isna(df_meta_ewas_train['smoking']), :]
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['bmi'] < 30, :]
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['sample_type'] != 'disease tissue', :]
print(df_meta_ewas_train.shape)
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['age'] != -1, :]
print(df_meta_ewas_train.shape)

df_meta_ewas_test = \
    df_meta_ewas.loc[df_meta_ewas['project_id'].apply(lambda x: x in list_test_gse), :]
print(df_meta_ewas_test.shape)


# load fudan data
fudan_data_450k = \
    np.load("/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/disease/Processed_Fudan_450k_rm10.npy",
            allow_pickle = True).item()
df_meta_fudan = \
    pd.read_csv("/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/disease/samples_Fudan_450k.txt",
                sep='\t', index_col=0)
df_meta_fudan['tissue'] = 'unknown'
df_meta_fudan['project_id'] = 'Fudan'
df_meta_fudan['bmi'] = -1
# cpg_list_450k = cpg_list_450k_age
df_meta_fudan = df_meta_fudan.loc[df_meta_fudan['sample_type'] != 'disease tissue', :]
file_cpg_450k_fudan = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/disease/cpgs_Fudan_450k_rm10.txt'
cpg_list_450k_fudan = read_stringList_FromFile(file_cpg_450k_fudan)

# load 850k data
x_npy_file = \
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/tyh/450kbind850k_model_creat/process_data/Processed_850k_450k.npy"
index_file = \
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/tyh/450kbind850k_model_creat/process_data/train_test_index.pkl"
cpg_list_train = read_stringList_FromFile(
    '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/tyh/450kbind850k_model_creat/process_data/cpgs_850k_450k_rm10.txt')

with open(index_file, 'rb') as f:
    old_train_index = pickle.load(f)
    old_test_index  = pickle.load(f)

ewas_train = np.load(x_npy_file, allow_pickle = True).item()
list_meta_col = ewas_train[old_train_index[0]]['additional'].keys()
dict_meta = {}
for col in list_meta_col:
    dict_meta[col] = []
list_sample = []
for one_sample in ewas_train.keys():
    list_sample.append(one_sample)
    for col in list_meta_col:
        dict_meta[col].append(ewas_train[one_sample]['additional'][col])
df_meta_train = pd.DataFrame(dict_meta, index=list_sample)
df_meta_train = df_meta_train.loc[df_meta_train['disease'] == 'control', :]
df_meta_850k_geo = df_meta_train.loc[df_meta_train['platform'] == '850k', :]
data_850k_geo = {}
list_850k_geo = []
for one_sample in df_meta_850k_geo['sample_id'].tolist():
    data_850k_geo[f"{one_sample}_geo"] = ewas_train[one_sample]
    list_850k_geo.append(f"{one_sample}_geo")
df_meta_850k_geo['sample_id'] = list_850k_geo
df_meta_850k_geo.index = list_850k_geo

# load 850k data
path_850k = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/850k/merge'
df_meta_850k = pd.read_csv(os.path.join(path_850k, 'meta_850k.tsv'), sep='\t', index_col=0)

data_850k = np.load(os.path.join(path_850k, 'Processed_850k.npy'),
                    allow_pickle = True).item()
cpg_list_850k = read_stringList_FromFile(os.path.join(path_850k, "cpgs_850k.txt"))
df_meta_850k_gmqn = df_meta_850k.loc[
                    [one for one in df_meta_850k.index if one in data_850k.keys()], :]

# test blood
path_test_blood = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/intermediate_data/test_blood/'
data_test_blood = np.load(
    os.path.join(path_test_blood, 'merge_blood.npy'), allow_pickle = True).item()
df_meta_test_blood = pd.read_csv(
    os.path.join(path_test_blood, 'merge_blood.tsv'), sep='\t', index_col=0)
df_meta_test_blood = df_meta_test_blood.dropna(subset='age')
list_sex = []
for sample_id in df_meta_test_blood.index:
    if isinstance(df_meta_test_blood.loc[sample_id, 'sex'], str):
        list_sex.append(df_meta_test_blood.loc[sample_id, 'sex'])
    elif isinstance(df_meta_test_blood.loc[sample_id, 'Sex'], str):
        list_sex.append(df_meta_test_blood.loc[sample_id, 'Sex'])
    elif isinstance(df_meta_test_blood.loc[sample_id, 'gender'], str):
        list_sex.append(df_meta_test_blood.loc[sample_id, 'gender'])
    else:
        list_sex.append('other')
df_meta_test_blood['sex'] = list_sex
df_meta_test_blood["sample_id"] = df_meta_test_blood.index
df_meta_test_blood = \
    df_meta_test_blood.loc[:, ["sample_id", "age", "sex", "tissue", 'project_id']]
df_meta_test_blood["bmi"] = -1
df_meta_test_blood["disease"] = "control"
df_meta_test_blood["sample_type"] = "control"
cpg_list_test_blood = read_stringList_FromFile(
    os.path.join(path_test_blood, 'merge_blood_cpgs.txt'))

#%%
# parameters
path_out = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/intermediate_data/age'
# disease_name = "type 2 diabetes"
# output_npz_file = os.path.join(path_out, "Processed_ewas_fudan_27k_DA_T2D.npy")
# folder_out = os.path.join(path_out, 'T2D_blood/')
# file_cpg_case = os.path.join(path_out, 'cpgs_ewas_fudan_T2D_27k')

output_npz_file = os.path.join(path_out, "Age_pred_all_450k_rm10_7.npy")
folder_out = os.path.join(path_out, 'Age_pred_all_450k_rm10_7/')
file_cpg_case = os.path.join(path_out, 'cpgs_Age_pred_all_450k_rm10_7.txt')
file_mat_train = os.path.join(path_out, 'mat_train_Age_pred_all_450k_rm10_7.csv')
# output_npz_file = os.path.join(path_out, "Processed_ewas_fudan_blood_450k_rm10_phenoage.npy")
# folder_out = os.path.join(path_out, 'ewas_fudan_blood_450k_rm10_phenoage/')
# file_cpg_case = os.path.join(path_out, 'cpgs_ewas_fudan_blood_450k_rm10_phenoage')
# output_npz_file = os.path.join(path_out, "Processed_ewas_fudan_blood_450k_rm10_.npy")
# folder_out = os.path.join(path_out, 'ewas_fudan_blood_450k_rm20_sd/')
# file_cpg_case = os.path.join(path_out, 'cpgs_ewas_fudan_blood_450k_rm20_sd')

dict_1 = ewas_data_450k
df_meta_1_train = df_meta_ewas_train
df_meta_1_test = df_meta_ewas_test
m2beta_1=False
cpg_list_1 = cpg_list_ewas
dict_2 = fudan_data_450k
df_meta_2 = df_meta_fudan
m2beta_2=True
cpg_list_2 = cpg_list_450k_fudan
dict_3 = data_850k_geo
df_meta_3 = df_meta_850k_geo
m2beta_3=False
cpg_list_3 = cpg_list_train
dict_4 = data_850k
df_meta_4 = df_meta_850k_gmqn
m2beta_4=False
cpg_list_4 = cpg_list_850k
dict_5 = data_test_blood
df_meta_5 = df_meta_test_blood
m2beta_5=False
cpg_list_5 = cpg_list_test_blood


# %%
# disease cpgs
set_cpg_T2D = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_T2DGlu_plus.txt"))
set_cpg_CCD = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_CCD.txt"))
set_phenoage = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_phenoage.txt"))
set_age = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_age.txt"))
cpgs_AD = set(read_stringList_FromFile(
    '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_AD.txt'))
set_PD = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_PD.txt"))
set_SCZ = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_SCZ.txt"))
set_IBD = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_IBD.txt"))
set_AID = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_AID.txt"))
set_COPD = set(read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/transfer/feature_prepare/cpgs_EWAS_COPD.txt"))
set_disease = set_cpg_T2D.union(set_cpg_CCD).union(set_phenoage).union(set_age).union(
    cpgs_AD).union(set_PD).union(set_SCZ).union(set_IBD).union(set_AID).union(set_COPD)

# train data cpgs
overlap_cpgs = set(cpg_list_1).intersection(cpg_list_2).intersection(cpg_list_3).intersection(cpg_list_4)
print(len(overlap_cpgs))
# list_index_1 = [cpg_list_1.index(cpg) for cpg in cpg_list_1 if cpg in overlap_cpgs]
list_overlap_1 = [cpg for cpg in cpg_list_1 if cpg in overlap_cpgs]
# write_stringList_2File(file_cpg_case, list_overlap_1)


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


def process_data(dict_one, df_meta_one, cpg_list_one, m2beta_one=False):

    list_ctrl_one = df_meta_one['sample_id'].tolist()

    list_feature_ctrl_one = []
    for sample_id_one in list_ctrl_one:
        sample_one = dict_one[sample_id_one]
        old_feature_one = sample_one["feature"]
        list_feature_ctrl_one.append(old_feature_one[:, np.newaxis])

    feature_ctrl_one = np.concatenate(list_feature_ctrl_one, axis=1)
    if m2beta_one:
        feature_ctrl_one = (1.0001*np.exp(feature_ctrl_one)-0.0001)/(1+np.exp(feature_ctrl_one))
    feature_ctrl_one = pd.DataFrame(feature_ctrl_one, index=cpg_list_one, columns=list_ctrl_one)
    feature_ctrl_one = feature_ctrl_one.loc[cpg_list_one, :]

    return feature_ctrl_one


# %%
feature_ctrl_1_train = process_train_data(dict_1, df_meta_1_train, cpg_list_1, list_overlap_1)

feature_ctrl_2 = \
    process_train_data(dict_2, df_meta_2, cpg_list_2, list_overlap_1, m2beta_one=m2beta_2)

feature_ctrl_3 = process_train_data(dict_3, df_meta_3, cpg_list_3, list_overlap_1)

feature_ctrl_4 = process_train_data(dict_4, df_meta_4, cpg_list_4, list_overlap_1)


# %%
# test data
additional_list = ["sample_id", "tissue", "age", "sex", "bmi", 'disease', 'sample_type', 'project_id']

list_ctrl_1_test = df_meta_1_test['sample_id'].tolist()
data_dict_ctrl_1_test = order_cpg_to_ref_fill0_2(
    cpg_list_1, list_overlap_1, dict_1, list_ctrl_1_test, df_meta_1_test, additional_list)

# data 5
data_dict_ctrl_5 = order_cpg_to_ref_fill0_2(
    cpg_list_5, list_overlap_1, dict_5, df_meta_5['sample_id'].tolist(), df_meta_5, additional_list)

# %%
# transform into dict
save_dict = {}

# train data
X_np_train = pd.concat(
    [feature_ctrl_1_train.T, feature_ctrl_2.T, feature_ctrl_3.T, feature_ctrl_4.T])
# X_np_train = pd.concat(
#     [feature_ctrl_1_train.T, feature_ctrl_2.T, feature_ctrl_3.T, feature_ctrl_4.T])
y_pd_train = pd.concat([df_meta_1_train, df_meta_2, df_meta_3, df_meta_4])
y_pd_train = y_pd_train.dropna(subset='age')
y_pd_train = y_pd_train.loc[y_pd_train['age'] >= 0, :]
y_pd_train["sex"] = y_pd_train["sex"].fillna("other")
sel_train_samples = random.sample(y_pd_train['sample_id'].tolist(), (11313-2549))
X_np_train = X_np_train.loc[sel_train_samples, :]
y_pd_train = y_pd_train.loc[sel_train_samples, :]
X_np_train.index = [f"train_{i}" for i in range(len(sel_train_samples))]
y_pd_train.index = [f"train_{i}" for i in range(len(sel_train_samples))]

additional_list = ["age", "sex"]
for row_number in range(len(y_pd_train)):
    key_rename = y_pd_train.index[row_number]
    feature_np = np.array(X_np_train.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd_train.iloc[row_number]["age"].astype(np.float32)])
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd_train.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

# validation data
feature_val_1 = process_data(data_dict_ctrl_1_test, df_meta_1_test, list_overlap_1)
feature_val_2 = process_data(data_dict_ctrl_5, df_meta_5, list_overlap_1)
X_np_val = pd.concat([feature_val_1.T, feature_val_2.T], axis=0)
y_pd_val = pd.concat([df_meta_1_test, df_meta_5])
y_pd_val = y_pd_val.dropna(subset='age')
y_pd_val = y_pd_val.loc[y_pd_val['age'] >= 0, :]
y_pd_val["sex"] = y_pd_val["sex"].fillna("other")
sel_val_samples = random.sample(y_pd_val['sample_id'].tolist(), 2549)
X_np_val = X_np_val.loc[sel_val_samples, :]
y_pd_val = y_pd_val.loc[sel_val_samples, :]
X_np_val.index = [f"val_{i}" for i in range(len(sel_val_samples))]
y_pd_val.index = [f"val_{i}" for i in range(len(sel_val_samples))]

additional_list = ["age", "sex"]
for row_number in range(len(y_pd_val)):
    key_rename = y_pd_val.index[row_number]
    feature_np = np.array(X_np_val.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd_val.iloc[row_number]["age"].astype(np.float32)])
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd_val.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

# %%
# save data
output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/models/train_data/epiAge_traindata.npz"
np.savez(output_npz_file,
         data=save_dict, cpgs=list_overlap_1,
         train_index=list(y_pd_train.index), val_index=list(y_pd_val.index))

output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/models/train_data/epiAge_traindata_compress.npz"
np.savez_compressed(output_npz_file,
                    data=save_dict, cpgs=list_overlap_1,
                    train_index=list(y_pd_train.index), val_index=list(y_pd_val.index))
