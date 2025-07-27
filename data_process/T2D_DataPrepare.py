# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: DA_PD.py
# @time: 2023/8/15 11:34

import numpy as np
import pandas as pd
import random
import os
os.chdir('/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/zhangyu001/bioage/code')
from utils.file_utils import read_stringList_FromFile
from utils.common_utils import data_augmentation, order_cpg_to_ref_fill0_2, get_new_index


#%%
# keep samples
keep_tissue = \
    {'whole blood', 'peripheral blood mononuclear cell', 'CD4+ T cell', 'CD8+ T cell',
     'CD14+ monocyte', 'leukocyte', 'lymphocyte', 'cord blood'}
list_test_gse = \
    ['GSE72680', 'GSE62003', 'GSE199700',
     'GSE76285', 'GSE197881', 'GSE196696', 'GSE87571', 'GSE210255', 'GSE55763']

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
df_meta_disease_ctrl = df_meta_disease_ctrl.loc[df_meta_disease_ctrl['project_id'].apply(lambda x: x not in list_test_gse), :]
df_meta_disease_test = \
    df_meta_disease.loc[df_meta_disease['project_id'].apply(lambda x: x in list_test_gse), :]
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
df_meta_age_ctrl = df_meta_age.loc[df_meta_age['project_id'].apply(lambda x: x not in list_test_gse), :]
df_meta_age_test = \
    df_meta_age.loc[df_meta_age['project_id'].apply(lambda x: x in list_test_gse), :]
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

# GSE210255
GSE210255_850k = np.load(
    '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/GENOA/GSE210255_process/Processed_all_850k.npy',
    allow_pickle = True).item()
cpg_list_GSE210255 = read_stringList_FromFile(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/GENOA/GSE210255_process/GSE210255_cpgs_850k.txt")
df_meta_GSE210255 = pd.read_csv(
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/GENOA/GSE210255_process/GSE210255_meta.tsv", index_col=0, sep='\t')
# df_meta_GSE210255 = df_meta_GSE210255.loc[:, ["sample_id", "age", "sex", 'project_id']]
df_meta_GSE210255["bmi"] = -1
df_meta_GSE210255["disease"] = 'control'
df_meta_GSE210255["sample_type"] = 'control'
df_meta_GSE196696["platform"] = '850k'


# GSE207927
path_850k = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/850k'
path_GSE207927_process = os.path.join(path_850k, 'GSE207927_process')
file_GSE207927_cpgs = os.path.join(path_GSE207927_process, 'GSE207927_cpgs_850k.txt')
file_GSE207927_npy = os.path.join(path_GSE207927_process, 'Processed_all_850k.npy')
data_GSE207927 = np.load(file_GSE207927_npy, allow_pickle = True).item()
cpg_list_GSE207927 = read_stringList_FromFile(file_GSE207927_cpgs)
df_meta_GSE207927 = pd.read_csv(os.path.join(path_GSE207927_process, 'GSE207927_meta.tsv'), sep='\t', index_col=0)
df_meta_GSE207927['disease'] = 'control'
df_meta_GSE207927['sample_type'] = 'control'
df_meta_GSE207927['project_id'] = 'GSE207927'


# GSE55763
path_LOLIPOP = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/LOLIPOP'
GSE55763_450k = np.load(os.path.join(path_LOLIPOP, 'Processed_all_GMQN.npy'),allow_pickle = True).item()
cpg_list_GSE55763 = read_stringList_FromFile(os.path.join(path_LOLIPOP, 'GSE55763_cpgs_GMQN.txt'))
df_meta_GSE55763 = pd.read_csv(os.path.join(path_LOLIPOP, 'GSE55763_meta.txt'), index_col=0)
df_meta_GSE55763 = df_meta_GSE55763.loc[:, ["sample_id", "age", "sex", 'project_id']]
df_meta_GSE55763["tissue"] = 'whole blood'
df_meta_GSE55763["bmi"] = -1
df_meta_GSE55763["disease"] = 'control'
df_meta_GSE55763["sample_type"] = 'control'
df_meta_GSE55763["project_id"] = 'GSE55763'


# blood bmi data
path_out_bmi = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS_process/bmi/450k_rm10"
file_meta_bmi = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/Data/EWAS/bmi/sample_bmi.txt"
file_cpg_450k_bmi = os.path.join(path_out_bmi, 'cpgs_list.txt')
file_npz_bmi = os.path.join(path_out_bmi, "Processed.npy")
ewas_bmi_data_450k = np.load(file_npz_bmi, allow_pickle = True).item()
df_meta_bmi = pd.read_csv(file_meta_bmi, sep=' ', index_col=0)
list_test_gse = ['GSE72680', 'GSE105123']
df_meta_bmi_blood = df_meta_bmi.loc[df_meta_bmi['project_id'].apply(lambda x: x in list_test_gse), :]
cpg_list_450k_bmi = read_stringList_FromFile(file_cpg_450k_bmi)

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
disease_name = "type 2 diabetes"
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

# %%
overlap_cpgs = \
    set(cpg_list_2).intersection(set(cpg_list_1)).intersection(set(cpg_list_3)).intersection(
        set(cpg_list_4)).intersection(cpg_list_GSE197881).intersection(
        cpg_list_GSE210255).intersection(cpg_list_GSE207927).intersection(cpg_list_GSE199700)
print(len(overlap_cpgs))
list_overlap_1 = [cpg for cpg in cpg_list_1 if cpg in overlap_cpgs]
list_index_1 = get_new_index(cpg_list_1, list_overlap_1)


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
# diabetes
df_mat_t2d = df_mat_GSE197881.T.loc[:, list_overlap_1]
df_mat_pd = df_mat_GSE199700.T.loc[:, list_overlap_1]

# %%
meta_ctrl_2 = df_meta_2.loc[df_meta_2['disease'] == "control", :]
feature_ctrl_2 = process_train_data(dict_2, meta_ctrl_2, cpg_list_2, list_overlap_1)

feature_ctrl_3 = process_train_data(dict_3, df_meta_3, cpg_list_3, list_overlap_1)

feature_ctrl_4 = process_train_data(dict_4, df_meta_4, cpg_list_4, list_overlap_1)

# test data
additional_list = ["sample_id", "age", "sex", "bmi", 'disease', 'sample_type', 'project_id']

list_GSE210255 = df_meta_GSE210255['sample_id'].tolist()
GSE210255_850k = order_cpg_to_ref_fill0_2(
    cpg_list_GSE210255, list_overlap_1, GSE210255_850k, list_GSE210255, df_meta_GSE210255, additional_list)
feature_GSE210255 = process_data(GSE210255_850k, df_meta_GSE210255, list_overlap_1)

list_GSE207927 = df_meta_GSE207927['sample_id'].tolist()
data_GSE207927 = order_cpg_to_ref_fill0_2(
    cpg_list_GSE207927, list_overlap_1, data_GSE207927, list_GSE207927, df_meta_GSE207927, additional_list)
feature_GSE207927 = process_data(data_GSE207927, df_meta_GSE207927, list_overlap_1)

list_GSE55763 = df_meta_GSE55763['sample_id'].tolist()
GSE55763_450k = order_cpg_to_ref_fill0_2(
    cpg_list_GSE55763, list_overlap_1, GSE55763_450k, list_GSE55763, df_meta_GSE55763, additional_list)
feature_GSE55763 = process_data(GSE55763_450k, df_meta_GSE55763, list_overlap_1)

feature_ctrl_5 = process_data(ewas_bmi_data_450k, df_meta_bmi_blood, cpg_list_450k_bmi, list_overlap_1)

feature_ctrl_6 = process_data(dict_1, df_meta_disease_test, cpg_list_1, list_overlap_1)

feature_ctrl_7 = process_data(dict_3, df_meta_age_test, cpg_list_3, list_overlap_1)


#%%
# read T2D score from Nature aging
path_pretrain = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/intermediate_data/T2D/pretrain_data_blood_450k_850k_T2D'
df_mrs_correct = \
    pd.read_csv(os.path.join(path_pretrain, 'meta_corrected.tsv'), sep='\t', index_col=0)
df_mrs_correct["age"] = df_mrs_correct["age"].astype(np.float32)
df_mrs_correct["T2D_risk"] = df_mrs_correct["T2D_risk"].astype(np.float32)


# %%
# data augmentation
# case
list_case_1_train = random.sample(list_case_1, 29)
list_case_1_test = [one for one in list_case_1 if one not in list_case_1_train]
df_meta_1_case_train = df_mrs_correct.loc[list_case_1, :]
aug_case_1, meta_aug_case_1 = \
    data_augmentation(feature_case_1, df_meta_1_case_train, 70, by_project=False)


# %%
# transform into dict
save_dict = {}

# train data
X_np_merge = pd.concat([aug_case_1, feature_ctrl_2.T, feature_ctrl_3.T, feature_ctrl_5.T])
y_pd_merge = pd.concat([meta_aug_case_1,
                        df_mrs_correct.loc[feature_ctrl_2.columns, :],
                        df_mrs_correct.loc[[one for one in feature_ctrl_3.columns if one in df_mrs_correct.index], :],
                        df_mrs_correct.loc[feature_ctrl_5.columns, :]])
y_pd_merge["sex"] = y_pd_merge["sex"].fillna("other")
y_pd_merge["age"] = y_pd_merge["age"].astype(np.float32)
y_pd_merge["T2D_risk"] = y_pd_merge["T2D_risk"].fillna(-10000)

train_dataset = np.unique(X_np_merge['project_id'])
train_use = set(random.sample(train_dataset, int(len(train_dataset)*0.8)))
val_use = set(train_dataset).difference(train_use)
y_pd_train = y_pd_merge.loc[y_pd_merge['project_id'].apply(lambda x: x in train_use), :]
y_pd_val = y_pd_merge.loc[y_pd_merge['project_id'].apply(lambda x: x in val_use), :]
pretrain_index = y_pd_merge.loc[y_pd_merge['T2D_risk'] > -10, :].index

additional_list = ["age", "sex", 'sample_type', 'project_id']
type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
for row_number in range(len(y_pd_merge)):
    key_rename = y_pd_merge.index[row_number]
    feature_np = np.array(X_np_merge.loc[key_rename, :]).astype(np.float32)
    mrs = np.asarray([np.array(y_pd_merge.iloc[row_number]['T2D_risk']).astype(np.float32)])

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    save_dict[key_rename]["target"] = mrs

    additional = {}
    for additional_name in additional_list:
        value = y_pd_merge.iloc[row_number][additional_name]
        additional[additional_name] = value

    additional["type_index"] = type2index[additional["sample_type"]]

    save_dict[key_rename]["additional"] = additional


# test data
X_np_test = pd.concat([feature_case_1.T.loc[list_case_1_test, :],
                       feature_ctrl_1.T, feature_ctrl_4.T, feature_GSE210255.T,
                       feature_GSE207927.T, feature_GSE55763.T])
y_pd_test = pd.concat([
    df_mrs_correct.loc[list_case_1_test, :],
    df_mrs_correct.loc[[one for one in feature_ctrl_1.columns if one in df_mrs_correct.index], :],
    df_mrs_correct.loc[feature_ctrl_4.columns, :],
    df_meta_GSE210255, df_meta_GSE207927, df_meta_GSE55763
])
X_np_test = pd.concat([X_np_test, feature_case_1.T.loc[list_case_1_test, :], feature_sir_1.T, df_mat_t2d])
y_pd_test = pd.concat([y_pd_test, df_mrs_correct.loc[list_case_1_test, :],
                       meta_sir_1, df_mrs_correct.loc[df_meta_GSE197881.index, :]])
y_pd_test["sex"] = y_pd_test["sex"].fillna("other")

additional_list = ["age", "sex", 'sample_type', 'project_id']
type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
for row_number in range(len(y_pd_test)):
    key_rename = y_pd_test.index[row_number]
    feature_np = np.array(X_np_test.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    mrs = np.asarray([np.array(y_pd_test.iloc[row_number]["T2D_risk"]).astype(np.float32)])
    save_dict[key_rename]["target"] = mrs

    additional = {}
    for additional_name in additional_list:
        testue = y_pd_test.iloc[row_number][additional_name]
        additional[additional_name] = testue

    additional["type_index"] = type2index[additional["sample_type"]]
    additional["project_id"] = "project_test"
    save_dict[key_rename]["additional"] = additional

# %%
# save data
output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/models/train_data/T2D_traindata.npz"
np.savez(output_npz_file,
         data=save_dict, cpgs=list_overlap_1,
         train_index=list(y_pd_train.index), val_index=list(y_pd_val.index),
         pretrain_index=pretrain_index)

output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/test_data/T2D_testdata.npz"
np.savez(output_npz_file, data=save_dict, cpgs=list_overlap_1, test_index=list(y_pd_test.index))
