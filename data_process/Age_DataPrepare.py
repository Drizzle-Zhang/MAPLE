# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @time: 2024/3/21 11:34

import numpy as np
import pandas as pd
import random
import pickle
import os
os.chdir('/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/zhangyu001/bioage/code')
from utils.file_utils import read_stringList_FromFile
from utils.common_utils import get_new_index, order_cpg_to_ref_fill0_2


# %%
# load all ewas datahub data
path_ewas = '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/intermediate_data/merge_ewas_datahub'
npz_file_ewas = os.path.join(path_ewas, "meth_beta_all_450k_rm10.npy")
file_cpg_ewas = os.path.join(path_ewas, 'cpgs_all_450k_rm10.txt')
file_meta_ewas = os.path.join(path_ewas, 'meta_all_450k.tsv')
ewas_data_450k = np.load(npz_file_ewas, allow_pickle = True).item()
df_meta_ewas = pd.read_csv(file_meta_ewas, sep = "\t", index_col=0)
cpg_list_ewas = read_stringList_FromFile(file_cpg_ewas)

list_test_gse = \
    ['GSE64509', 'GSE72680', 'GSE109042', 'GSE78743', 'GSE78743', 'GSE74193', 'GSE61452',
     'GSE61259', 'GSE50498', 'GSE61450', 'GSE61257', 'GSE61258', 'GSE48325', 'GSE61453',
     'GSE196696', 'GSE210255', 'GSE55763', 'GSE210254', 'GSE87571', 'GSE111223']

df_meta_ewas_train = \
    df_meta_ewas.loc[df_meta_ewas['project_id'].apply(lambda x: x not in list_test_gse), :]
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['obese'] != 'obese', :]
df_meta_ewas_train = df_meta_ewas_train.loc[pd.isna(df_meta_ewas_train['infection']), :]
df_meta_ewas_train = df_meta_ewas_train.loc[pd.isna(df_meta_ewas_train['smoking']), :]
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['bmi'] < 30, :]
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['sample_type'] != 'disease tissue', :]
df_meta_ewas_train = df_meta_ewas_train.loc[df_meta_ewas_train['age'] > -1, :]

df_meta_ewas_test = \
    df_meta_ewas.loc[df_meta_ewas['project_id'].apply(lambda x: x in list_test_gse), :]


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
dict_1 = ewas_data_450k
df_meta_1_train = df_meta_ewas_train
df_meta_1_test = df_meta_ewas_test
m2beta_1=False
cpg_list_1 = cpg_list_ewas
dict_4 = data_850k
df_meta_4 = df_meta_850k_gmqn
m2beta_4=False
cpg_list_4 = cpg_list_850k
dict_5 = data_test_blood
df_meta_5 = df_meta_test_blood
m2beta_5=False
cpg_list_5 = cpg_list_test_blood


# %%
# train data cpgs
overlap_cpgs = set(cpg_list_1).intersection(cpg_list_4)
print(len(overlap_cpgs))
list_overlap_1 = [cpg for cpg in cpg_list_1 if cpg in overlap_cpgs]


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
X_np_merge = pd.concat([feature_ctrl_1_train.T, feature_ctrl_4.T])
y_pd_merge = pd.concat([df_meta_1_train, df_meta_4])
y_pd_merge = y_pd_merge.dropna(subset='age')
y_pd_merge = y_pd_merge.loc[y_pd_merge['age'] >= 0, :]
y_pd_merge["sex"] = y_pd_merge["sex"].fillna("other")

train_dataset = np.unique(X_np_merge['project_id'])
train_use = set(random.sample(train_dataset, int(len(train_dataset)*0.8)))
val_use = set(train_dataset).difference(train_use)
y_pd_train = y_pd_merge.loc[y_pd_merge['project_id'].apply(lambda x: x in train_use), :]
y_pd_val = y_pd_merge.loc[y_pd_merge['project_id'].apply(lambda x: x in val_use), :]

additional_list = ["age", "sex"]
for row_number in range(len(y_pd_train)):
    key_rename = y_pd_train.index[row_number]
    feature_np = np.array(X_np_merge.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd_train.iloc[row_number]["age"].astype(np.float32)])
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        value = y_pd_train.iloc[row_number][additional_name]
        additional[additional_name] = value

    save_dict[key_rename]["additional"] = additional

# test data
feature_test_1 = process_data(data_dict_ctrl_1_test, df_meta_1_test, list_overlap_1)
feature_test_2 = process_data(data_dict_ctrl_5, df_meta_5, list_overlap_1)
X_np_test = pd.concat([feature_test_1.T, feature_test_2.T], axis=0)
y_pd_test = pd.concat([df_meta_1_test, df_meta_5])
y_pd_test = y_pd_test.dropna(subset='age')
y_pd_test = y_pd_test.loc[y_pd_test['age'] >= 0, :]
y_pd_test["sex"] = y_pd_test["sex"].fillna("other")

additional_list = ["age", "sex"]
for row_number in range(len(y_pd_test)):
    key_rename = y_pd_test.index[row_number]
    feature_np = np.array(X_np_test.loc[key_rename, :]).astype(np.float32)

    save_dict[key_rename] = {}
    save_dict[key_rename]["feature"] = feature_np
    age = np.asarray([y_pd_test.iloc[row_number]["age"].astype(np.float32)])
    save_dict[key_rename]["target"] = age

    additional = {}
    for additional_name in additional_list:
        testue = y_pd_test.iloc[row_number][additional_name]
        additional[additional_name] = testue

    save_dict[key_rename]["additional"] = additional

# %%
# save data
output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/train_data/epiAge_traindata.npz"
np.savez(output_npz_file,
         data=save_dict, cpgs=list_overlap_1,
         train_index=list(y_pd_train.index), val_index=list(y_pd_val.index))

output_npz_file = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/zhangyu/maple/test_data/epiAge_testdata.npz"
np.savez(output_npz_file, data=save_dict, cpgs=list_overlap_1, test_index=list(y_pd_test.index))
