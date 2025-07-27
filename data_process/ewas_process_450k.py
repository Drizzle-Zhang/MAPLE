
#%%
import os
import numpy as np
import pandas as pd
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File, FileUtils
from sklearn.impute import SimpleImputer


def read_single_csv(input_path, sep=','):
    df_chunk = pd.read_csv(input_path, sep=sep, chunksize=1000, index_col=0)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df = pd.concat(res_chunk)
    return res_df


def impute_methy(df_methy, df_meta_in, by_project=True):
    df_methy = df_methy.T
    if by_project:
        project_ids = np.unique(df_meta_in['project_id'])
        list_imp = []
        list_samples = []
        for one_id in project_ids:
            df_meta_in_sub = df_meta_in.loc[df_meta_in['project_id'] == one_id, :]
            df_methy_sub = df_methy.loc[df_meta_in_sub['sample_id'].tolist(), :]
            imp_mean = SimpleImputer(keep_empty_features=True)
            df_methy_imp = imp_mean.fit_transform(df_methy_sub)
            list_imp.append(df_methy_imp)
            list_samples.extend(df_meta_in_sub['sample_id'].tolist())
        x_imp = np.concatenate(list_imp, axis=0)
    else:
        imp_mean = SimpleImputer()
        x_imp = imp_mean.fit_transform(df_methy)
        list_samples = list(df_methy.index)

    return x_imp, list_samples


def generate_npy(file_meta, file_methy, file_cpg_450k, output_npz_file):
    # EWAS dataset
    df_meta_ewas = pd.read_csv(file_meta, sep = " ")
    df_methy_ewas = read_single_csv(file_methy, sep='\t')

    df_meta_ewas = \
        df_meta_ewas.loc[:, ["sample_id", "tissue", "age", "sex", 'disease', 'sample_type', 'project_id']]

    # missing value
    missing_rates = pd.Series(np.sum(np.isnan(df_methy_ewas), axis=1) / df_methy_ewas.shape[1],
                              index=df_methy_ewas.index)
    cpgs_rm20 = list(missing_rates.loc[missing_rates < 0.2].index)
    print(f"Num of CpG sites: {len(cpgs_rm20)}")

    # overlap methy site
    write_stringList_2File(file_cpg_450k, cpgs_rm20)
    df_ewas_2 = df_methy_ewas.loc[cpgs_rm20, :]
    # impute
    imp_ewas, imp_samples_ewas = impute_methy(df_ewas_2, df_meta_ewas)
    sample_statistics = df_meta_ewas.loc[imp_samples_ewas, :]
    X_np = imp_ewas.astype(np.float32)

    additional_list = ["sample_id", "tissue", "age", "sex", 'disease', 'sample_type']
    y_pd = sample_statistics

    y_pd["sex"] = y_pd["sex"].fillna("other")
    y_pd["disease"] = y_pd["disease"].fillna("control")
    y_pd_count = y_pd.groupby(["tissue"], as_index = False).count()
    y_pd_count = y_pd_count.sort_values(by = "sample_id", ascending= False)
    tissue2index = {}
    index2tissue = {}
    for index, tissue in enumerate(list(y_pd_count["tissue"])):
        tissue2index[tissue] = index
        index2tissue[index] = tissue

    sex2index = {"M":0, "F":1, "other":2}
    type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
    all_keys = []
    save_dict = {}
    target_name = "age"
    for row_number in range(len(y_pd)):
        key_rename = y_pd.iloc[row_number].sample_id
        all_keys.append(key_rename)
        feature_np = X_np[row_number, :]

        save_dict[key_rename] = {}
        save_dict[key_rename]["feature"] = feature_np
        age = np.asarray([y_pd.iloc[row_number][target_name].astype(np.float32)])
        save_dict[key_rename]["target"] = age

        additional = {}
        for additional_name in additional_list:
            value = y_pd.iloc[row_number][additional_name]
            additional[additional_name] = value

        additional["sex_index"] = sex2index[additional["sex"]]
        additional["tissue_index"] = tissue2index[additional["tissue"]]
        additional["type_index"] = type2index[additional["sample_type"]]

        save_dict[key_rename]["additional"] = additional

    np.save(output_npz_file, save_dict)

    return


def generate_npy_2(file_meta, file_methy, file_cpg_450k, output_npz_file,
                   disease=True, tissue=True, rm_na_ratio=None):
    # EWAS dataset
    df_meta_ewas = pd.read_csv(file_meta, sep = " ")
    df_methy_ewas = read_single_csv(file_methy, sep='\t')

    if disease:
        df_meta_ewas = \
            df_meta_ewas.loc[:, ["sample_id", "tissue", "age", "sex", 'disease', 'sample_type', 'project_id']]
    else:
        df_meta_ewas = \
            df_meta_ewas.loc[:, ["sample_id", "tissue", "age", "sex", 'sample_type', 'project_id']]

    # missing value
    if rm_na_ratio is None:
        cpgs_rm20 = list(df_methy_ewas.index)
    else:
        missing_rates = pd.Series(np.sum(np.isnan(df_methy_ewas), axis=1) / df_methy_ewas.shape[1],
                                  index=df_methy_ewas.index)
        cpgs_rm20 = list(missing_rates.loc[missing_rates < rm_na_ratio].index)
    print(f"Num of CpG sites: {len(cpgs_rm20)}")

    # overlap methy site
    write_stringList_2File(file_cpg_450k, cpgs_rm20)
    df_ewas_2 = df_methy_ewas.loc[cpgs_rm20, :]

    # impute
    df_meta_ewas['gse_id'] = df_meta_ewas['project_id'].tolist()
    if disease and tissue:
        df_meta_ewas['project_id'] = \
            df_meta_ewas.apply(lambda x: f"{x['gse_id']}_{x['tissue']}_{x['disease']}", axis=1)
    elif disease:
        df_meta_ewas['project_id'] = \
            df_meta_ewas.apply(lambda x: f"{x['gse_id']}_{x['disease']}", axis=1)
    elif tissue:
        df_meta_ewas['project_id'] = \
            df_meta_ewas.apply(lambda x: f"{x['gse_id']}_{x['tissue']}", axis=1)
    imp_ewas, imp_samples_ewas = impute_methy(df_ewas_2, df_meta_ewas)
    sample_statistics = df_meta_ewas.loc[imp_samples_ewas, :]
    X_np = imp_ewas.astype(np.float32)

    additional_list = ["sample_id", "tissue", "age", "sex", 'project_id', 'sample_type']
    y_pd = sample_statistics

    y_pd["sex"] = y_pd["sex"].fillna("other")
    y_pd["age"] = y_pd["age"].fillna(-10)
    y_pd_count = y_pd.groupby(["tissue"], as_index = False).count()
    y_pd_count = y_pd_count.sort_values(by = "sample_id", ascending= False)
    tissue2index = {}
    index2tissue = {}
    for index, tissue in enumerate(list(y_pd_count["tissue"])):
        tissue2index[tissue] = index
        index2tissue[index] = tissue

    sex2index = {"M":0, "m":0, "F":1, "other":2}
    type2index = {"control":0, "disease tissue":1, "adjacent normal":0}
    all_keys = []
    save_dict = {}
    target_name = "age"
    for row_number in range(len(y_pd)):
        key_rename = y_pd.iloc[row_number].sample_id
        all_keys.append(key_rename)
        feature_np = X_np[row_number, :]

        save_dict[key_rename] = {}
        save_dict[key_rename]["feature"] = feature_np
        age = np.asarray([y_pd.iloc[row_number][target_name].astype(np.float32)])
        save_dict[key_rename]["target"] = age

        additional = {}
        for additional_name in additional_list:
            value = y_pd.iloc[row_number][additional_name]
            additional[additional_name] = value

        additional["sex_index"] = sex2index[additional["sex"]]
        additional["tissue_index"] = tissue2index[additional["tissue"]]
        additional["type_index"] = type2index[additional["sample_type"]]

        save_dict[key_rename]["additional"] = additional

    np.save(output_npz_file, save_dict)

    return


if __name__ == "__main__":
    pass

    #%%
    # remove CpGs whose missing rate is more than 10%
    #%%
    # EWAS disease dataset
    path_disease = '/home/zhangyu/mnt_path/Data/EWAS/disease/'
    file_meta_disease = os.path.join(path_disease, "sample_disease.txt")
    file_methy_disease = os.path.join(path_disease, "disease_methylation_v1.txt")
    path_out_disease = "/home/zhangyu/mnt_path/Data/EWAS_process/disease/450k_rm10"
    FileUtils.makedir(path_out_disease)
    file_cpg_450k_disease = os.path.join(path_out_disease, 'cpgs_list.txt')
    file_npz_disease = os.path.join(path_out_disease, "Processed.npy")
    generate_npy_2(file_meta_disease, file_methy_disease, file_cpg_450k_disease, file_npz_disease,
                   rm_na_ratio=0.1)

    #%%
    # EWAS age dataset
    path_age = '/home/zhangyu/mnt_path/Data/EWAS/age/'
    file_meta_age = os.path.join(path_age, "sample_age.txt")
    file_methy_age = os.path.join(path_age, "age_methylation_v1.txt")
    path_out_age = "/home/zhangyu/mnt_path/Data/EWAS_process/age/450k_rm10"
    FileUtils.makedir(path_out_age)
    file_cpg_450k_age = os.path.join(path_out_age, 'cpgs_list.txt')
    file_npz_age = os.path.join(path_out_age, "Processed.npy")
    generate_npy_2(file_meta_age, file_methy_age, file_cpg_450k_age, file_npz_age,
                   disease=False, rm_na_ratio=0.1)

    #%%
    # EWAS tissue dataset
    path_tissue = '/home/zhangyu/mnt_path/Data/EWAS/tissue/'
    file_meta_tissue = os.path.join(path_tissue, "sample_tissue.txt")
    file_methy_tissue = os.path.join(path_tissue, "tissue_methylation_v1.txt")
    path_out_tissue = "/home/zhangyu/mnt_path/Data/EWAS_process/tissue/450k_rm10"
    FileUtils.makedir(path_out_tissue)
    file_cpg_450k_tissue = os.path.join(path_out_tissue, 'cpgs_list.txt')
    file_npz_tissue = os.path.join(path_out_tissue, "Processed.npy")
    generate_npy_2(file_meta_tissue, file_methy_tissue, file_cpg_450k_tissue, file_npz_tissue,
                   rm_na_ratio=0.1)

    #%%
    # EWAS brain dataset
    path_brain = '/home/zhangyu/mnt_path/Data/EWAS/brain/'
    file_meta_brain = os.path.join(path_brain, "sample_brain.txt")
    file_methy_brain = os.path.join(path_brain, "brain_methylation_v1.txt")
    path_out_brain = "/home/zhangyu/mnt_path/Data/EWAS_process/brain/450k_rm10"
    FileUtils.makedir(path_out_brain)
    file_cpg_450k_brain = os.path.join(path_out_brain, 'cpgs_list.txt')
    file_npz_brain = os.path.join(path_out_brain, "Processed.npy")
    generate_npy_2(file_meta_brain, file_methy_brain, file_cpg_450k_brain, file_npz_brain,
                   disease=False, rm_na_ratio=0.1)

    #%%
    # EWAS blood dataset
    path_blood = '/home/zhangyu/mnt_path/Data/EWAS/blood/'
    file_meta_blood = os.path.join(path_blood, "sample_blood.txt")
    file_methy_blood = os.path.join(path_blood, "blood_methylation_v1.txt")
    path_out_blood = "/home/zhangyu/mnt_path/Data/EWAS_process/blood/450k_rm10"
    FileUtils.makedir(path_out_blood)
    file_cpg_450k_blood = os.path.join(path_out_blood, 'cpgs_list.txt')
    file_npz_blood = os.path.join(path_out_blood, "Processed.npy")
    generate_npy_2(file_meta_blood, file_methy_blood, file_cpg_450k_blood, file_npz_blood,
                   disease=False, rm_na_ratio=0.1)

    #%%
    # EWAS bmi dataset
    path_bmi = '/home/zhangyu/mnt_path/Data/EWAS/bmi/'
    file_meta_bmi = "/home/zhangyu/mnt_path/Data/EWAS/bmi/sample_bmi.txt"
    file_methy_bmi = "/home/zhangyu/mnt_path/Data/EWAS/bmi/bmi_methylation_v1.txt"
    path_out_bmi = "/home/zhangyu/mnt_path/Data/EWAS_process/bmi/450k_rm10"
    FileUtils.makedir(path_out_bmi)
    file_cpg_450k_bmi = os.path.join(path_out_bmi, 'cpgs_list.txt')
    file_npz_bmi = os.path.join(path_out_bmi, "Processed.npy")
    generate_npy_2(file_meta_bmi, file_methy_bmi, file_cpg_450k_bmi, file_npz_bmi, rm_na_ratio=0.1)

    #%%
    # EWAS sex dataset
    path_sex = '/home/zhangyu/mnt_path/Data/EWAS/sex/'
    file_meta_sex = os.path.join(path_sex, "sample_sex.txt")
    file_methy_sex = os.path.join(path_sex, "sex_methylation_v1.txt")
    path_out_sex = "/home/zhangyu/mnt_path/Data/EWAS_process/sex/450k_rm10"
    FileUtils.makedir(path_out_sex)
    file_cpg_450k_sex = os.path.join(path_out_sex, 'cpgs_list.txt')
    file_npz_sex = os.path.join(path_out_sex, "Processed.npy")
    generate_npy_2(file_meta_sex, file_methy_sex, file_cpg_450k_sex, file_npz_sex, rm_na_ratio=0.1)


    #%%
    # reserve all CpGs
    #%%
    # EWAS tissue dataset
    path_tissue = '/home/zhangyu/mnt_path/Data/EWAS/tissue/'
    file_meta_tissue = os.path.join(path_tissue, "sample_tissue.txt")
    file_methy_tissue = os.path.join(path_tissue, "tissue_methylation_v1.txt")
    path_out_tissue = "/home/zhangyu/mnt_path/Data/EWAS_process/tissue/450k"
    FileUtils.makedir(path_out_tissue)
    file_cpg_450k_tissue = os.path.join(path_out_tissue, 'cpgs_list.txt')
    file_npz_tissue = os.path.join(path_out_tissue, "Processed_tissue_450k.npy")
    generate_npy_2(file_meta_tissue, file_methy_tissue, file_cpg_450k_tissue, file_npz_tissue,
                   disease=False)

    #%%
    # EWAS disease dataset
    path_disease = '/home/zhangyu/mnt_path/Data/EWAS/disease/'
    file_meta_disease = os.path.join(path_disease, "sample_disease.txt")
    file_methy_disease = os.path.join(path_disease, "disease_methylation_v1.txt")
    path_out_disease = "/home/zhangyu/mnt_path/Data/EWAS_process/disease/450k"
    FileUtils.makedir(path_out_disease)
    file_cpg_450k_disease = os.path.join(path_out_disease, 'cpgs_list.txt')
    file_npz_disease = os.path.join(path_out_disease, "Processed_disease_450k.npy")
    generate_npy_2(file_meta_disease, file_methy_disease, file_cpg_450k_disease, file_npz_disease)

    #%%
    # EWAS brain dataset
    path_brain = '/home/zhangyu/mnt_path/Data/EWAS/brain/'
    file_meta_brain = os.path.join(path_brain, "sample_brain.txt")
    file_methy_brain = os.path.join(path_brain, "brain_methylation_v1.txt")
    path_out_brain = "/home/zhangyu/mnt_path/Data/EWAS_process/brain/450k"
    FileUtils.makedir(path_out_brain)
    file_cpg_450k_brain = os.path.join(path_out_brain, 'cpgs_list.txt')
    file_npz_brain = os.path.join(path_out_brain, "Processed_brain_450k.npy")
    generate_npy_2(file_meta_brain, file_methy_brain, file_cpg_450k_brain, file_npz_brain,
                   disease=False)

    #%%
    # EWAS blood dataset
    path_blood = '/home/zhangyu/mnt_path/Data/EWAS/blood/'
    file_meta_blood = os.path.join(path_blood, "sample_blood.txt")
    file_methy_blood = os.path.join(path_blood, "blood_methylation_v1.txt")
    path_out_blood = "/home/zhangyu/mnt_path/Data/EWAS_process/blood/450k"
    FileUtils.makedir(path_out_blood)
    file_cpg_450k_blood = os.path.join(path_out_blood, 'cpgs_list.txt')
    file_npz_blood = os.path.join(path_out_blood, "Processed_blood_450k.npy")
    generate_npy_2(file_meta_blood, file_methy_blood, file_cpg_450k_blood, file_npz_blood,
                   disease=False)

    #%%
    # EWAS age dataset
    path_age = '/home/zhangyu/mnt_path/Data/EWAS/age/'
    file_meta_age = os.path.join(path_age, "sample_age.txt")
    file_methy_age = os.path.join(path_age, "age_methylation_v1.txt")
    path_out_age = "/home/zhangyu/mnt_path/Data/EWAS_process/age/450k"
    FileUtils.makedir(path_out_age)
    file_cpg_450k_age = os.path.join(path_out_age, 'cpgs_list.txt')
    file_npz_age = os.path.join(path_out_age, "Processed_age_450k.npy")
    generate_npy_2(file_meta_age, file_methy_age, file_cpg_450k_age, file_npz_age,
                   disease=False)

    #%%
    # EWAS bmi dataset
    file_meta_bmi = "/home/zhangyu/mnt_path/Data/EWAS/bmi/sample_bmi.txt"
    file_methy_bmi = "/home/zhangyu/mnt_path/Data/EWAS/bmi/bmi_methylation_v1.txt"
    file_cpg_450k_bmi = '/home/zhangyu/mnt_path/Data/EWAS_process/bmi/cpgs_bmi_450k_rm20.txt'
    file_npz_bmi = "/home/zhangyu/mnt_path/Data/EWAS_process/bmi/Processed_bmi_450k.npy"
    generate_npy(file_meta_bmi, file_methy_bmi, file_cpg_450k_bmi, file_npz_bmi)
