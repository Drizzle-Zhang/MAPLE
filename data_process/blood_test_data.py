# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: blood_test_data.py
# @time: 2024/6/19 23:09

import os
from time import time
import datatable as dt
import numpy as np
import pandas as pd
from datetime import datetime
os.chdir('/home/zhangyu/bioage/code')
from utils.file_utils import read_stringList_FromFile, write_stringList_2File, FileUtils
from utils.common_utils import data_to_np_dict, get_new_index, order_cpg_to_ref_fill0, impute_methy


def mat2npy(path_tmp, df_mat_in, df_meta_in, file_cpgs_ref, missing_rate=1):

    missing_rates = \
        pd.Series(np.sum(np.isnan(df_mat_in), axis=1) / df_mat_in.shape[1],
                  index=df_mat_in.index)
    df_mat = df_mat_in.loc[missing_rates.loc[missing_rates <= missing_rate].index, :]

    file_cpgs = os.path.join(path_tmp, 'all_cpgs.txt')

    write_stringList_2File(file_cpgs, list(df_mat.index))

    X_mat_imp, samples = impute_methy(df_mat, by_project=False)
    df_mat_imp = pd.DataFrame(X_mat_imp, index=samples, columns=df_mat.index)

    # save dict
    list_samples = df_meta_in.index
    file_npy = os.path.join(path_tmp, 'Processed_all.npy')
    data_to_np_dict(df_mat_imp, df_meta_in, list_samples, ["sample_id", "age"], file_npy)

    # generate new dict
    old_dict = np.load(file_npy, allow_pickle = True).item()
    old_cpg_list = read_stringList_FromFile(file_cpgs)
    ref_cpg_list = read_stringList_FromFile(file_cpgs_ref)
    save_dict = order_cpg_to_ref_fill0(old_cpg_list, ref_cpg_list, old_dict, list_samples,
                                       additional_list=("sample_id", "age"))

    return save_dict


if __name__ == '__main__':
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    path_tmp_time = os.path.join('/home/zhangyu/mnt_path/test', folder_name)

    path_data = '/home/zhangyu/mnt_path/Data/'
    vec_gse = ['GSE196696', 'GSE210255', 'GSE55763', 'GSE210254',
               'GSE72680', 'GSE87571']
    vec_folder_mat = \
        ['850k/preprocess_GSE196696/', 'GENOA/preprocess_GSE210255/',
         'LOLIPOP/preprocess_GSE55763/', 'GENOA/preprocess_GSE210254/',
         '450k/preprocess_GSE72680/', '450k/preprocess_GSE87571/']
    vec_meta = ['850k/GSE196696_process/GSE196696_meta.tsv',
                  'GENOA/GSE210255_process/GSE210255_meta.tsv',
                  'LOLIPOP/GSE55763_meta.tsv',
                  'GENOA/GSE210254_process/GSE210254_meta.tsv',
                  '450k/GSE72680_meta.tsv', '450k/GSE87571_meta.tsv']
    vec_merge_type = ['1', '2', '1', '2', '1','2']

    list_methods = ['RAW', 'SWAN', 'BMIQ', 'GMQN']
    list_method_files = ['beta_raw', 'beta_swan', 'beta_bmiq', 'beta_gmqn']

    df_450k = pd.read_csv(
        '/home/zhangyu/mnt_path/Data/platform_files/minfi_450k.csv', index_col=0
    )
    cpgs_450k = df_450k.index
    df_850k = pd.read_csv(
        '/home/zhangyu/mnt_path/Data/platform_files/minfi_850k.csv', index_col=0
    )
    cpgs_850k = set(df_850k.index)
    cpgs_merge = [one for one in cpgs_450k if one in cpgs_850k]

    path_out = '/home/zhangyu/mnt_path/intermediate_data/test_blood/'
    file_cpgs_450k_850k = os.path.join(path_out, 'merge_blood_cpgs.txt')
    write_stringList_2File(file_cpgs_450k_850k, cpgs_merge)
    save_dict = {}

    time_start = time()
    for i_method in range(len(list_methods)):
        for i in range(len(vec_gse)):
            df_mat = dt.fread(
                f"{path_data}{vec_folder_mat[i]}{list_method_files[i_method]}.txt").to_pandas()
            df_mat.index = df_mat['C0'].tolist()
            df_mat = df_mat.drop(columns='C0')
            df_meta = pd.read_csv(f"{path_data}{vec_meta[i]}", sep='\t', index_col=0)
            if vec_merge_type[i] == '1':
                df_meta_ori = df_meta
                df_meta_ori.index = df_meta_ori['ori_sample_id'].tolist()
                df_mat.columns = df_meta_ori.loc[df_mat.columns, 'sample_id']
            elif vec_merge_type[i] == '2':
                cols_mat = [one.split('_')[0] for one in df_mat.columns]
                df_mat.columns = cols_mat
            df_meta_sub = df_meta
            df_meta_sub.index = \
                df_meta_sub['sample_id'].apply(lambda x: f"{x}_{list_methods[i_method]}")
            df_meta_sub['sample_id'] = df_meta_sub.index
            df_mat_sub = df_mat
            df_mat_sub.columns = [f"{one}_{list_methods[i_method]}" for one in df_mat.columns]
            path_age = os.path.join(path_tmp_time, 'Age')
            FileUtils.makedir(path_age)
            sub_dict = mat2npy(path_age, df_mat, df_meta, file_cpgs_450k_850k)
            save_dict.update(sub_dict)
            if (i_method == 0) & (i == 0):
                df_meta_total = df_meta_sub
            else:
                df_meta_total = pd.concat([df_meta_total, df_meta_sub], axis=0)

    np.save(os.path.join(path_out, 'merge_blood.npy'), save_dict)
    df_meta_total.to_csv(os.path.join(path_out, 'merge_blood.tsv'), sep='\t')

    time_end = time()
    print(time_end - time_start)
