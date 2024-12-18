

#%%
import os
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from Bio import Entrez
import time
from typing import List
from pygtrans import Translate
import pandas as pd
from tqdm import tqdm


def list_intersection(list_a , list_b):
    """
    This function takes two lists as input and returns a list containing the intersection of the two lists.
    """
    result =  [item for item in list_a  if item in list_b]
    return result

def list_minus(list_a, list_b):
    return [item for item in list_a if item not in list_b]


def split_and_impute(matrix, block_size, k_neighbors=5):
    """
    主要用于实现KNN impute用的
    Split a large matrix into small blocks and impute each block using KNNImputer.
    :param matrix: the large matrix to be imputed
    :param block_size: the size of the blocks
    :param k_neighbors: the number of nearest neighbors to use for imputation
    :return: the imputed matrix
    """
    # Calculate the number of blocks in each dimension
    num_blocks_row = int(np.ceil(matrix.shape[0] / block_size[0]))
    num_blocks_col = int(np.ceil(matrix.shape[1] / block_size[1]))

    # Split the matrix into blocks and impute each block
    imputed_blocks = []
    for i in range(num_blocks_row):
        # print(f"i: {i}")
        for j in range(num_blocks_col):
            print(f"i: {i}, j: {j}")
            # Calculate the indices for the current block
            start_row = i * block_size[0]
            end_row = min(start_row + block_size[0], matrix.shape[0])
            start_col = j * block_size[1]
            end_col = min(start_col + block_size[1], matrix.shape[1])

            # Extract the current block
            block = matrix[start_row:end_row, start_col:end_col]

            # Impute the current block using KNNImputer
            imputer = KNNImputer(n_neighbors = k_neighbors)
            imputed_block = imputer.fit_transform(block)

            # Append the imputed block to the list
            imputed_blocks.append(imputed_block)

    # Combine the imputed blocks into a single matrix

    imputed_matrix = np.zeros_like(matrix)
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            print(i,j)
            # Calculate the indices for the current block
            start_row = i * block_size[0]
            end_row = min(start_row + block_size[0], matrix.shape[0])
            start_col = j * block_size[1]
            end_col = min(start_col + block_size[1], matrix.shape[1])

            # Get the imputed block
            block_idx = i * num_blocks_col + j
            print(block_idx)
            imputed_block = imputed_blocks[block_idx]
            print(imputed_block.shape)

            # Combine the imputed block into the imputed matrix
            imputed_matrix[start_row:end_row, start_col:end_col] = imputed_block

    return imputed_matrix
# %%

# %%

def generate_mapping(df_table, feature_name,index2value = {} ):

    """
    对于df_table表的 feature_name 特征
    """
    df_temp = df_table.groupby(feature_name).count()
    df_temp = df_temp.sort_values(by = df_temp.columns[0], ascending= False)
    # 产出双向映射
    if(len(index2value) == 0):
        value2index = {}    
        min_index = 1
    else:
        value2index = {}    
        for key,value in index2value.items():
            value2index[value] = key
            min_index = len(index2value) + 1

    for feature_value, _ in df_temp.iterrows():
        if(feature_value not in value2index):
            value2index[feature_value] = min_index
            index2value[min_index] = feature_value
            min_index += 1

    return index2value, value2index



# %%
def translate_names_with_retry(input_list):
    client = Translate()
    original_list = []
    translated_cn_list = []
    
    for index, name in enumerate(input_list):
        for attempt in range(3):
            try:
                text = client.translate(name)
                text_cn = text.translatedText
                original_list.append(name)
                translated_cn_list.append(text_cn)
                print(index, name, text_cn)
                time.sleep(0.02)
                break  # If the request is successful, break out of the for loop
            except:  # Catch all exceptions
                print(f"Attempt {attempt + 1} of 3 failed for {name}")
                time.sleep(1)  # Wait for 1 second before retrying
        
    return original_list, translated_cn_list




def search_pmid_info(PMID_list: List[str], sleep_time: float = 0.1) -> pd.DataFrame:

    """
    This function searches PubMed for information about a list of papers specified by their PubMed IDs (PMIDs).
    For each paper, the function retrieves its title, publication date, and abstract (if available) and returns
    the results as a Pandas DataFrame.
    
    Parameters:
        PMID_list (list): A list of PubMed IDs (PMIDs) to be searched.
        sleep_time (float): A float value specifying the time to sleep between each PubMed API request. 
                            Default is 0.1 seconds.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the retrieved information for each paper, including PMID, 
                           publication date, title, and abstract (if available).
    """


    print(f"total # of paper of be searched = {len(PMID_list)}")

    start_time = time.time()
    pmid_list = []
    year_list = []
    title_list = []
    abstract_list = []
    find_number =  0
    count = 0
    for pmid in tqdm(PMID_list):
        print(count)
        count += 1
        time.sleep(sleep_time)
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml",
            rettype="medline", timeout=5)
            # Parse the XML file to extract the paper's title
            record = Entrez.read(handle)
            title = record['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
            pubYear = record['PubmedArticle'][0]['MedlineCitation']['Article']["Journal"]["JournalIssue"]["PubDate"]["Year"]
            abstractText = record['PubmedArticle'][0]['MedlineCitation']['Article']["Abstract"]["AbstractText"]
            
            pmid_list.append( pmid )
            year_list.append(pubYear )
            title_list.append(title)
            abstract_list.append( abstractText)
            find_number += 1
            print(title)
            handle.close()

        except:
            print(f"not find {pmid}")
            pmid_list.append( pmid )
            year_list.append("None" )
            title_list.append("None")
            abstract_list.append("None")

    end_time = time.time()
    print(f"total used time:{end_time - start_time}, find number:{find_number}")
    
    return_df = pd.DataFrame( data = {"PMID": pmid_list, 
                                   "Publish Year": year_list,
                                   "Title": title_list,
                                   "Abstract": abstract_list})
    
    return return_df


def random_dropout_dataframe(dataframe, dropout_prob=0.2):
    mask = np.random.rand(*dataframe.shape) < dropout_prob
    dropout_df = dataframe.copy()
    dropout_df[mask] = np.nan
    return dropout_df


def read_single_csv(input_path, sep=','):
    df_chunk = pd.read_csv(input_path, sep=sep, chunksize=1000, index_col=0)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df = pd.concat(res_chunk)
    return res_df


def impute_methy(df_methy, by_project=True, df_meta_in=None, use_col=None):
    df_methy = df_methy.T
    if by_project:
        project_ids = np.unique(df_meta_in[use_col])
        list_imp = []
        list_samples = []
        for one_id in project_ids:
            df_meta_in_sub = df_meta_in.loc[df_meta_in[use_col] == one_id, :]
            df_methy_sub = df_methy.loc[df_meta_in_sub['sample_id'].tolist(), :]
            imp_mean = SimpleImputer(keep_empty_features=True)
            df_methy_imp = imp_mean.fit_transform(df_methy_sub)
            list_imp.append(df_methy_imp)
            list_samples.extend(df_meta_in_sub['sample_id'].tolist())
            # print(df_methy_imp.shape)
            # print(len(df_meta_in_sub['sample_id'].tolist()))
        x_imp = np.concatenate(list_imp, axis=0)
    else:
        imp_mean = SimpleImputer()
        x_imp = imp_mean.fit_transform(df_methy)
        list_samples = list(df_methy.index)

    return x_imp, list_samples


def data_augmentation(df_mat_in, df_meta_in, num_aug, by_project=True, use_col='project_id',
                      dropout_prob=0.2):
    list_mat = []
    list_meta = []
    list_mat.append(df_mat_in.T)
    list_meta.append(df_meta_in)
    for i in range(num_aug):
        df_drop = random_dropout_dataframe(df_mat_in, dropout_prob=dropout_prob)
        df_drop.columns = [f"{col}_{i}" for col in df_mat_in.columns]
        df_drop_meta = df_meta_in.copy()
        df_drop_meta.index = df_drop.columns
        df_drop_meta['sample_id'] = df_drop.columns
        list_cpgs = list(df_mat_in.index)
        # list_samples = list(df_mat_in.columns) + list(df_drop.columns)
        # print(df_mat_in.shape)
        # print(df_drop.shape)
        df_imp, samples_imp = impute_methy(pd.concat([df_mat_in, df_drop], axis=1),
                                           df_meta_in=pd.concat([df_meta_in, df_drop_meta], axis=0),
                                           by_project=by_project, use_col=use_col)
        # print(df_imp.shape)
        df_imp = pd.DataFrame(df_imp, columns=list_cpgs, index=samples_imp)
        # df_imp = pd.DataFrame(df_imp)
        df_drop_imp = df_imp.loc[df_drop.columns, :]
        list_mat.append(df_drop_imp)
        list_meta.append(df_drop_meta)

    df_mat_out = pd.concat(list_mat, axis=0)
    df_meta_out = pd.concat(list_meta, axis=0)

    return df_mat_out, df_meta_out


def get_new_index(original_cpg_list, ref_cpg_list):
    ori_array = np.array(original_cpg_list)
    ori_indices = {val: idx for idx, val in enumerate(ori_array)}
    indices_in_ref_order = [ori_indices[val] for val in ref_cpg_list]

    return indices_in_ref_order

def order_cpg_to_ref(original_cpg_list, ref_cpg_list, np_dict, list_samples):
    #Calculate index in ref in the order of ref
    ori_array = np.array(original_cpg_list)
    # ref_array = np.array(original_cpg_list)
    # common_elements = np.intersect1d(ori_array, ref_array, return_indices=True)
    # common_indices = common_elements[1]
    ori_indices = {val: idx for idx, val in enumerate(ori_array)}
    indices_in_ref_order = [ori_indices[val] for val in ref_cpg_list]
    #select cpgs in original_cpg_list by ref
    save_dict = {}
    for sample_id in list_samples:
        one_sample = np_dict[sample_id]
        old_feature = one_sample["feature"]
        one_sample["feature"] = old_feature[indices_in_ref_order]
        save_dict[sample_id] = one_sample

    return save_dict


def order_cpg_to_ref_fill0(original_cpg_list, ref_cpg_list, np_dict, list_samples):
    #Calculate index in ref in the order of ref
    set_old_cpg_list = set(original_cpg_list)
    plus_cpgs = [one for one in ref_cpg_list if one not in set_old_cpg_list]
    if len(plus_cpgs) > 0:
        old_cpg_list = original_cpg_list + plus_cpgs
    else:
        old_cpg_list = original_cpg_list
    ori_array = np.array(old_cpg_list)
    # ref_array = np.array(original_cpg_list)
    # common_elements = np.intersect1d(ori_array, ref_array, return_indices=True)
    # common_indices = common_elements[1]
    ori_indices = {val: idx for idx, val in enumerate(ori_array)}
    indices_in_ref_order = [ori_indices[val] for val in ref_cpg_list]
    #select cpgs in original_cpg_list by ref
    save_dict = {}
    for sample_id in list_samples:
        one_sample = np_dict[sample_id]
        if len(plus_cpgs) > 0:
            old_feature = np.concatenate((one_sample["feature"],
                                          np.zeros(len(plus_cpgs), dtype=np.float32)))
        else:
            old_feature = one_sample["feature"]
        one_sample["feature"] = old_feature[indices_in_ref_order]
        save_dict[sample_id] = one_sample

    return save_dict


def data_to_np_dict(methy_np, df_meta, sample_list, columns_list, file_out_path):
    X_np = methy_np.astype(np.float32)
    y_pd = df_meta.loc[sample_list, :]

    all_keys = []
    save_dict = {}
    target_name = "age"
    additional_list = columns_list
    err_num=0
    for row_number in range(len(y_pd)):
        key_rename = y_pd.iloc[row_number].sample_id

        try:
            # age = np.asarray([np.array(y_pd.iloc[row_number][target_name], dtype=np.float32)])
            all_keys.append(key_rename)
            feature_np = np.array(X_np.loc[key_rename, :]).astype(np.float32)

            save_dict[key_rename] = {}
            save_dict[key_rename]["feature"] = feature_np
            #age = np.asarray([y_pd.iloc[row_number][target_name].astype(np.float32)])
            # save_dict[key_rename]["target"] = age

            additional = {}
            for additional_name in additional_list:
                value = y_pd.iloc[row_number][additional_name]
                additional[additional_name] = value

            save_dict[key_rename]["additional"] = additional

        except(ValueError):
            err_num = err_num + 1
            print(key_rename)
            continue

    if err_num != len(y_pd):
        np.save(file_out_path, save_dict)
    #     print(err_num)
    # else:
    #     print(err_num)
