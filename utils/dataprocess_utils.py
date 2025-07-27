import os
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from Bio import Entrez
import time
from typing import List
from pygtrans import Translate
import pandas as pd
from tqdm import tqdm


def list_intersection(list_a, list_b):
    """
    Find the intersection of two lists.
    
    Args:
        list_a (list): First list
        list_b (list): Second list
        
    Returns:
        list: Elements that appear in both list_a and list_b
    """
    result = [item for item in list_a if item in list_b]
    return result


def list_minus(list_a, list_b):
    """
    Subtract list_b from list_a (set difference).
    
    Args:
        list_a (list): First list
        list_b (list): Second list
        
    Returns:
        list: Elements that appear in list_a but not in list_b
    """
    return [item for item in list_a if item not in list_b]


def split_and_impute(matrix, block_size, k_neighbors=5):
    """
    Split a large matrix into blocks and impute missing values using KNN.
    
    This function handles large matrices by splitting them into smaller blocks,
    applying KNN imputation to each block, and then recombining the results.
    
    Args:
        matrix: Large matrix to be imputed
        block_size: Size of the blocks (tuple of row_size, col_size)
        k_neighbors: Number of nearest neighbors for KNN imputation
        
    Returns:
        numpy.ndarray: Imputed matrix
    """
    # Calculate the number of blocks in each dimension
    num_blocks_row = int(np.ceil(matrix.shape[0] / block_size[0]))
    num_blocks_col = int(np.ceil(matrix.shape[1] / block_size[1]))

    # Split the matrix into blocks and impute each block
    imputed_blocks = []
    for i in range(num_blocks_row):
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
            imputer = KNNImputer(n_neighbors=k_neighbors)
            imputed_block = imputer.fit_transform(block)

            # Append the imputed block to the list
            imputed_blocks.append(imputed_block)

    # Combine the imputed blocks into a single matrix
    imputed_matrix = np.zeros_like(matrix)
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            print(i, j)
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


def generate_mapping(df_table, feature_name, index2value={}):
    """
    Generate bidirectional mappings between feature values and indices.
    
    This function creates mappings between categorical feature values and numerical indices,
    which can be used for embedding lookups or other operations requiring numerical indices.
    
    Args:
        df_table (pd.DataFrame): DataFrame containing the feature
        feature_name (str): Name of the feature to create mappings for
        index2value (dict): Existing index to value mapping to extend (optional)
        
    Returns:
        tuple: (index2value, value2index) mapping dictionaries
    """
    # Group and count feature values
    df_temp = df_table.groupby(feature_name).count()
    df_temp = df_temp.sort_values(by=df_temp.columns[0], ascending=False)
    
    # Create or extend bidirectional mappings
    if len(index2value) == 0:
        value2index = {}    
        min_index = 1
    else:
        value2index = {}    
        for key, value in index2value.items():
            value2index[value] = key
            min_index = len(index2value) + 1

    # Add new values to mappings
    for feature_value, _ in df_temp.iterrows():
        if feature_value not in value2index:
            value2index[feature_value] = min_index
            index2value[min_index] = feature_value
            min_index += 1

    return index2value, value2index


def translate_names_with_retry(input_list):
    """
    Translate a list of names with retry mechanism for failed requests.
    
    Args:
        input_list (list): List of names to translate
        
    Returns:
        tuple: (original_list, translated_cn_list) containing original and translated names
    """
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
                break  # If the request is successful, break out of the retry loop
            except:  # Catch all exceptions
                print(f"Attempt {attempt + 1} of 3 failed for {name}")
                time.sleep(1)  # Wait for 1 second before retrying
        
    return original_list, translated_cn_list


def search_pmid_info(PMID_list: List[str], sleep_time: float = 0.1) -> pd.DataFrame:
    """
    Search PubMed for information about papers based on their PubMed IDs.
    
    This function retrieves title, publication date, and abstract for each PMID,
    with a delay between requests to avoid overloading the PubMed API.
    
    Args:
        PMID_list (list): List of PubMed IDs to search
        sleep_time (float): Time to sleep between API requests
        
    Returns:
        pd.DataFrame: DataFrame containing retrieved information for each paper
    """
    print(f"Total # of papers to be searched = {len(PMID_list)}")

    start_time = time.time()
    pmid_list = []
    year_list = []
    title_list = []
    abstract_list = []
    find_number = 0
    count = 0
    
    for pmid in tqdm(PMID_list):
        print(count)
        count += 1
        time.sleep(sleep_time)
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml",
                                  rettype="medline", timeout=5)
            # Parse the XML file to extract paper information
            record = Entrez.read(handle)
            title = record['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
            pubYear = record['PubmedArticle'][0]['MedlineCitation']['Article']["Journal"]["JournalIssue"]["PubDate"]["Year"]
            abstractText = record['PubmedArticle'][0]['MedlineCitation']['Article']["Abstract"]["AbstractText"]
            
            pmid_list.append(pmid)
            year_list.append(pubYear)
            title_list.append(title)
            abstract_list.append(abstractText)
            find_number += 1
            print(title)
            handle.close()
        except:
            print(f"Not found: {pmid}")
            pmid_list.append(pmid)
            year_list.append("None")
            title_list.append("None")
            abstract_list.append("None")

    end_time = time.time()
    print(f"Total used time: {end_time - start_time}, papers found: {find_number}")
    
    # Create DataFrame with results
    return_df = pd.DataFrame(data={"PMID": pmid_list, 
                                   "Publish Year": year_list,
                                   "Title": title_list,
                                   "Abstract": abstract_list})
    
    return return_df


def random_dropout_dataframe(dataframe, dropout_prob=0.2):
    """
    Randomly introduce missing values (NaN) in a DataFrame.
    
    This function is used for data augmentation by creating variations
    of the original data with randomly missing values.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        dropout_prob (float): Probability of dropping each value
        
    Returns:
        pd.DataFrame: DataFrame with randomly introduced NaN values
    """
    mask = np.random.rand(*dataframe.shape) < dropout_prob
    dropout_df = dataframe.copy()
    dropout_df[mask] = np.nan
    return dropout_df


def read_single_csv(input_path, sep=','):
    """
    Read a large CSV file in chunks and combine into a single DataFrame.
    
    Args:
        input_path (str): Path to CSV file
        sep (str): Separator character in the CSV file
        
    Returns:
        pd.DataFrame: Combined DataFrame from all chunks
    """
    df_chunk = pd.read_csv(input_path, sep=sep, chunksize=1000, index_col=0)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df = pd.concat(res_chunk)
    return res_df


def impute_methy(df_methy, by_project=True, df_meta_in=None, use_col=None):
    """
    Impute missing values in methylation data.
    
    This function can impute either by project (if by_project=True) or globally.
    Project-based imputation helps preserve batch-specific patterns.
    
    Args:
        df_methy (pd.DataFrame): Methylation data with CpG sites as rows
        by_project (bool): Whether to impute separately for each project
        df_meta_in (pd.DataFrame): Sample metadata (required if by_project=True)
        use_col (str): Column in df_meta_in identifying projects (required if by_project=True)
        
    Returns:
        tuple: (imputed_matrix, sample_list) containing the imputed data and sample IDs
    """
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
        x_imp = np.concatenate(list_imp, axis=0)
    else:
        imp_mean = SimpleImputer()
        x_imp = imp_mean.fit_transform(df_methy)
        list_samples = list(df_methy.index)

    return x_imp, list_samples


def data_augmentation(df_mat_in, df_meta_in, num_aug, by_project=True, use_col='project_id',
                      dropout_prob=0.2):
    """
    Augment methylation data by creating variations with random dropout.
    
    This function creates multiple augmented versions of the input data by:
    1. Creating copies with random missing values
    2. Imputing those missing values to create variations
    
    Args:
        df_mat_in (pd.DataFrame): Methylation matrix (CpG sites as rows)
        df_meta_in (pd.DataFrame): Sample metadata
        num_aug (int): Number of augmented copies to create
        by_project (bool): Whether to impute by project
        use_col (str): Column in df_meta_in identifying projects
        dropout_prob (float): Probability of dropping values
        
    Returns:
        tuple: (df_mat_out, df_meta_out) containing augmented data and metadata
    """
    # Start with original data
    list_mat = []
    list_meta = []
    list_mat.append(df_mat_in.T)
    list_meta.append(df_meta_in)
    
    # Create augmented copies
    for i in range(num_aug):
        # Create version with random dropout
        df_drop = random_dropout_dataframe(df_mat_in, dropout_prob=dropout_prob)
        df_drop.columns = [f"{col}_{i}" for col in df_mat_in.columns]
        
        # Create metadata for augmented samples
        df_drop_meta = df_meta_in.copy()
        df_drop_meta.index = df_drop.columns
        df_drop_meta['sample_id'] = df_drop.columns
        list_cpgs = list(df_mat_in.index)
        
        # Impute the combined data (original + augmented)
        df_imp, samples_imp = impute_methy(pd.concat([df_mat_in, df_drop], axis=1),
                                           df_meta_in=pd.concat([df_meta_in, df_drop_meta], axis=0),
                                           by_project=by_project, use_col=use_col)
        
        # Extract the augmented part after imputation
        df_imp = pd.DataFrame(df_imp, columns=list_cpgs, index=samples_imp)
        df_drop_imp = df_imp.loc[df_drop.columns, :]
        
        # Add to list of augmented data
        list_mat.append(df_drop_imp)
        list_meta.append(df_drop_meta)

    # Combine all augmented versions
    df_mat_out = pd.concat(list_mat, axis=0)
    df_meta_out = pd.concat(list_meta, axis=0)

    return df_mat_out, df_meta_out


def get_new_index(original_cpg_list, ref_cpg_list):
    """
    Get indices to reorder CpG sites to match reference order.
    
    Args:
        original_cpg_list (list): Original list of CpG site IDs
        ref_cpg_list (list): Reference list of CpG site IDs
        
    Returns:
        list: Indices in original_cpg_list that correspond to the order in ref_cpg_list
    """
    ori_array = np.array(original_cpg_list)
    ori_indices = {val: idx for idx, val in enumerate(ori_array)}
    indices_in_ref_order = [ori_indices[val] for val in ref_cpg_list]

    return indices_in_ref_order


def order_cpg_to_ref(original_cpg_list, ref_cpg_list, np_dict, list_samples):
    """
    Reorder CpG sites in data to match reference order.
    
    Args:
        original_cpg_list (list): Original list of CpG site IDs
        ref_cpg_list (list): Reference list of CpG site IDs
        np_dict (dict): Dictionary containing methylation data
        list_samples (list): List of sample IDs
        
    Returns:
        dict: Dictionary with reordered features
    """
    # Calculate indices in original list that match reference order
    ori_array = np.array(original_cpg_list)
    ori_indices = {val: idx for idx, val in enumerate(ori_array)}
    indices_in_ref_order = [ori_indices[val] for val in ref_cpg_list]
    
    # Reorder features for each sample
    save_dict = {}
    for sample_id in list_samples:
        one_sample = np_dict[sample_id]
        old_feature = one_sample["feature"]
        one_sample["feature"] = old_feature[indices_in_ref_order]
        save_dict[sample_id] = one_sample

    return save_dict


def order_cpg_to_ref_fill0(original_cpg_list, ref_cpg_list, np_dict, list_samples):
    """
    Reorder CpG sites in data to match reference order, filling in zeros for missing sites.
    
    This function handles the case where the reference list contains CpG sites
    not present in the original data by filling in zeros.
    
    Args:
        original_cpg_list (list): Original list of CpG site IDs
        ref_cpg_list (list): Reference list of CpG site IDs
        np_dict (dict): Dictionary containing methylation data
        list_samples (list): List of sample IDs
        
    Returns:
        dict: Dictionary with reordered features and zeros for missing sites
    """
    # Find CpG sites in reference but not in original list
    set_old_cpg_list = set(original_cpg_list)
    plus_cpgs = [one for one in ref_cpg_list if one not in set_old_cpg_list]
    
    # Create extended list if needed
    if len(plus_cpgs) > 0:
        old_cpg_list = original_cpg_list + plus_cpgs
    else:
        old_cpg_list = original_cpg_list
    
    # Calculate indices in extended list that match reference order
    ori_array = np.array(old_cpg_list)
    ori_indices = {val: idx for idx, val in enumerate(ori_array)}
    indices_in_ref_order = [ori_indices[val] for val in ref_cpg_list]
    
    # Reorder features for each sample, with zeros for missing sites
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
    """
    Convert methylation data and metadata to a dictionary format and save as numpy file.
    
    Args:
        methy_np (pd.DataFrame): Methylation data
        df_meta (pd.DataFrame): Sample metadata
        sample_list (list): List of sample IDs
        columns_list (list): List of metadata columns to include
        file_out_path (str): Path to save the output numpy file
        
    Returns:
        None
    """
    X_np = methy_np.astype(np.float32)
    y_pd = df_meta.loc[sample_list, :]

    all_keys = []
    save_dict = {}
    target_name = "age"
    additional_list = columns_list
    err_num = 0
    
    for row_number in range(len(y_pd)):
        key_rename = y_pd.iloc[row_number].sample_id

        try:
            # Add sample to dictionary
            all_keys.append(key_rename)
            feature_np = np.array(X_np.loc[key_rename, :]).astype(np.float32)

            save_dict[key_rename] = {}
            save_dict[key_rename]["feature"] = feature_np

            # Add metadata
            additional = {}
            for additional_name in additional_list:
                value = y_pd.iloc[row_number][additional_name]
                additional[additional_name] = value

            save_dict[key_rename]["additional"] = additional

        except ValueError:
            err_num = err_num + 1
            print(key_rename)
            continue

    # Save dictionary if any samples were successfully processed
    if err_num != len(y_pd):
        np.save(file_out_path, save_dict)
