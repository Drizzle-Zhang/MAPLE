# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: pipeline.py
# @time: 2023/12/25 15:50


import os
from typing import List
import subprocess
import random
import pickle
import warnings
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from models.FC import FCNet_H_class, FCNet_H
from models.CT import ContrastiveEncoder
from utils.file_utils import read_stringList_FromFile, write_stringList_2File, FileUtils
from utils.dataload_utils import geo_npz_Dataset_inference
from utils.dataprocess_utils import data_to_np_dict, order_cpg_to_ref_fill0, impute_methy
from config import ArgsDataClass, Config
import shutil
import time


def load_disease_model(args: ArgsDataClass, device: str = "cuda"):
    feature_size = args.feature_size
    encoder_hidden_list = [int(i.strip()) for i in args.encoder_hidden_str.split(",")]
    decoder_hidden_list = [int(i.strip()) for i in args.decoder_hidden_str.split(",")]
    encoder = ContrastiveEncoder(feature_channel = feature_size,
                                 hidden_list= encoder_hidden_list,
                                 h_dim = args.latent_size )
    predictor_model = FCNet_H_class(feature_channel = args.latent_size,
                                    output_channel = 2,
                                    hidden_list = decoder_hidden_list,
                                    if_bn = False,
                                    if_dp = False)
    # load model params
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    predictor_model.load_state_dict(checkpoint["predictor_state_dict"])
    encoder = encoder.to(device)
    predictor_model = predictor_model.to(device)

    return encoder, predictor_model


def load_age_model(args: ArgsDataClass, device: str = "cuda"):
    encoder_hidden_list = [int(i.strip()) for i in args.encoder_hidden_str.split(",")]
    decoder_hidden_list = [int(i.strip()) for i in args.decoder_hidden_str.split(",")]
    encoder = ContrastiveEncoder(feature_channel = args.feature_size,
                                 hidden_list= encoder_hidden_list,
                                 h_dim = args.latent_size )

    predictor_model = FCNet_H(feature_channel = args.latent_size,
                              output_channel = 1,
                              hidden_list = decoder_hidden_list,
                              if_bn = False,
                              if_dp = False)

    # load model params
    checkpoint = torch.load(args.checkpoint_path)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    predictor_model.load_state_dict(checkpoint["predictor_state_dict"])
    encoder = encoder.to(device)
    predictor_model = predictor_model.to(device)

    return encoder, predictor_model


def mat2npy(path_tmp, df_mat_in, df_meta_in, file_cpgs_ref, missing_rate=0.8):

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
    data_to_np_dict(df_mat_imp, df_meta_in, list_samples, ["sample_id"], file_npy)

    # generate new dict
    old_dict = np.load(file_npy, allow_pickle = True).item()
    old_cpg_list = read_stringList_FromFile(file_cpgs)
    ref_cpg_list = read_stringList_FromFile(file_cpgs_ref)
    save_dict = order_cpg_to_ref_fill0(old_cpg_list, ref_cpg_list, old_dict, list_samples)
    file_idx = os.path.join(path_tmp, 'test_index.pkl')
    with open(file_idx, 'wb') as f:
        pickle.dump(list_samples, f)
        pickle.dump([], f)
    file_save_dict_out = os.path.join(path_tmp, 'Processed.npy')
    np.save(file_save_dict_out, save_dict)

    return


def risk_score(path_tmp, encoder, predictor_model, adata_in, df_meta_in, disease, device):
    file_save_dict = os.path.join(path_tmp, 'Processed.npy')
    file_idx = os.path.join(path_tmp, 'test_index.pkl')
    data_dict = geo_npz_Dataset_inference(file_save_dict, file_idx, if_train = True)
    dataloader = DataLoader(data_dict, batch_size = 10000, num_workers = 0, shuffle= False)

    with torch.no_grad():
        for batch_idx, (feature, additional) in enumerate(dataloader):
            feature = feature.to(torch.float32).to(device)
            latent_feature = encoder(feature)

    adata_pred = ad.AnnData(X=latent_feature.cpu().detach().numpy(), obs=df_meta_in)
    adata_in.obs['type'] = 'background'
    adata_pred.obs['type'] = 'new_sample'
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    adata_out = ad.concat([adata_in, adata_pred], join='outer')
    sc.pp.scale(adata_out, max_value=5)
    sc.tl.pca(adata_out, svd_solver='arpack', n_comps=2)
    # sc.pl.pca(adata_out, color='type', save=)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # sc.pl.pca(adata_out, color="type", show=False, ax=ax)
    # plt.savefig(os.path.join(path_tmp, 'PCA_type.png'))
    if disease == 'CVD':
        adata_out.obs['Risk_score'] = adata_out.obsm['X_pca'][:, 0]*0.5 + adata_out.obsm['X_pca'][:, 1]*0.5
        disease_name = 'stroke'
    elif disease == 'T2D':
        adata_out.obs['Risk_score'] = adata_out.obsm['X_pca'][:, 0]*0.69 + adata_out.obsm['X_pca'][:, 1]*0.31
        disease_name = 'type 2 diabetes'

    df_pc_risk = adata_out.obs.copy()
    df_pc_risk['sample_id'] = df_pc_risk.index
    # df_pc_risk = df_pc_risk.loc[
    #              ~((df_pc_risk['disease'] != 'control') & (df_pc_risk['sample_id'].apply(lambda x: len(x.split('_'))==2))), :]
    # # df_pc_risk = df_pc_risk.loc[df_pc_risk['tissue'] == 'whole blood']
    # import pdb; pdb.set_trace()
    min_risk = np.median(df_pc_risk.loc[(df_pc_risk['age'] < 25), 'Risk_score'])
    max_risk = np.median(df_pc_risk.loc[df_pc_risk['disease'] == disease_name, 'Risk_score'])
    if min_risk > max_risk:
        min_risk = np.percentile(df_pc_risk.loc[(df_pc_risk['age'] <= 18), 'Risk_score'], 50)
        #     min_risk = np.percentile(df_pc_risk.loc[(df_pc_risk['bmi'] > 18.5) & (df_pc_risk['bmi'] < 25), 'PC1'], 75)
        max_risk = np.percentile(df_pc_risk.loc[df_pc_risk['disease'] == disease_name, 'Risk_score'], 25)
        risk_scores = (df_pc_risk['Risk_score'] - max_risk) / (min_risk - max_risk)
        df_pc_risk['Risk_score_norm'] = 1 - risk_scores
    else:
        min_risk = np.percentile(df_pc_risk.loc[(df_pc_risk['age'] <= 18), 'Risk_score'], 50)
        #     min_risk = np.percentile(df_pc_risk.loc[(df_pc_risk['bmi'] > 18.5) & (df_pc_risk['bmi'] < 25), 'PC1'], 25)
        max_risk = np.percentile(df_pc_risk.loc[df_pc_risk['disease'] == disease_name, 'Risk_score'], 75)
        df_pc_risk['Risk_score_norm'] = (df_pc_risk['Risk_score'] - min_risk) / (max_risk - min_risk)

    df_pc_risk.loc[df_pc_risk['Risk_score_norm'] > 1, 'Risk_score_norm'] = 1
    df_pc_risk.loc[df_pc_risk['Risk_score_norm'] < 0, 'Risk_score_norm'] = 0

    if disease == 'CVD':
        df_pc_risk.loc[df_pc_risk['Risk_score_norm'] > 0.8, 'Risk_score_norm'] = (df_pc_risk.loc[df_pc_risk['Risk_score_norm'] > 0.8, 'Risk_score_norm']-0.8)*0.5 + (0.8+0.2*0.5)
        df_pc_risk.loc[df_pc_risk['Risk_score_norm'] <= 0.8, 'Risk_score_norm'] = (df_pc_risk.loc[df_pc_risk['Risk_score_norm'] <= 0.8, 'Risk_score_norm'])*(0.8+0.2*0.5)/0.8

    df_meta_out = df_meta_in.copy()
    df_pc_risk = df_pc_risk.drop_duplicates(subset='sample_id')
    df_meta_out['Risk_score_norm'] = df_pc_risk.loc[df_meta_in.index, 'Risk_score_norm']

    return df_meta_out


def age_pred(path_tmp, encoder, predictor_model, df_meta_in, device):
    file_save_dict = os.path.join(path_tmp, 'Processed.npy')
    file_idx = os.path.join(path_tmp, 'test_index.pkl')
    data_dict = geo_npz_Dataset_inference(file_save_dict, file_idx, if_train = True)
    dataloader = DataLoader(data_dict, batch_size = 10000, num_workers = 0, shuffle= False)

    list_res = []
    with torch.no_grad():
        for batch_idx, (feature, additional) in enumerate(dataloader):
            feature = feature.to(torch.float32).to(device)
            latent_feature = encoder(feature)
            out = predictor_model(latent_feature)
            pred = out.cpu().detach().numpy()*100
            list_res.append(
                pd.DataFrame({'Pred_Age': pred[:,0]},
                             index=additional['sample_id']))
    df_res = pd.concat(list_res)
    df_meta_out = df_meta_in.copy()
    df_meta_out['Pred_Age'] = df_res.loc[df_meta_in.index, 'Pred_Age']

    return df_meta_out


class BioAgePipeline:
    def __init__(self, conf: Config, serialize_mode: bool = False):
        self.serialize_mode = serialize_mode
        self.input_format = conf.input_format
        self.device = conf.device
        self.encoder_cvd, self.predictor_model_cvd = load_disease_model(conf.args_cvd, conf.device)
        self.encoder_t2d, self.predictor_model_t2d = load_disease_model(conf.args_t2d, conf.device)
        self.encoder_age, self.predictor_model_age = load_age_model(conf.args_age, conf.device)
        self.data_root = os.path.dirname(os.path.abspath(__file__))
        # load background data
        self.adata_cvd = sc.read_h5ad(os.path.join(self.data_root, f"data/cvd_risk_scores.h5ad"))
        self.adata_t2d = sc.read_h5ad(os.path.join(self.data_root, f"data/t2d_risk_scores.h5ad"))
        self.cache_dir = conf.cache_dir
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.script_idat = os.path.join(self.data_root, 'raw_process/idat_process.R')
        self.file_cpgs_cvd = os.path.join(self.data_root, 'data/cpgs_blood_CVD')
        self.file_cpgs_t2d = os.path.join(self.data_root, 'data/cpgs_blood_T2D')
        self.file_cpgs_age = os.path.join(self.data_root, 'data/cpgs_age.txt')

        # temp folder
        if not self.serialize_mode:
            current_time = int(time.time()*1000)
            random_num = random.randint(0, 1000)
            self.temp_path = os.path.join(self.cache_dir, f"{current_time}_{random_num}")
            self.file_out = os.path.join(self.temp_path, f'Age_Risk_pred_{current_time}.tsv')
        else:
            self.temp_path = self.cache_dir
            self.file_out = os.path.join(self.temp_path, f'Age_Risk_pred.tsv')
        if os.path.exists(self.temp_path):
            shutil.rmtree(self.temp_path)
        FileUtils.makedir(self.temp_path)
        print(f"Create temp folder {self.temp_path}")


    def idat2mat(self, idat_files: List):
        path_idat = os.path.join(self.temp_path, 'idat_files')
        if os.path.exists(path_idat):
            raise ValueError(f"path_idat {path_idat} already exists")
        FileUtils.makedir(path_idat)
        for file_idat in idat_files:
            shutil.copy(file_idat, os.path.join(path_idat, os.path.basename(file_idat)))

        # processed data
        file_mat = os.path.join(self.temp_path, 'test_mat.csv')
        # script of raw data preprocessing
        subprocess.run(f"Rscript {self.script_idat} {path_idat} {file_mat}", shell=True)

        return file_mat

    def predict_beta(self, beta_file: str, file_meta: str):
        file_mat = beta_file
        df_mat_sample = pd.read_csv(file_mat, sep=',', index_col=0)
        df_meta_sample = pd.read_csv(file_meta, sep=',', index_col=0)
        # the column names of methylation matrix must be consistent with the row names of sample info matrix
        assert len(df_mat_sample.columns) == len(set(df_mat_sample.columns).intersection(df_meta_sample.index)), "The sample names of DNA methylation file are inconsistent with those of sample info file."
        df_mat_sample = df_mat_sample.loc[:, df_meta_sample.index]
        path_cvd = os.path.join(self.temp_path, 'CVD')
        if os.path.exists(path_cvd):
            raise ValueError(f"path_cvd {path_cvd} already exists")
        FileUtils.makedir(path_cvd)
        mat2npy(path_cvd, df_mat_sample, df_meta_sample, self.file_cpgs_cvd)
        path_t2d = os.path.join(self.temp_path, 'T2D')
        if os.path.exists(path_t2d):
            raise ValueError(f"path_t2d {path_t2d} already exists")
        FileUtils.makedir(path_t2d)
        mat2npy(path_t2d, df_mat_sample, df_meta_sample, self.file_cpgs_t2d)
        path_age = os.path.join(self.temp_path, 'Age')
        if os.path.exists(path_age):
            raise ValueError(f"path_age {path_age} already exists")
        FileUtils.makedir(path_age)

        mat2npy(path_age, df_mat_sample, df_meta_sample, self.file_cpgs_age)

        # risk score
        df_meta_cvd = risk_score(
            path_cvd, self.encoder_cvd, self.predictor_model_cvd, self.adata_cvd, df_meta_sample, 'CVD', self.device)
        df_meta_t2d = risk_score(
            path_t2d, self.encoder_t2d, self.predictor_model_t2d, self.adata_t2d, df_meta_sample, 'T2D', self.device)
        df_meta_age = age_pred(
            path_age, self.encoder_age, self.predictor_model_age, df_meta_sample, self.device)
        df_meta_age['CVD_risk'] = df_meta_cvd['Risk_score_norm']
        df_meta_age['T2D_risk'] = df_meta_t2d['Risk_score_norm']

        # 'df_meta_age' is the final output
        df_meta_age.to_csv(self.file_out, sep=',')

        return df_meta_age, self.file_out
