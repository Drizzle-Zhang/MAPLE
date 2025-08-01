
#%%
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.backends.cudnn as cudnn
import PIL.Image
from torchvision.transforms import ToTensor
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import os
import argparse
import time
import random
import logging

from utils.file_utils import read_stringList_FromFile, FileUtils
from utils.model_helper import ModelUtils, get_regularization, LossSoft
from models.FC import FCNet_H_class, FCNet_H
from models.CT import ContrastiveEncoder, ContrastiveDescriminator, ContrastiveDescriminatorClass
from utils.dataload_utils import geo_npz_Dataset_train
from utils.evaluate_helper import classification_eval, methylage_evaluate_score



def evaluate_age(encoder_model, predictor_model, loader, args, logger, prefix, epoch):
    age_norm_value = 100.0

    with torch.no_grad():
        for batch_idx, (feature, targets, additional) in enumerate(loader):

            feature = feature.to(device)
            targets = targets.to(device)

            latent_feature = encoder_model(feature)
            predicts = predictor_model(latent_feature) * age_norm_value

            [mae_value, rmse_value, r_value,  medae_value] = methylage_evaluate_score(predicts, targets)
            me_value = torch.mean(predicts - targets)
            if args.save_log == 'True':
                logger.info("#"*20)
                logger.info( f"epoch:{epoch:3d}, {prefix} me:{me_value:6.3f}, mae:{mae_value:6.3f}, "
                             f"rmse:{rmse_value:6.3f}, R:{r_value:3.3f}, Med:{medae_value:6.3f}" )
            break

    return mae_value, rmse_value, r_value, medae_value


def evaluate_risk(encoder_model, decoder_model, predictor_model, criterion, loader, prefix, epoch):

    with torch.no_grad():
        for batch_idx, (feature, targets, additional) in enumerate(loader):
            for batch_idx2, (feature2, targets2, additional2) in enumerate(loader):
                feature = feature.to(device)
                feature2 = feature2.to(device)
                encoder1 = encoder_model(feature)
                encoder2 = encoder_model(feature2)
                concat_feature = torch.concat([encoder1, encoder2, encoder1- encoder2], dim = 1)
                targets = targets.to(device)
                targets2 = targets2.to(device)
                bool_mrs = (targets > -10)
                bool_mrs2 = (targets2 > -10)
                predict_mrs_diff = decoder_model(concat_feature)
                true_mrs_diff =  (targets[bool_mrs & bool_mrs2] - targets2[bool_mrs & bool_mrs2])
                predict_mrs_diff = predict_mrs_diff[bool_mrs & bool_mrs2]
                true_mrs_diff = true_mrs_diff.view((true_mrs_diff.shape[0], 1)).float()
                predict_mrs_diff = predict_mrs_diff.view((true_mrs_diff.shape[0], 1)).float()
                # import pdb; pdb.set_trace()
                if len(true_mrs_diff) > 5:
                    loss_mrs = criterion(predict_mrs_diff, true_mrs_diff)
                    [mae_value, rmse_value, R_value,  medae_value] = \
                        methylage_evaluate_score(predict_mrs_diff, true_mrs_diff)
                    logger.info( f"epoch:{epoch:3d}, {prefix}, mrs_loss: {loss_mrs:6.3f}, "
                                 f"mae:{mae_value:6.3f}, {prefix} rmse:{rmse_value:6.3f}, R:{R_value:3.3f}, Med:{medae_value:6.3f}" )
                else:
                    loss_mrs = 0
                break
            break

    with torch.no_grad():
        if prefix != "train":
            list_pred = []
            list_true = []
            for batch_idx, (feature, targets, additional) in enumerate(loader):
                feature = feature.to(device)
                true_status = additional['type_index'].to(device)
                latent_feature = encoder_model(feature)
                predicts = predictor_model(latent_feature)
                list_pred.append(predicts)
                list_true.append(true_status)

            acc_val, precision_val, recall_val, f1_val, roc_val, prc_val = \
                classification_eval(torch.cat(list_pred), torch.cat(list_true))
            logger.info("#"*20)
            logger.info(
                f"{prefix} epoch:{epoch:3d}, Acc:{acc_val:.3f}, precision:{precision_val:.3f}, "
                f"Recall:{recall_val:3.3f}, F1:{f1_val:6.3f}, "
                f"AUROC:{roc_val:3.3f}, AUPRC:{prc_val:6.3f}")

            if np.isnan(f1_val):
                f1_val = 0
            if np.isnan(roc_val):
                roc_val = 0
            if np.isnan(prc_val):
                prc_val = 0

            return medae_value, acc_val, f1_val, roc_val, prc_val

    return


def save_embedding(encoder_model, predictor_model, loader, prefix, args):
    age_norm_value = 100.0
    encoder_model.eval()

    with torch.no_grad():
        sample_list = []
        tissue_list = []
        for batch_idx, (feature, targets,additional) in enumerate(loader):
            feature = feature.to(device)
            if args.problem_type == 'EpigeneticAge':
                targets = targets.to(device)
            else:
                targets = additional['type_index'].to(device)

            latent_feature = encoder_model(feature)
            if args.problem_type == 'EpigeneticAge':
                predicts = predictor_model(latent_feature) * age_norm_value
            else:
                predicts = predictor_model(latent_feature)

            latent_feature = latent_feature.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            predicts = predicts.cpu().detach().numpy()

            if batch_idx == 0:
                latents_concat = latent_feature
                targets_concat = targets
                predicts_concat = predicts
            else:
                latents_concat = np.concatenate([latents_concat, latent_feature], axis = 0)
                targets_concat = np.concatenate([targets_concat, targets], axis = 0)
                predicts_concat =  np.concatenate([predicts_concat, predicts], axis = 0)

            sample_list += additional["sample_id"]
            tissue_list += additional["tissue"]

    np.savez(os.path.join(
        args.path_save, prefix + "_" + "best_epoch_embedding.npz"),
        latents = latents_concat,
        targets = targets_concat,
        predicts_concat = predicts_concat,
        sample_list =  sample_list,
        tissue_list = tissue_list
    )

    return


def train_risk(args, logger):
    # load data
    train_dataset = geo_npz_Dataset_train(file_npy = args.data_source, data_type='train')
    val_dataset = geo_npz_Dataset_train(file_npy = args.data_source, data_type='val')
    pretrain_dataset = geo_npz_Dataset_train(file_npy = args.data_source, data_type='pretrain')
    data_npy = np.load(args.data_source, allow_pickle=True)
    feature_size = len(data_npy['cpgs'])

    logger.info(f"Num of train samples: {len(train_dataset)} \n"
                f"Num of val samples: {len(val_dataset)} \n")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size = args.batch_size,
                                  shuffle= True,
                                  drop_last = True )
    val_dataloader = DataLoader(val_dataset,
                                 batch_size = args.batch_size * 3,
                                 shuffle= True )

    # models
    encoder_hidden_list = [int(i.strip()) for i in args.encoder_hidden_str.split(",")]
    decoder_hidden_list = [int(i.strip()) for i in args.decoder_hidden_str.split(",")]
    encoder = ContrastiveEncoder(feature_channel = feature_size,
                                 hidden_list= encoder_hidden_list,
                                 h_dim = args.latent_size,
                                 if_dp=False )
    decoder = ContrastiveDescriminator(feature_channel = args.latent_size*3,
                                           hidden_list= decoder_hidden_list,
                                           h_dim = 1,
                                           if_dp=False )
    predictor_model = FCNet_H_class(feature_channel = args.latent_size,
                                    output_channel = 2,
                                    hidden_list = decoder_hidden_list,
                                    if_bn = False,
                                    if_dp = False)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        predictor_model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        predictor_model = nn.DataParallel(predictor_model)

    # loss function
    criterion_soft = LossSoft(epsilon=0.1)
    criterion_ce = nn.CrossEntropyLoss()
    contrast_optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.learning_rate},
        {'params': decoder.parameters(), 'lr': args.learning_rate}
    ], weight_decay=1e-4)
    predictor_optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.learning_rate*0.5},
        {'params': predictor_model.parameters(), 'lr': args.learning_rate*0.2}
    ])

    # training
    project_id_list = [pretrain_dataset[i][2]["project_id"] for i, _ in enumerate(pretrain_dataset)]
    counter_dataset = Counter(project_id_list)
    project_id_list = [elem for elem, freq in counter_dataset.items() if freq > 30]
    val_interval = 10
    best_mae = 0.1
    best_f1 = 0.1
    best_auc = 0.5
    best_epoch = 0
    for epoch in range(1, args.num_epochs+1):
        encoder.train()
        decoder.train()
        predictor_model.train()
        # dataset_id = random.choice(project_id_list)
        list_loss = []
        for dataset_id in random.sample(project_id_list, 3):
            training_dataset_batch = tuple(
                pretrain_dataset[i] for i in range(len(pretrain_dataset)) if pretrain_dataset[i][2]['project_id'] == dataset_id)
            train_dataloader_batch = DataLoader(training_dataset_batch,
                                                batch_size = args.batch_size,
                                                shuffle= True)

            for batch_idx, (feature, targets, additional) in enumerate(train_dataloader_batch):
                for batch_idx2, (feature2, targets2, additional2) in enumerate(train_dataloader_batch):
                    feature = feature.to(device)
                    feature2 = feature2.to(device)
                    targets = targets.to(device)
                    targets2 = targets2.to(device)

                    encoder1 = encoder(feature)
                    encoder2 = encoder(feature2)
                    concat_feature = torch.concat([encoder1, encoder2, encoder1 - encoder2], dim = 1)
                    predict_mrs = decoder(concat_feature)
                    true_mrs_diff =  (targets - targets2)
                    if len(true_mrs_diff) > 2:
                        loss_mrs = criterion_soft(predict_mrs, true_mrs_diff)
                    else:
                        loss_mrs = 0
                    if args.encoder_regularization == 'True':
                        loss_reg = get_regularization(encoder) * args.coef_regularization
                    else:
                        loss_reg = 0
                    loss =  loss_mrs + loss_reg

                    if loss > 1:
                        import pdb; pdb.set_trace()

                    contrast_optimizer.zero_grad( )
                    loss.backward()
                    contrast_optimizer.step()
                    list_loss.append(loss.cpu().detach().numpy())
                    break
                break
        if args.save_log == 'True':
            logger.info( f"epoch:{epoch:3d}, train mrs loss:{np.mean(list_loss):6.5f}" )

        if epoch % val_interval == 0:
            for i in range(val_interval//2):
                for batch_idx, (feature, targets,additional) in enumerate(train_dataloader):
                    feature = feature.to(device)
                    true_status = additional['type_index'].to(device)
                    latent_feature = encoder(feature)
                    predicts = predictor_model(latent_feature)
                    loss_ce = criterion_ce(predicts, true_status)
                    predictor_optimizer.zero_grad()
                    loss_ce.backward()
                    predictor_optimizer.step()
                    break

            encoder.eval()
            decoder.eval()
            predictor_model.eval()
            evaluate_risk(
                encoder, decoder, predictor_model, criterion_soft, train_dataloader, "train", epoch
            )
            mae_val, acc_val, f1_val, roc_val, prc_val = evaluate_risk(
                encoder, decoder, predictor_model, criterion_soft, val_dataloader, "validation", epoch
            )

            if (mae_val < best_mae) & (f1_val > best_f1) & (prc_val > best_auc):
                best_mae = mae_val
                best_f1 = f1_val
                best_auc = prc_val
                best_epoch = epoch
                if args.save_model == 'True':
                    if torch.cuda.device_count() > 1:
                        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                      "predictor_state_dict": predictor_model.module.state_dict()}
                    else:
                        checkpoint ={"encoder_state_dict": encoder.state_dict(),
                                     "predictor_state_dict": predictor_model.state_dict()}
                    filename = os.path.join(os.path.join(args.path_save, 'checkpoints'), "best_model.pt")
                    ModelUtils.save_checkpoint(checkpoint, filename=filename)
                if args.save_embeddings == 'True':
                    save_embedding(encoder, predictor_model, train_dataloader, "train", args)
                    save_embedding(encoder, predictor_model, val_dataloader,  "val" , args)
            if args.save_log == 'True':
                logger.info( f"Best epoch:{best_epoch:3d}, ct_MAE:{best_mae:6.5f}, "
                             f"F1:{best_mae:6.5f}, AUPRC:{best_mae:6.5f}" )

            # early stop
            if (epoch - best_epoch >= args.patience_epoch) & (best_epoch > 0):
                print("Model have been saved in 'best_model.pt'")
                return

    return


def train_age(args, logger):
    # load data
    train_dataset = geo_npz_Dataset_train(file_npy = args.data_source, data_type='train')
    val_dataset = geo_npz_Dataset_train(file_npy = args.data_source, data_type='val')
    data_npy = np.load(args.data_source, allow_pickle=True)
    feature_size = len(data_npy['cpgs'])
    logger.info(f"Num of train samples: {len(train_dataset)} \n"
                f"Num of val samples: {len(val_dataset)} \n"
                )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size = args.batch_size,
                                  shuffle= True,
                                  drop_last = False )
    val_dataloader = DataLoader(val_dataset,
                                 batch_size = args.batch_size * 10,
                                 shuffle= True )

    # models
    encoder_hidden_list = [int(i.strip()) for i in args.encoder_hidden_str.split(",")]
    decoder_hidden_list = [int(i.strip()) for i in args.decoder_hidden_str.split(",")]
    encoder = ContrastiveEncoder(feature_channel = feature_size,
                                 hidden_list= encoder_hidden_list,
                                 h_dim = args.latent_size )
    decoder = ContrastiveDescriminator(feature_channel = args.latent_size*3,
                                       hidden_list= decoder_hidden_list,
                                       h_dim = 1 )
    predictor_model = FCNet_H(feature_channel = args.latent_size,
                              output_channel = 1,
                              hidden_list = decoder_hidden_list,
                              if_bn = False,
                              if_dp = False)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        predictor_model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        predictor_model = nn.DataParallel(predictor_model)

    # loss function
    age_norm_value = 100.0
    huber_delta = args.huber_delta
    criterion_huber = nn.HuberLoss(delta = huber_delta/age_norm_value)

    # optimizer
    contrast_optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.learning_rate},
        {'params': decoder.parameters(), 'lr': args.learning_rate},
    ])
    predictor_optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.learning_rate*0.5},
        {'params': predictor_model.parameters(), 'lr': args.learning_rate*0.2}
    ])


    # training
    val_interval = 10
    best_mae = 100.0
    best_epoch = 0
    for epoch in range(1, args.num_epochs+1):
        encoder.train()
        decoder.train()

        for batch_idx, (feature, targets,additional) in enumerate(train_dataloader):
            for batch_idx2, (feature2, targets2,additional2) in enumerate(train_dataloader):

                feature = feature.to(device)
                feature2 = feature2.to(device)
                targets = targets.to(device) / age_norm_value
                targets2 = targets2.to(device) / age_norm_value

                encoder1 = encoder(feature)
                encoder2 = encoder(feature2)
                concat_feature = torch.concat([encoder1, encoder2, encoder1 - encoder2], dim = 1)
                predict_value = decoder(concat_feature)

                true_diff =  (targets - targets2)
                loss_regression =  criterion_huber(predict_value, true_diff)
                if args.encoder_regularization == 'True':
                    loss_reg = get_regularization(encoder) * args.coef_regularization
                    loss = loss_regression + loss_reg
                else:
                    loss = loss_regression

                contrast_optimizer.zero_grad()
                loss.backward()
                contrast_optimizer.step()

                if args.save_log == 'True':
                    logger.info( f"epoch:{epoch:3d}, train loss:{loss:6.5f}" )

                break
            break

        if epoch % val_interval == 0:
            for i in range(5):
                for batch_idx, (feature, targets,additional) in enumerate(train_dataloader):
                    feature = feature.to(device)
                    targets = targets.to(device) / age_norm_value

                    latent_feature = encoder(feature)
                    predicts = predictor_model(latent_feature)
                    grad_loss = criterion_huber(predicts, targets)

                    predictor_optimizer.zero_grad()
                    grad_loss.backward()
                    predictor_optimizer.step()

            encoder.eval()
            decoder.eval()
            predictor_model.eval()
            evaluate_age(encoder, predictor_model, train_dataloader, args, logger, prefix = "train", epoch = epoch)
            mae_val, rmse_val, r_val, medae_val = \
                evaluate_age(encoder, predictor_model, val_dataloader, args, logger,  prefix = "validation", epoch = epoch)

            if mae_val < best_mae:
                best_mae = mae_val
                best_epoch = epoch
                if args.save_model == 'True':
                    if best_mae < 3:
                        if torch.cuda.device_count() > 1:
                            checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                          "predictor_state_dict": predictor_model.module.state_dict()}
                        else:
                            checkpoint ={"encoder_state_dict": encoder.state_dict(),
                                         "predictor_state_dict": predictor_model.state_dict()}
                        filename = os.path.join(os.path.join(args.path_save, 'checkpoints'), "best_model.pt")
                        ModelUtils.save_checkpoint(checkpoint, filename=filename)
                if args.save_embeddings == 'True':
                    if best_mae < 3:
                        save_embedding(encoder, predictor_model , train_dataloader, prefix =  "train", args = args)
                        save_embedding(encoder, predictor_model, val_dataloader,  prefix = "val", args = args)
            if args.save_log == 'True':
                logger.info( f"Best epoch:{best_epoch:3d}, MAE:{best_mae:6.5f}" )

            # early stop
            if epoch - best_epoch >= args.patience_epoch:
                print("Model have been saved in 'best_model.pt'")
                # torch.save(model, os.path.join(self.ckpt_path, f"model_param_epoch_{epoch}.pt"))
                return

            encoder.train()
            decoder.train()
            predictor_model.train()

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Train methylation')
    parser.add_argument('--problem_type', type=str, choices = ("EpigeneticAge","DiseaseRisk"), default="EpigeneticAge")
    parser.add_argument('--data_source', type=str, default="./train_dataset/epiAge_traindata.npz")
    parser.add_argument("--encoder_hidden_str", type=str, default= "1024,1024,512,512,256,256,64,64")
    parser.add_argument("--decoder_hidden_str", type=str, default= "32,32,16,16")
    parser.add_argument("--latent_size", type=int, default= 32)
    parser.add_argument("--encoder_regularization", type=str, choices=('True', 'False'), default='False')
    parser.add_argument("--coef_regularization", type=float, default=1e-4)
    parser.add_argument("--huber_delta", type=float, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1500,
                        help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=6000,
                        help='num_epochs')
    parser.add_argument('--patience_epoch', type=int, default=100)
    parser.add_argument('--save_model', choices=('True', 'False'), default="True")
    parser.add_argument('--save_embeddings', choices=('True', 'False'),
                        type=str, default="False")
    parser.add_argument('--save_log', type=str, choices=('True', 'False'), default='True')
    parser.add_argument('--path_save', type=str, default='./MAPLE_train_out')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.save_embeddings == 'True':
        FileUtils.makedir(os.path.join(args.path_save, 'results'))
    if args.save_model == 'True':
        FileUtils.makedir(os.path.join(args.path_save, 'checkpoints'))


    if args.save_log == 'True':
        FileUtils.makedir(os.path.join(args.path_save, 'logs'))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logger = logging.getLogger(os.path.join(args.path_save, 'logs'))
        logger.setLevel(logging.INFO)
        log_file = os.path.join(os.path.join(args.path_save, 'logs'), "train_log.txt")
        if os.path.exists(log_file):
            os.remove(log_file)

        filehandler = logging.FileHandler(log_file)
        streamhandler = logging.StreamHandler()
        filehandler.setLevel(logging.INFO)
        streamhandler.setLevel(logging.INFO)
        logger.addHandler(streamhandler)
        logger.addHandler(filehandler)

        logger.info( f"seed:{args.seed:3d} \n"
                     f"data_source: {args.data_source} \n"
                     f"encoder_hidden_str: {args.encoder_hidden_str} \n"
                     f"decoder_hidden_str: {args.decoder_hidden_str} \n"
                     f"latent_size: {args.latent_size} \n"
                     f"encoder_regularization: {args.encoder_regularization} \n"
                     f"coef_regularization: {args.coef_regularization} \n"
                     f"huber_delta: {args.huber_delta} \n"
                     f"learning_rate: {args.learning_rate} \n"
                     f"batch_size: {args.batch_size} \n"
                     f"num_epochs: {args.num_epochs} \n")

    # random variable
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        cudnn.deterministic = True
    torch.set_printoptions(precision = 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_start = time.time()
    if args.problem_type == 'EpigeneticAge':
        train_age(args, logger)
    else:
        train_risk(args, logger)
    time_end = time.time()
    print(time_end - time_start)
