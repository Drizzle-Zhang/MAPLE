import os
import shutil
import json
from bioage_pipeline import BioAgePipeline
from config import Config
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Predict epigenetic age and disease risk score')
    parser.add_argument('--input_format', type=str, choices = ("idat","beta_value"), default='idat')
    parser.add_argument('--input_path', type=str, default='./examples/input_data')
    parser.add_argument('--sample_info', type=str, default='./examples/test_meta.csv')
    parser.add_argument('--output_path', type=str, default='./examples/MAPLE_output.csv')

    args = parser.parse_args()

    return args


def predict(args, conf):
    meta_file_name = args.sample_info
    conf.input_format = args.input_format
    ppl = BioAgePipeline(conf)
    if args.input_format == 'idat':
        idat_file_names = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path) if file[-5:] == '.idat']
        beta_file_name = ppl.idat2mat(idat_file_names)
    else:
        beta_file_name = os.path.join(args.input_path, 'Beta_values.csv')
    df_mat_res, file_res = ppl.predict_beta(beta_file_name, meta_file_name)

    shutil.move(file_res, args.output_path)

    return


if __name__ == '__main__':

    args = parse_args()

    with open("./config.json", 'r') as fp:
        dict_conf = json.loads(fp.read())
    conf = Config.from_dict(dict_conf)

    start_time = time.time()
    predict(args, conf)
    end_time = time.time()
    print(f'predict time: {end_time - start_time}s')
