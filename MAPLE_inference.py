import os
import shutil
import json
import time
import argparse
from bioage_pipeline import BioAgePipeline
from config import Config


def parse_args():
    """
    Parse command line arguments for MAPLE inference.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Predict epigenetic age and disease risk score')
    parser.add_argument('--input_path', type=str, default='./examples/input_data',
                        help='Path to input DNA methylation data')
    parser.add_argument('--sample_info', type=str, default='./examples/test_meta.csv',
                        help='Path to sample metadata file')
    parser.add_argument('--output_path', type=str, default='./examples/MAPLE_output.csv',
                        help='Path for output predictions')

    args = parser.parse_args()
    return args


def predict(args, conf):
    """
    Run prediction pipeline using the provided arguments and configuration.
    
    Args:
        args (argparse.Namespace): Command line arguments
        conf (Config): Configuration object
    
    Returns:
        None
    """
    meta_file_name = args.sample_info    
    # Initialize pipeline
    ppl = BioAgePipeline(conf)
    
    # Use provided beta value matrix
    beta_file_name = args.input_path
    
    # Run prediction and get results
    df_mat_res, file_res = ppl.predict_beta(beta_file_name, meta_file_name)

    # Move results to output path
    shutil.move(file_res, args.output_path)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Load configuration from JSON file
    with open("./config.json", 'r') as fp:
        dict_conf = json.loads(fp.read())
    conf = Config.from_dict(dict_conf)

    # Run prediction with timing
    start_time = time.time()
    predict(args, conf)
    end_time = time.time()
    print(f'Prediction completed in {end_time - start_time:.2f} seconds')
