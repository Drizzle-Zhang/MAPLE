# MAPLE: Methylation Age and Disease-risk Prediction through Pairwise LEarning

## Overview
Conventional epigenetic clocks encounter challenges in generalizability, especially when there exist significant batch effects between the training and test datasets, restricting their clinical applicability for aging assessment. Here, we present MAPLE, a robust computational framework for Methylation Age and disease-risk prediction through Pairwise LEarning. MAPLE utilizes pairwise learning to discern the relative relationships between two DNA methylation profiles regarding age or disease risk. It effectively identifies aging- or disease-related biological signals while mitigating technical biases in the data. MAPLE outperforms five competing methods, achieving a median absolute error of 1.6 years across 31 benchmark tests from diverse studies, sequencing platforms, data preprocessing methods, and tissue types. Furthermore, MAPLE excels in assessing aging-related disease risk, with mean AUCs of 0.97 for disease identification and 0.85 for pre-disease status detection. In conclusion, MAPLE represents a reliable tool with great potential for accessing epigenetic age and aging-related disease risk clinically.

## Features
- **Robust Methylation Age Prediction**: Accurately predicts epigenetic age across diverse datasets
- **Disease Risk Assessment**: Evaluates risk scores for cardiovascular disease (CVD) and type 2 diabetes (T2D)
- **Platform Independence**: Compatible with various sequencing platforms and preprocessing methods
- **Technical Bias Mitigation**: Minimizes the impact of batch effects on predictions

## Environment Setup
### Prerequisites
- Python == 3.10  
- R == 4.3.3  
- curl == 7.58.0

### Installation
1. Run the setup script to create the conda environment and install Python & R packages

	```bash
	./SetupEnvironment.sh
	```

2. Activate the conda environment

	```bash
	conda activate maple
	```

3. Verify R package installation
   The script already includes version checks; the final output should look like:

	```bash
	BiocManager installed? TRUE
	Bioconductor release: 
	[1] '3.18'
	Setting options('download.file.method.GEOquery'='auto')
	Setting options('GEOquery.inmemory.gpl'=FALSE)
		 Package Installed Version
	1 data.table      TRUE  1.14.6
	2      minfi      TRUE  1.48.0
	3      ENmix      TRUE  1.38.1
	4       gmqn      TRUE   0.1.0
	```

## Model Checkpoints
To use MAPLE, you’ll need to download the model checkpoints as a compressed archive, extract the archive, and place the extracted model parameters into the "checkpoints" folder.

Download model parameters: [Google Drive](https://drive.google.com/file/d/1cYjp4UqXEgM8QHV2p-qekoadW-37itH8/view?usp=drive_link)

## Inference

1. Preprocess raw DNA methylation data (IDAT → Beta matrix)

   Input: directory of raw IDAT files  
   Output: preprocessed Beta-value matrix

	```bash
	Rscript ./raw_process/idat_process.R \
	  ./examples/RAW \
	  ./examples/input_data/Beta_values.csv
	```

2. Run MAPLE inference on preprocessed data

   Input: Beta-value matrix from preprocessing, CSV with test sample metadata  
   Output: predicted results CSV

	```bash
	python MAPLE_inference.py \
	  --input_path ./examples/input_data/Beta_values.csv \
	  --sample_info ./examples/input_data/test_meta.csv \
	  --output_path ./examples/MAPLE_output.csv
	```

3. Output

MAPLE generates a CSV file containing:
- Predicted epigenetic age
- CVD risk score
- T2D risk score

## Training

1. Training Data Preparation

​	Users can download preprocessed training data from [Google Drive](https://drive.google.com/file/d/1_O6fP066Yiq_2qdwcEv3kdAnZtfAsTYW/view?usp=drive_link) to perform model training locally.

2. Execute MAPLE Training

​	Run the following command to train the MAPLE model:

```
python MAPLE_train.py \
  --problem_type EpigeneticAge \  # Task type for training 
  --data_source ./train_dataset/epiAge_traindata.npz \  # Path to the training data  
  --path_save ./MAPLE_train_out  # Output directory for logs and trained models  
```

