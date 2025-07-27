# Data Processing Pipeline

Scripts for methylation data processing and preparation.

#### **1. Raw IDAT Data Processing **

- `stroke.R`, `rawdata_process.R`, `heart_disease.R`, `850k_GMQN.R`, `diabetes.R`:
  R scripts for processing raw methylation data (IDAT files).

#### **2. Beta Matrix Generation & Metadata Cleaning **

- `heart_disease.py`, `dict_850k_ctrl.py`, `GEO_process_450k.py`, `diabetes.py`, `stroke.py`, `GEO_process_850k.py`:
  Generate Beta value matrices and clean corresponding metadata for methylation data.

#### **3. EWAS DataHub Processing**

- `merge_ewas_datahub.py`, `ewas_process_450k.py`:
  Process methylation data sourced from the EWAS DataHub database.

#### **4. Training Data Preparation**

- `Age_DataPrepare.py`, `CVD_DataPrepare.py`, `T2D_DataPrepare.py`:
  Generate formatted datasets for model training.

