# Python
conda create -n maple python=3.10 -y
conda activate maple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# R
Rscript ./raw_process/install_pkgs.R
Rscript ./raw_process/verify_version.R