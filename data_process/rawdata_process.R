# I recommend using minfi to read the raw data and do preprocess.
# But other packages can also be used. 
library(data.table)
library(minfi)
# RGSet = read.metharray.exp("idat/")
# MSet <- preprocessRaw(RGSet) # Other preprocess methods can also be used.
# m = data.frame(getMeth(MSet))
# um = data.frame(getUnmeth(MSet))
# df_signal_intensity <- 
#     read.table('/home/zhangyu/mnt_path/Data/LOLIPOP/GSE55763_unmethylated_methylated_signal_intensities.txt',
#                sep = '\t')
df_signal_intensity <- 
    fread('/home/zhangyu/mnt_path/Data/LOLIPOP/GSE55763_unmethylated_methylated_signal_intensities.txt', 
          sep = '\t', header = TRUE)
# rownames(df_signal_intensity) <- df_signal_intensity$ID_REF
row_cpgs <- df_signal_intensity$ID_REF
col_samples <- colnames(df_signal_intensity)[2:dim(df_signal_intensity)[2]]
m_colnames <- unlist(lapply(strsplit(col_samples, ' ', fixed = T), function(x){x[1]}))
m_colnames <- unique(m_colnames)

m_sel <- unlist(lapply(strsplit(col_samples, ' ', fixed = T), function(x){x[2] == 'Methylated'}))
m_sel <- col_samples[m_sel]
m_mat <- data.frame(df_signal_intensity[, ..m_sel], row.names = row_cpgs, check.names = F)
colnames(m_mat) <- m_colnames

um_sel <- unlist(lapply(strsplit(col_samples, ' ', fixed = T), function(x){x[2] == 'Unmethylated'}))
um_sel <- col_samples[um_sel]
um_mat <- data.frame(df_signal_intensity[, ..um_sel], row.names = row_cpgs, check.names = F)
colnames(um_mat) <- m_colnames

library(gmqn)
# You can skip this line if you want to use default reference.
ref_450k = set_reference(m_mat, um_mat)
# beta.GMQN.swan = gmqn_swan_parallel(m_mat, um_mat, ncpu = 40, ref = ref)
beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, ncpu = 20)

fwrite(beta.GMQN.bmiq, 
       file = '/home/zhangyu/mnt_path/Data/LOLIPOP/GSE55763_beta_GMQN_BMIQ.txt', 
       row.names = T, sep = ',')

# stroke cpgs
list_stroke_cpgs <- read.table('/home/zhangyu/mnt_path/Data/EWAS_process/disease/cpgs_blood_450k_850k_stroke_0')
list_stroke_cpgs <- list_stroke_cpgs$V1
beta.GMQN.bmiq_stroke <- beta.GMQN.bmiq[list_stroke_cpgs,]

# meta GSE55763
df_meta_GSE55763 <- read.csv('/home/zhangyu/mnt_path/Data/LOLIPOP/GSE55763_meta.txt', row.names = 'X')
rownames(df_meta_GSE55763) <- df_meta_GSE55763$ori_sample_id

