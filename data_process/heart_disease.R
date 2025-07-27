library(data.table)
library(minfi)
library(ENmix)
library(gmqn)

# select overlap cpgs
annon_450k_1 <- annon_450k
colnames(annon_450k_1) <- c('probe_type_450k', 'color_450k')
annon_850k_1 <- annon_850k
colnames(annon_850k_1) <- c('probe_type_850k', 'color_850k')
df_anno <- merge(annon_450k_1, annon_850k_1, by = 'row.names')
cpgs_overlap <- df_anno$Row.names


path_HD <- '/home/zhangyu/mnt_path/Data/heart_disease/'


######################
# GSE87016 helthy + coronary artery ectasia
# path_GSE197080 <- paste0(path_stroke, "GSE197080_RAW/")
# path_out_GSE197080 <- paste0(path_stroke, 'GSE197080_850k_450k_beta_GMQN_BMIQ.csv')
# generate_beta(path_GSE197080, path_out_GSE197080, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 6)
df_signal_intensity <- 
    fread(paste0(path_HD, 'GSE87016_Unmethylated_and_methylated_signal.txt'), 
          sep = '\t', header = TRUE)
row_cpgs <- df_signal_intensity$TargetID
col_samples <- colnames(df_signal_intensity)[2:dim(df_signal_intensity)[2]]
m_colnames <- unlist(lapply(strsplit(col_samples, '.', fixed = T), 
                            function(x){gsub(' ', '_', x[1])}))
m_colnames <- paste(unique(m_colnames), sep = '_')

m_sel <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[2] == 'Methylated signal'}))
m_sel <- col_samples[m_sel]
m_mat <- data.frame(df_signal_intensity[, ..m_sel], row.names = row_cpgs, check.names = F)
colnames(m_mat) <- m_colnames
m_mat <- m_mat[cpgs_overlap,]

um_sel <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[2] == 'Unmethylated signal'}))
um_sel <- col_samples[um_sel]
um_mat <- data.frame(df_signal_intensity[, ..um_sel], row.names = row_cpgs, check.names = F)
colnames(um_mat) <- m_colnames
um_mat <- um_mat[cpgs_overlap,]

beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, ncpu = 20)

fwrite(beta.GMQN.bmiq, 
       file = paste0(path_HD, 'GSE87016_beta_GMQN_BMIQ.csv'), 
       row.names = T, sep = ',')

######################
path_HD = '/home/zhangyu/mnt_path/Data/heart_disease/'
df_mat_HD= fread(paste0(path_HD, 'GSE56046_methylome_normalized.txt'), sep='\t', header = TRUE)
sample_names <- colnames(df_mat_HD)[2:dim(df_mat_HD)[2]]
cpg_names <- df_mat_HD$ID_REF
df_mat_HD <- as.data.frame(df_mat_HD[,2:dim(df_mat_HD)[2]])
rownames(df_mat_HD) <- cpg_names
df_meta_HD <- read.csv(paste0(path_HD, 'GSE56046_process/GSE56046_meta.txt'), row.names = 'X')
samples_HD <- paste0(rownames(df_meta_HD), ".Mvalue")
df_mat_all = df_mat_HD[, samples_HD]
df_mat_all_beta = (1.0001*exp(df_mat_all)-0.0001)/(1+exp(df_mat_all))
colnames(df_mat_all_beta) = df_meta_HD[,'sample_id']
df_mat_all_beta <- as.data.frame(df_mat_all_beta)
fwrite(df_mat_all_beta, 
       file = paste0(path_HD, 'GSE56046_process/GSE56046_beta.csv'), row.names = T, sep = ',')

path_HD = '/home/zhangyu/mnt_path/Data/heart_disease/'
df_mat_HD= fread(paste0(path_HD, 'GSE220622_processed_M-value.tsv'), sep='\t', header = TRUE)
sample_names <- colnames(df_mat_HD)[2:dim(df_mat_HD)[2]]
cpg_names <- df_mat_HD$ID_REF
df_mat_HD <- as.data.frame(df_mat_HD[,2:dim(df_mat_HD)[2]])
rownames(df_mat_HD) <- cpg_names
df_meta_HD <- read.csv(paste0(path_HD, 'GSE220622_process/GSE220622_meta.txt'), row.names = 'X')
sample_sel <- unlist(lapply(strsplit(sample_names, '_', fixed = T), function(x){x[1] == 'SUAZ'}))
sample_sel <- sample_names[sample_sel]
df_mat_all = df_mat_HD[, sample_sel]
df_mat_all_beta = (1.0001*exp(df_mat_all)-0.0001)/(1+exp(df_mat_all))
colnames(df_mat_all_beta) = df_meta_HD[,'sample_id']
df_mat_all_beta <- as.data.frame(df_mat_all_beta)
fwrite(df_mat_all_beta, 
       file = paste0(path_HD, 'GSE220622_process/GSE220622_beta.csv'), row.names = T, sep = ',')

