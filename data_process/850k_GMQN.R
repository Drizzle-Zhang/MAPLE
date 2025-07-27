library(data.table)
library(minfi)
library(gmqn)

# copy files
copy_files <- function(path_source, list_sample, path_destination) {
    raw_files <- list.files(path_source)
    for (raw_file in raw_files) {
        sub_list <- strsplit(raw_file, '_')
        if (sub_list[[1]][1] %in% list_sample) {
            file.copy(paste0(path_source, raw_file), 
                      paste0(path_destination, raw_file))
        }
    }
}

# generate beta values
generate_beta <- function(path_in, path_out, type, sel_cpgs = NULL, ncpu = 20) {
    RGSet = read.metharray.exp(path_in, force=TRUE)
    MSet <- preprocessRaw(RGSet)
    m = data.frame(getMeth(MSet))
    um = data.frame(getUnmeth(MSet))
    if (!is.null(sel_cpgs)) {
        m <- m[sel_cpgs,]
        um <- um[sel_cpgs,]
    }
    beta.GMQN.bmiq = gmqn_bmiq_parallel(m, um, type = type, ncpu = ncpu)
    fwrite(beta.GMQN.bmiq, file = path_out, row.names = T, sep = ',')
}

############################
# 850k -> 450k
# select overlap cpgs
annon_450k_1 <- annon_450k
colnames(annon_450k_1) <- c('probe_type_450k', 'color_450k')
annon_850k_1 <- annon_850k
colnames(annon_850k_1) <- c('probe_type_850k', 'color_850k')
df_anno <- merge(annon_450k_1, annon_850k_1, by = 'row.names')
cpgs_overlap <- df_anno$Row.names

path_850k <- '/home/zhangyu/mnt_path/Data/850k/'

############################
# GSE196696
df_signal_intensity <- 
    fread('/home/zhangyu/mnt_path/Data/850k/GSE196696_signal_intensities.tsv', 
          sep = '\t', header = TRUE)
row_cpgs <- df_signal_intensity$ID_REF
col_samples <- colnames(df_signal_intensity)[2:dim(df_signal_intensity)[2]]
m_colnames <- unlist(lapply(strsplit(col_samples, ' ', fixed = T), function(x){x[1]}))
m_colnames <- unique(m_colnames)

m_sel <- unlist(lapply(strsplit(col_samples, ' ', fixed = T), function(x){x[2] == 'Methylated'}))
m_sel <- col_samples[m_sel]
m_mat <- data.frame(df_signal_intensity[, ..m_sel], row.names = row_cpgs, check.names = F)
colnames(m_mat) <- m_colnames
m_mat <- m_mat[cpgs_overlap,]

um_sel <- unlist(lapply(strsplit(col_samples, ' ', fixed = T), function(x){x[2] == 'Unmethylated'}))
um_sel <- col_samples[um_sel]
um_mat <- data.frame(df_signal_intensity[, ..um_sel], row.names = row_cpgs, check.names = F)
colnames(um_mat) <- m_colnames
um_mat <- um_mat[cpgs_overlap,]

beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, type = '450k', ncpu = 20)

fwrite(beta.GMQN.bmiq, 
       file = '/home/zhangyu/mnt_path/Data/850k/GSE196696_beta_GMQN_BMIQ_450k.txt', 
       row.names = T, sep = ',')

############################
# GSE132203
path_GSE132203 <- paste0(path_850k, "GSE132203_RAW/")
path_out_GSE132203 <- paste0(path_850k, 'GSE132203_850k_450k_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE132203, path_out_GSE132203, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)


############################
# GSE152026
df_signal_intensity <- 
    fread('/home/zhangyu/mnt_path/Data/850k/GSE152026_EUGEI_raw_Signal.csv', 
          sep = ',', header = TRUE)
row_cpgs <- df_signal_intensity$V1
col_samples <- colnames(df_signal_intensity)[2:dim(df_signal_intensity)[2]]
m_colnames <- unlist(lapply(strsplit(col_samples, '_', fixed = T), function(x){paste0(x[1], '_', x[2])}))
m_colnames <- unique(m_colnames)

m_sel <- unlist(lapply(strsplit(col_samples, '_', fixed = T), function(x){x[3] == 'Methylated'}))
m_sel <- col_samples[m_sel]
m_mat <- data.frame(df_signal_intensity[, ..m_sel], row.names = row_cpgs, check.names = F)
colnames(m_mat) <- m_colnames
m_mat <- m_mat[cpgs_overlap,]

um_sel <- unlist(lapply(strsplit(col_samples, '_', fixed = T), function(x){x[3] == 'Unmethylated'}))
um_sel <- col_samples[um_sel]
um_mat <- data.frame(df_signal_intensity[, ..um_sel], row.names = row_cpgs, check.names = F)
colnames(um_mat) <- m_colnames
um_mat <- um_mat[cpgs_overlap,]

beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, type = '450k', ncpu = 20)

fwrite(beta.GMQN.bmiq, 
       file = '/home/zhangyu/mnt_path/Data/850k/GSE152026_850k_450k_beta_GMQN_BMIQ_450k.csv', 
       row.names = T, sep = ',')


############################
# AIRWAVE GSE147740
path_AIRWAVE <- '/home/zhangyu/mnt_path/Data/AIRWAVE/'
path_GSE147740 <- paste0(path_AIRWAVE, "GSE147740_RAW/")
path_out_GSE147740 <- paste0(path_AIRWAVE, 'GSE147740_850k_450k_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE147740, path_out_GSE147740, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)


############################
# GENOA GSE210255
path_GENOA <- '/home/zhangyu/mnt_path/Data/GENOA/'
path_raw <- paste0(path_GENOA, 'GSE210256_RAW/')
path_GSE210255 <- paste0(path_GENOA, 'GSE210255_RAW/')
df_meta_GSE210255 <- read.csv(paste0(path_GENOA, "GSE210255_meta.txt"), row.names = 'X')
list_sample_GSE210255 <- df_meta_GSE210255$sample_id
copy_files(path_raw, list_sample_GSE210255, path_GSE210255)
path_out_GSE210255 <- paste0(path_GENOA, 'GSE210255_850k_450k_beta_GMQN_BMIQ_replication.txt')
generate_beta(path_GSE210255, path_out_GSE210255, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)





############################
############################
# 850k
path_850k <- '/home/zhangyu/mnt_path/Data/850k/'

############################
# GSE196696
df_signal_intensity <- 
    fread('/home/zhangyu/mnt_path/Data/850k/GSE196696_signal_intensities.tsv', 
          sep = '\t', header = TRUE)
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

beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, type = '850k', ncpu = 20)

fwrite(beta.GMQN.bmiq, 
       file = '/home/zhangyu/mnt_path/Data/850k/GSE196696_beta_GMQN_BMIQ.txt', 
       row.names = T, sep = ',')

############################
# GSE132203
path_GSE132203 <- paste0(path_850k, "GSE132203_RAW/")
path_out_GSE132203 <- paste0(path_850k, 'GSE132203_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE132203, path_out_GSE132203, type = '850k', ncpu = 20)


############################
# GSE152026
df_signal_intensity <- 
    fread('/home/zhangyu/mnt_path/Data/850k/GSE152026_EUGEI_raw_Signal.csv', 
          sep = ',', header = TRUE)
row_cpgs <- df_signal_intensity$V1
col_samples <- colnames(df_signal_intensity)[2:dim(df_signal_intensity)[2]]
m_colnames <- unlist(lapply(strsplit(col_samples, '_', fixed = T), function(x){paste0(x[1], '_', x[2])}))
m_colnames <- unique(m_colnames)

m_sel <- unlist(lapply(strsplit(col_samples, '_', fixed = T), function(x){x[3] == 'Methylated'}))
m_sel <- col_samples[m_sel]
m_mat <- data.frame(df_signal_intensity[, ..m_sel], row.names = row_cpgs, check.names = F)
colnames(m_mat) <- m_colnames

um_sel <- unlist(lapply(strsplit(col_samples, '_', fixed = T), function(x){x[3] == 'Unmethylated'}))
um_sel <- col_samples[um_sel]
um_mat <- data.frame(df_signal_intensity[, ..um_sel], row.names = row_cpgs, check.names = F)
colnames(um_mat) <- m_colnames

beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, type = '850k', ncpu = 20)

fwrite(beta.GMQN.bmiq, 
       file = '/home/zhangyu/mnt_path/Data/850k/GSE152026_beta_GMQN_BMIQ.csv', 
       row.names = T, sep = ',')


############################
# AIRWAVE GSE147740
path_AIRWAVE <- '/home/zhangyu/mnt_path/Data/AIRWAVE/'
path_GSE147740 <- paste0(path_AIRWAVE, "GSE147740_RAW/")
path_out_GSE147740 <- paste0(path_AIRWAVE, 'GSE147740_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE147740, path_out_GSE147740, type = '850k', ncpu = 20)


############################
# GENOA GSE210255
path_GENOA <- '/home/zhangyu/mnt_path/Data/GENOA/'
path_raw <- paste0(path_GENOA, 'GSE210256_RAW/')
path_GSE210255 <- paste0(path_GENOA, 'GSE210255_RAW/')
df_meta_GSE210255 <- read.csv(paste0(path_GENOA, "GSE210255_meta.txt"), row.names = 'X')
list_sample_GSE210255 <- df_meta_GSE210255$sample_id
copy_files(path_raw, list_sample_GSE210255, path_GSE210255)
path_out_GSE210255 <- paste0(path_GENOA, 'GSE210255_beta_GMQN_BMIQ.txt')
generate_beta(path_GSE210255, path_out_GSE210255, type = '850k', ncpu = 20)

