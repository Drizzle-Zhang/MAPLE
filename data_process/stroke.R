library(data.table)
library(minfi)
library(ENmix)
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

set_reference <- function(m, um, type = '450k') {
    
    set.seed(1)
    
    if (sum(row.names(m) == row.names(um)) != dim(m)[1]) {
        stop('The rownames of m and um are different!')
    }
    print('Processing the input data---------------------------------------')
    probe = row.names(m)
    m = as.numeric(apply(m, 1, mean))
    um = as.numeric(apply(um, 1, mean))
    
    if (type == '450k') {
        t1.red <- row.names(annon_450k)[which(annon_450k$color == 'Red' & annon_450k$probe_type == 'I')]
        t1.green <- row.names(annon_450k)[which(annon_450k$color == 'Grn' & annon_450k$probe_type == 'I')]
        type <- annon_450k[probe, 'probe_type']
    } else if (type == '850k') {
        t1.red <- row.names(annon_850k)[which(annon_850k$color == 'Red' & annon_850k$probe_type == 'I')]
        t1.green <- row.names(annon_850k)[which(annon_850k$color == 'Grn' & annon_850k$probe_type == 'I')]
        type <- annon_850k[probe, 'probe_type']
    }
    
    t1.red.index <- match(t1.red, probe)
    t1.red.index <- t1.red.index[!is.na(t1.red.index)]
    t1.red.signal <- c(m[t1.red.index], um[t1.red.index])
    t1.red.model <- Mclust(t1.red.signal, G=2, verbose = F)
    t1.red.ref.mean <- t1.red.model$parameters$mean
    t1.red.ref.sd <- sqrt(t1.red.model$parameters$variance$sigmasq)
    
    t1.green.index <- match(t1.green,probe)
    t1.green.index <- t1.green.index[!is.na(t1.green.index)]
    t1.green.signal <- c(m[t1.green.index],um[t1.green.index])
    t1.green.model <- Mclust(t1.green.signal, G=2, verbose = F)
    t1.green.ref.mean <- t1.green.model$parameters$mean
    t1.green.ref.sd <- sqrt(t1.green.model$parameters$variance$sigmasq)
    
    return(list(t1.green.ref.mean = t1.green.ref.mean,
                t1.green.ref.sd = t1.green.ref.sd,
                t1.red.ref.mean = t1.red.ref.mean,
                t1.red.ref.sd = t1.red.ref.sd))
    
}

# select overlap cpgs
annon_450k_1 <- annon_450k
colnames(annon_450k_1) <- c('probe_type_450k', 'color_450k')
annon_850k_1 <- annon_850k
colnames(annon_850k_1) <- c('probe_type_850k', 'color_850k')
df_anno <- merge(annon_450k_1, annon_850k_1, by = 'row.names')
cpgs_overlap <- df_anno$Row.names


path_stroke <- '/home/zhangyu/mnt_path/Data/stroke/'

######################
# GSE203399 Discovery 450k
df_meta_GSE203399_1 <- read.csv(
    paste0(path_stroke, "GSE203399_process/GSE203399-GPL13534_meta.txt"), row.names = 'X')
list_sample_discovery <- df_meta_GSE203399_1$sample_id

path_raw <- paste0(path_stroke, "GSE203399_RAW/")
path_discovery <- paste0(path_raw, "Discovery_450k/")
copy_files(path_raw, list_sample_discovery, path_discovery)
path_out_discovery <- paste0(path_stroke, 'GSE203399_450k_beta_GMQN_BMIQ_discovery.txt')
generate_beta(path_discovery, path_out_discovery, type = '450k', ncpu = 20)


######################
# GSE203399 Replication
df_meta_GSE203399_2 <- read.csv(
    paste0(path_stroke, "GSE203399_process/GSE203399_850k_meta.txt"), row.names = 'X')
list_sample_450k <- df_meta_GSE203399_2[1:32, 'sample_id']
list_sample_850k <- df_meta_GSE203399_2[33:dim(df_meta_GSE203399_2)[1], 'sample_id']

path_raw <- paste0(path_stroke, "GSE203399_RAW/")
path_450k <- paste0(path_raw, "Replication_450k/")
path_850k <- paste0(path_raw, "Replication_850k/")
# 450k
copy_files(path_raw, list_sample_450k, path_450k)
path_out_450k <- paste0(path_stroke, 'GSE203399_450k_beta_GMQN_BMIQ_replication.txt')
generate_beta(path_450k, path_out_450k, type = '450k', ncpu = 20)

# 850k
copy_files(path_raw, list_sample_850k, path_850k)
path_out_850k <- paste0(path_stroke, 'GSE203399_850k_450k_beta_GMQN_BMIQ_replication.txt')
generate_beta(path_850k, path_out_850k, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)

######################
# GSE197080 helthy + stroke
# path_GSE197080 <- paste0(path_stroke, "GSE197080_RAW/")
# path_out_GSE197080 <- paste0(path_stroke, 'GSE197080_850k_450k_beta_GMQN_BMIQ.csv')
# generate_beta(path_GSE197080, path_out_GSE197080, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 6)
df_signal_intensity <- 
    fread('/home/zhangyu/mnt_path/Data/stroke/GSE197080_matrix_signal.txt', 
          sep = '\t', header = TRUE)
row_cpgs <- df_signal_intensity$TargetID
col_samples <- colnames(df_signal_intensity)[3:dim(df_signal_intensity)[2]]
m_colnames <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[1]}))
m_colnames <- unique(m_colnames)

m_sel <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[2] == 'Signal_B'}))
m_sel <- col_samples[m_sel]
m_mat <- data.frame(df_signal_intensity[, ..m_sel], row.names = row_cpgs, check.names = F)
colnames(m_mat) <- m_colnames
m_mat <- m_mat[cpgs_overlap,]

um_sel <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[2] == 'Signal_A'}))
um_sel <- col_samples[um_sel]
um_mat <- data.frame(df_signal_intensity[, ..um_sel], row.names = row_cpgs, check.names = F)
colnames(um_mat) <- m_colnames
um_mat <- um_mat[cpgs_overlap,]

beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, ncpu = 6)

fwrite(beta.GMQN.bmiq, 
       file = '/home/zhangyu/mnt_path/Data/stroke/GSE197080_beta_GMQN_BMIQ.csv', 
       row.names = T, sep = ',')


######################
# GSE69138 discovery
# df_meta_GSE69138 <- read.csv(
#     paste0(path_stroke, "GSE69138_process/GSE69138_meta_raw.txt"), row.names = 'X')
# list_sample_GSE69138 <- df_meta_GSE69138$sample_id
# 
df_signal_intensity <- 
    fread('/home/zhangyu/mnt_path/Data/stroke/GSE69138_signal_intensities.txt', 
          sep = '\t', header = TRUE)
row_cpgs <- df_signal_intensity$TargetID
col_samples <- colnames(df_signal_intensity)[5:dim(df_signal_intensity)[2]]
m_colnames <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[1]}))
m_colnames <- unique(m_colnames)

m_sel <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[2] == 'Signal_B'}))
m_sel <- col_samples[m_sel]
m_mat <- data.frame(df_signal_intensity[, ..m_sel], row.names = row_cpgs, check.names = F)
colnames(m_mat) <- m_colnames

um_sel <- unlist(lapply(strsplit(col_samples, '.', fixed = T), function(x){x[2] == 'Signal_A'}))
um_sel <- col_samples[um_sel]
um_mat <- data.frame(df_signal_intensity[, ..um_sel], row.names = row_cpgs, check.names = F)
colnames(um_mat) <- m_colnames

beta.GMQN.bmiq = gmqn_bmiq_parallel(m_mat, um_mat, ncpu = 20)

fwrite(beta.GMQN.bmiq, 
       file = '/home/zhangyu/mnt_path/Data/stroke/GSE69138_beta_GMQN_BMIQ_discovery.txt', 
       row.names = T, sep = ',')

# ENmix
anno<-c("IlluminaHumanMethylation450k", "ilmn12.hg19")
names(anno)<-c("array", "annotation")
mds_GSE69138<-MethylSet(Meth = as.matrix(m_mat), 
                        Unmeth = as.matrix(um_mat), annotation = anno)
mat_beta_ENmix <- mpreprocess(mds_GSE69138, nCores=30)
fwrite(mat_beta_ENmix, 
       file = '/home/zhangyu/mnt_path/Data/stroke/GSE69138_beta_ENmix_discovery.txt', 
       row.names = T, sep = ',')

one_cpg <- 'cg04192862'
one_beta <- mat_beta_ENmix[one_cpg,]
hist(one_beta, col = "skyblue")
     
#################################
annon_450k_1 <- annon_450k
colnames(annon_450k_1) <- c('probe_type_450k', 'color_450k')

annon_850k_1 <- annon_850k
colnames(annon_850k_1) <- c('probe_type_850k', 'color_850k')

df_anno <- merge(annon_450k_1, annon_850k_1, by = 'row.names')
ref = set_reference(m_mat, um_mat)
if (ref == 'default') {
    ref = list(t1.green.ref.mean=c(268.9645, 6884.7993),
               t1.green.ref.sd=c(203.8351, 3682.3554),
               t1.red.ref.mean=c(519.537, 8662.520),
               t1.red.ref.sd=c(380.9977, 5064.9526))
}

# 850k stroke rep
# $t1.green.ref.mean
# 1         2 
# 421.7841 6315.2334 
# 
# $t1.green.ref.sd
# [1]  174.0077 3317.2005
# 
# $t1.red.ref.mean
# 1        2 
# 1024.47 10986.21 
# 
# $t1.red.ref.sd
# [1]  570.983 5885.946

# 450k stroke rep
# $t1.green.ref.mean
# 1        2 
# 357.308 6941.831 
# 
# $t1.green.ref.sd
# [1]  186.2439 3411.8305
# 
# $t1.red.ref.mean
# 1         2 
# 233.8396 3557.0722 
# 
# $t1.red.ref.sd
# [1]  116.0657 2062.7383
