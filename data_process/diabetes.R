library(data.table)
library(minfi)
library(ENmix)
library(gmqn)


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

# select overlap cpgs
annon_450k_1 <- annon_450k
colnames(annon_450k_1) <- c('probe_type_450k', 'color_450k')
annon_850k_1 <- annon_850k
colnames(annon_850k_1) <- c('probe_type_850k', 'color_850k')
df_anno <- merge(annon_450k_1, annon_850k_1, by = 'row.names')
cpgs_overlap <- df_anno$Row.names


############################
# GSE199700
path_prediabetes <- '/home/zhangyu/mnt_path/Data/prediabetes/'
path_GSE199700 <- paste0(path_prediabetes, "GSE199700_RAW/")
path_out_GSE199700 <- paste0(path_prediabetes, 'GSE199700_850k_450k_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE199700, path_out_GSE199700, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)

#############################
# T1D
# GSE76169
path_t1d <- '/home/zhangyu/mnt_path/Data/T1D/'
path_GSE76169 <- paste0(path_t1d, "GSE76169_RAW/")
path_out_GSE76169 <- paste0(path_t1d, 'GSE76169_850k_450k_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE76169, path_out_GSE76169, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)

# GSE76170
path_t1d <- '/home/zhangyu/mnt_path/Data/T1D/'
path_GSE76170 <- paste0(path_t1d, "GSE76170_RAW/")
path_out_GSE76170 <- paste0(path_t1d, 'GSE76170_850k_450k_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE76170, path_out_GSE76170, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)

#############################
# T2D
# GSE197881
path_t2d <- '/home/zhangyu/mnt_path/Data/T2D/'
path_GSE197881 <- paste0(path_t2d, "GSE197881_RAW/")
path_out_GSE197881 <- paste0(path_t2d, 'GSE197881_850k_450k_beta_GMQN_BMIQ.csv')
generate_beta(path_GSE197881, path_out_GSE197881, type = '450k', sel_cpgs = cpgs_overlap, ncpu = 20)

