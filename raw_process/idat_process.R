# Load required libraries
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(minfi))
suppressPackageStartupMessages(library(ENmix))
suppressPackageStartupMessages(library(gmqn))

# Generate beta values using GMQN BMIQ normalization
generate_beta <- function(path_in, path_out, type, sel_cpgs = NULL, ncpu = 20) {
    RGSet = read.metharray.exp(path_in, force=TRUE)
    MSet <- preprocessRaw(RGSet)
    m = data.frame(getMeth(MSet))
    um = data.frame(getUnmeth(MSet))
    if (!is.null(sel_cpgs)) {
        m <- m[sel_cpgs,]
        um <- um[sel_cpgs,]
    }
    # Perform normalization in parallel
    suppressMessages(beta.GMQN.bmiq <- gmqn_bmiq_parallel(m, um, type = type, ncpu = ncpu, verbose = FALSE))
    fwrite(beta.GMQN.bmiq, file = path_out, row.names = T, sep = ',')
}

# Select overlapping CpG sites between 450k and 850k arrays
annon_450k_1 <- annon_450k
colnames(annon_450k_1) <- c('probe_type_450k', 'color_450k')
annon_850k_1 <- annon_850k
colnames(annon_850k_1) <- c('probe_type_850k', 'color_850k')
df_anno <- merge(annon_450k_1, annon_850k_1, by = 'row.names')
cpgs_overlap <- df_anno$Row.names

# GMQN processing based on input arguments
all_args <- commandArgs(trailingOnly = TRUE)
path_raw <- all_args[1]
path_out <- all_args[2]
generate_beta(path_raw, path_out, type = '450k', ncpu = 20)