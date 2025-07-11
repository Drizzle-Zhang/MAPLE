options(
  repos = c(
    CRAN = "https://mirrors.pku.edu.cn/CRAN/"
  )
)
options(BioC_mirror = "https://mirrors.westlake.edu.cn/bioconductor")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes")

BiocManager::install(version = "3.18", ask = FALSE)

BiocManager::install(
  c("minfi", "ENmix"),
  version      = "3.18",
  ask          = FALSE,
  dependencies = TRUE,
  Ncpus        = parallel::detectCores()
)

remotes::install_version(
  "data.table",
  version         = "1.14.6",
  build_vignettes = FALSE,
  quiet           = TRUE
)

remotes::install_github("MengweiLi-project/gmqn")
