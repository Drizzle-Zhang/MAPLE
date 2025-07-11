# 1. 检查 BiocManager 客户端
has_biocman <- requireNamespace("BiocManager", quietly = TRUE)
message("BiocManager installed? ", has_biocman)

# 2. 打印 Bioconductor Release 版本（如果 BiocManager 已安装）
if (has_biocman) {
  # BiocManager::version() 返回一个列表，用 print() 处理
  message("Bioconductor release: ")
  print(BiocManager::version())
}

# 3. 检查其他包是否安装
packages <- c("data.table", "minfi", "ENmix", "gmqn")
installed <- sapply(packages, requireNamespace, quietly = TRUE)

# 4. 获取已安装包的版本
get_pkg_version <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(as.character(packageVersion(pkg)))
  } else {
    return(NA)
  }
}

versions <- sapply(packages, get_pkg_version)

# 输出结果表格
df <- data.frame(
  Package   = packages,
  Installed = installed,
  Version   = versions,
  row.names = NULL,
  stringsAsFactors = FALSE
)
print(df)