library(stringr)
library(dplyr)
library(AnnotationDbi)
library(org.Hs.eg.db)
library(clusterProfiler)
library(DOSE)
library(enrichplot)
library(psych)
library(ggplot2)

gene <- read.csv('03.Cox_M2_all.csv')

genes <- gene[gene$p_val_fdr < 0.05, ]$Pro_code

module_name <- 'RCI'

#### GOBP
gmt_bp <- read.gmt("c5.go.bp.v2024.1.Hs.symbols.gmt")

ego_bp <- enricher(
  gene = genes,
  TERM2GENE = gmt_bp,
  pAdjustMethod = "fdr",
  pvalueCutoff = 0.05
)

csv_file <- paste0("07.GO_enrich_results_BP_", module_name, ".csv")
GO_df <- as.data.frame(ego_bp)
write.csv(GO_df, file = csv_file, row.names = FALSE)

####GOMF
gmt_mf <- read.gmt("c5.go.mf.v2024.1.Hs.symbols.gmt")

ego_mf <- enricher(
  gene = genes,
  TERM2GENE = gmt_mf,
  pAdjustMethod = "fdr",
  pvalueCutoff = 0.05
)

csv_file <- paste0("07.GO_enrich_results_MF_", module_name, ".csv")
GO_df <- as.data.frame(ego_mf)
write.csv(GO_df, file = csv_file, row.names = FALSE)

#### GOCC
gmt_cc <- read.gmt("c5.go.cc.v2024.1.Hs.symbols.gmt")

ego_cc <- enricher(
  gene = genes,
  TERM2GENE = gmt_cc,
  pAdjustMethod = "fdr",
  pvalueCutoff = 0.05
)

csv_file <- paste0("07.GO_enrich_results_CC_", module_name, ".csv")
GO_df <- as.data.frame(ego_cc)
write.csv(GO_df, file = csv_file, row.names = FALSE)

#### KEGG
gmt_kegg <- read.gmt("c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt")

ego_kegg <- enricher(
  gene = genes,
  TERM2GENE = gmt_kegg,
  pAdjustMethod = "fdr",
  pvalueCutoff = 0.05
)

csv_file <- paste0("07.GO_enrich_results_KEGG_", module_name, ".csv")
GO_df <- as.data.frame(ego_kegg)
write.csv(GO_df, file = csv_file, row.names = FALSE)

#### reactome
gmt_reac <- read.gmt("/c2.cp.reactome.v2024.1.Hs.symbols.gmt")

ego_reac <- enricher(
  gene = genes,
  TERM2GENE = gmt_reac,
  pAdjustMethod = "fdr",
  pvalueCutoff = 0.05
)

csv_file <- paste0("07.GO_enrich_results_REAC_", module_name, ".csv")
GO_df <- as.data.frame(ego_reac)
write.csv(GO_df, file = csv_file, row.names = FALSE)

#### WP
gmt_wp <- read.gmt("c2.cp.wikipathways.v2024.1.Hs.symbols.gmt")

ego_wp <- enricher(
  gene = genes,
  TERM2GENE = gmt_wp,
  pAdjustMethod = "fdr",
  pvalueCutoff = 0.05
)

csv_file <- paste0("07.GO_enrich_results_WP_", module_name, ".csv")
GO_df <- as.data.frame(ego_wp)
write.csv(GO_df, file = csv_file, row.names = FALSE)
