library(ggplot2)
library(dplyr)
library(readr)
library(ggrepel)
library(stats)

plot_volcano <- function(df, outfile){

  colnames(df) <- c("protein", "HR", "HR_Lower_CI", "HR_Upper_CI", "p_value", "p_val_fdr", "p_val_bfi")

  df <- df %>%
    mutate(`-log10(p_value)` = -log10(p_value))

  top10_proteins <- df %>% filter(p_val_bfi < 0.05) %>%  arrange(p_val_bfi) %>%  slice_head(n = 10) %>% pull(protein)

  df <- df %>% mutate(sig = case_when(p_value < 0.01 & HR > 1 ~ "Up", p_value < 0.01 & HR < 1 ~ "Down", TRUE ~ "NS"))

  p <- ggplot(df, aes(x = HR, y = `-log10(p_value)`, color = sig)) +
    geom_point(size=1.5, alpha=1) +
    scale_color_manual(values = c("NS"="#a7aaac", "Up"="#db5143", "Down"="#3b6fb6")) +
    geom_hline(yintercept = -log10(0.05), linetype="dashed", color="black") +
    geom_vline(xintercept = 1, linetype="dashed", color="black") +
    geom_text_repel(data = subset(df, protein %in% top10_proteins),
                    aes(label = protein),
                    size = 2.5, color="black", max.overlaps = Inf) +
    theme_classic() +
    theme(legend.title = element_blank(),
          legend.position = "right",
          axis.text = element_text(size=10),
          axis.title = element_text(size=11)) +
    labs(x="Hazard Ratio (HR)", y="-log10(adjusted P-value)")
  
  ggsave(outfile, p, width=6, height=4, dpi=300)
}

model1 <- read_csv("02.Cox_M1_all.csv")
plot_volcano(model1, "04.model1_volcano_plot_HR_color.pdf")

model2 <- read_csv("03.Cox_M2_all.csv")
plot_volcano(model2, "04.model2_volcano_plot_HR_color.pdf")
