library(dplyr)
library(ggplot2)

data <- read.csv('00.data.csv')

features <- read.csv('08.ProImportance.csv')

protein <- head(features, n = 10)$Pro_code

for (protein_name in protein) {
  data <- data %>%
    mutate(percentile = percent_rank(!!sym(protein_name)) * 100)

  data$percentile_group <- cut(
    data$percentile,
    breaks = seq(0, 100, by = 1),
    include.lowest = TRUE
  )

  incidence_df <- data %>%
    group_by(percentile_group) %>%
    summarise(
      mean_percentile = mean(percentile, na.rm = TRUE),
      incidence = mean(rci == 1, na.rm = TRUE)
    )

  p <- ggplot(incidence_df, aes(x = mean_percentile, y = incidence, color = mean_percentile)) +
    geom_point(alpha = 0.9, size = 2.5, shape = 16) +
    geom_smooth(
      method = "loess",
      se = TRUE,
      color = NA,
      fill = "#7c7c7c",
      alpha = 0.1
    ) +
    geom_smooth(
      method = "loess",
      se = FALSE,
      color = "#7C7C7C",
      alpha = 0.2,
      linewidth = 1
    ) +
    scale_color_gradient(low = "#ADD8E6", high = "#00008B") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    labs(
      x = "Expression Percentile",
      y = "Observed RCI Event Rate [%]",
      color = "Percentile",
      title = paste0("RCI Event Rate for ", protein_name)
    ) +
    theme_classic() +
    theme(
    axis.title = element_text(color = "black"),
    axis.text  = element_text(color = "black")
  )

  ggsave(
    filename = paste0("15.event_rate_for_percentile/incidence_vs_", protein_name, ".pdf"),
    plot = p,
    width = 6, height = 4, dpi = 300
  )
}
