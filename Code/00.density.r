library(survival)
library(survminer)
library(ggplot2)

data <- read.csv('00.data.csv')

dissection_data <- data[data$rci == 1,]

p <- ggplot(dissection_data, aes(x = rci_y_time)) +
  geom_histogram(aes(y = ..density..),
                 binwidth = 0.5,
                 fill = "#E74C3C",
                 color = "#E74C3C",
                 linewidth = 1,
                 alpha = 0.7) +
  scale_x_continuous(breaks = seq(0, 17, by = 1)) +
  labs(title = "",
       subtitle = "",
       x = "Follow-up Diagnosis time (years)",
       y = "Density") +
  theme_classic() +
  theme(plot.title = element_text(size = 14, face = "bold"))

ggsave('01.density_follow_up.pdf', p)

dissection_data$years <- dissection_data$Age_at_recruitment + dissection_data$rci_y_time

p <- ggplot(dissection_data, aes(x = years)) +
  geom_histogram(aes(y = ..density..),
                 binwidth = 0.5,
                 fill = "#E74C3C",
                 color = "#E74C3C",
                 linewidth = 1,
                 alpha = 0.7) +
  labs(title = "",
       subtitle = "",
       x = "First reportted arterial dissection time(years)",
       y = "Density") +
  theme_classic() +
  theme(plot.title = element_text(size = 14, face = "bold"))

ggsave('01.density_years.pdf', p)
