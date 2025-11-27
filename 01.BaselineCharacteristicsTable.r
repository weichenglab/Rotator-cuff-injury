library(tableone)

rawdf <- read.csv('00.data.csv', header = 1)

model <- c('Age', 'Sex', 'Edu', 'Ethnic', 'TD', 'BMI', 'Smoke', 'Drink', 'Manual_work_cut', 'Diabetes', 'Hypertension', 'Obesity', 'Dyslipidemia')

contVars <- c('Age')

factorVar <- c('Sex', 'Edu', 'Ethnic', 'TD', 'BMI', 'Smoke', 'Drink', 'Manual_work_cut', 'Diabetes', 'Hypertension', 'Obesity', 'Dyslipidemia')

group_var <- 'rci'

tab1 <- CreateTableOne(vars = model, strata = group_var, data = rawdf, factorVars = factorVar, addOverall = TRUE, test = TRUE)

print(tab1, showAllLevels = TRUE, quote = TRUE, nospaces = TRUE, overall = TRUE)

tab1_df <- print(tab1, showAllLevels = TRUE, quote = FALSE, nospaces = TRUE, overall = TRUE, printToggle = FALSE)

library(openxlsx)

write.xlsx(tab1_df, file = "02.BaselineTable.xlsx", rowNames = TRUE)
