library(VennDiagram)

M1_result <- read.csv('01.Cox_M1_all.csv')

M1_res_P <- M1_result[M1_result$HR_p_val < 0.05, ]

M1_res_bfi <- M1_result[M1_result$p_val_bfi < 0.05, ]

M2_result <- read.csv('02.Cox_M2_all.csv')

M2_res_P <- M2_result[M2_result$HR_p_val < 0.05, ]

M2_res_bfi <- M2_result[M2_result$p_val_bfi < 0.05, ]


set1 <- M1_res_P$Pro_code
set2 <- M2_res_P$Pro_code

pdf("02.vennDiagram_P.pdf", width =10, height = 10)

venn.plot <-venn.diagram(
  x = list(Model1 = set1, Model2 = set2),
  filename = NULL,
  fill = c("#66c2a5", "#8da0cb"),
  alpha = 0.5,
  cex = 2,
  cat.cex = 1.5,
  main = ""
)

grid.draw(venn.plot)

dev.off()


set1 <- M1_res_bfi$Pro_code
set2 <- M2_res_bfi$Pro_code

pdf("02.vennDiagram_bfi.pdf", width =10, height = 10)

venn.plot <-venn.diagram(
  x = list(Model1 = set1, Model2 = set2),
  filename = NULL, 
  fill = c("#66c2a5", "#8da0cb"),
  alpha = 0.5,
  cex = 2,
  cat.cex = 1.5,
  main = ""
)

grid.draw(venn.plot)

dev.off()
