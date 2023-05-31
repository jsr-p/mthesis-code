library("fixest")
library(arrow)
library("magrittr")
library(tidyverse)

fe_res_table <- function(res) {
  cints <- confint(res)
  results <- data.frame(res$coeftable) 
  results$lower <- cints[, 1]
  results$upper <- cints[, 2]
  results$N <- res$nobs
  return (results)
}

df <- read_parquet("//nas1/XL2$/speciale/regdata/full_new_k30_fullk_fullk30.parquet") %>% 
  drop_na() %>% 
  mutate(
      year = cohort,
      cohort = as.factor(cohort),
      KOM = as.factor(KOM),
      INSTNR = as.factor(INSTNR)
  )

reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-w-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-reduced.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_grad-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/eu_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/gym_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/us_apply-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_large-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_q99-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_q95-_fullk_fullk30-all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k30_fullk_fullk30-deciles/real_neet-k_fullk-nw-INSTNR-all_ses_q90-_fullk_fullk30-all.csv")


