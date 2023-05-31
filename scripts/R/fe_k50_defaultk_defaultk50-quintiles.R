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

df <- read_parquet("//nas1/XL2$/speciale/regdata/full_new_k50_defaultk_defaultk50.parquet") %>% 
  drop_na() %>% 
  mutate(
      year = cohort,
      cohort = as.factor(cohort),
      KOM = as.factor(KOM),
      INSTNR = as.factor(INSTNR)
  )

reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-w-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-reduced-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_grad-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/eu_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/gym_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/us_apply-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_large-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_q99-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_q95-_defaultk_defaultk50-all-quintiles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_defaultk_defaultk50-quintiles/real_neet-k_defaultk-nw-INSTNR-all_ses_q90-_defaultk_defaultk50-all-quintiles.csv")


