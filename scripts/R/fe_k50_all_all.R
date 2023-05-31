library("fixest")
library(arrow)
library("magrittr")
library(tidyverse)

fe_res_table <- function(res) {
  cints <- confint(res)
  results <- data.frame(res$coeftable) 
  results$lower <- cints[, 1]
  results$upper <- cints[, 2]
  return (results)
}

df <- read_parquet("//nas1/XL2$/speciale/regdata/full_new_k50_all_all.parquet") %>% 
  drop_na() %>% 
  mutate(
      year = cohort,
      cohort = as.factor(cohort),
      KOM = as.factor(KOM),
      INSTNR = as.factor(INSTNR)
  )

reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad_k_all_w-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/eu_grad_k_all_w-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad_k_all_w-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/eu_grad_k_all_w-INSTNR-all_ses_large_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad_k_all_w-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/gym_grad_k_all_w-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad_k_all_w-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/gym_grad_k_all_w-INSTNR-all_ses_large_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad_k_all_w-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/us_grad_k_all_w-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad_k_all_w-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/us_grad_k_all_w-INSTNR-all_ses_large_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet_k_all_w-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/real_neet_k_all_w-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet_k_all_w-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/real_neet_k_all_w-INSTNR-all_ses_large_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad_k_all_nw-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/eu_grad_k_all_nw-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad_k_all_nw-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/eu_grad_k_all_nw-INSTNR-all_ses_large_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad_k_all_nw-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/gym_grad_k_all_nw-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad_k_all_nw-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/gym_grad_k_all_nw-INSTNR-all_ses_large_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad_k_all_nw-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/us_grad_k_all_nw-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad_k_all_nw-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/us_grad_k_all_nw-INSTNR-all_ses_large_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet_k_all_nw-INSTNR-all_ses_small_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/real_neet_k_all_nw-INSTNR-all_ses_small_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet_k_all_nw-INSTNR-all_ses_large_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all/real_neet_k_all_nw-INSTNR-all_ses_large_all_all.csv")


