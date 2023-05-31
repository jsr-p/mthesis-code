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
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_all-w-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/eu_grad-k_all-w-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_all-w-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/eu_grad-k_all-w-INSTNR-all_ses_large-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_all-w-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/gym_grad-k_all-w-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_all-w-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/gym_grad-k_all-w-INSTNR-all_ses_large-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_all-w-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/us_grad-k_all-w-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_all-w-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/us_grad-k_all-w-INSTNR-all_ses_large-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_all-w-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/real_neet-k_all-w-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_all-w-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/real_neet-k_all-w-INSTNR-all_ses_large-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_all-nw-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/eu_grad-k_all-nw-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-k_all-nw-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/eu_grad-k_all-nw-INSTNR-all_ses_large-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_all-nw-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/gym_grad-k_all-nw-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-k_all-nw-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/gym_grad-k_all-nw-INSTNR-all_ses_large-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_all-nw-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/us_grad-k_all-nw-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-k_all-nw-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/us_grad-k_all-nw-INSTNR-all_ses_large-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_all-nw-INSTNR-all_ses_small-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/real_neet-k_all-nw-INSTNR-all_ses_small-_all_all.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-k_all-nw-INSTNR-all_ses_large-_all_all.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_k50_all_all-deciles/real_neet-k_all-nw-INSTNR-all_ses_large-_all_all.csv")


