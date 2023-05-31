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

df <- read_parquet("//nas1/XL2$/speciale/regdata/full_new_radius200_defaultradius_defaultradius200.parquet") %>% 
  drop_na() %>% 
  mutate(
      year = cohort,
      cohort = as.factor(cohort),
      KOM = as.factor(KOM),
      INSTNR = as.factor(INSTNR)
  )

reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-w-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-reduced-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_grad-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/eu_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/gym_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/us_apply-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_large-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q99-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q95-_defaultradius_defaultradius200-all-deciles.csv")


reg <- feols(
  fml = as.formula(paste(readLines("//nas1/XL2$/speciale/dst-con/nsdata/formulas/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.txt"), collapse = " ")),
  data =  df
) 
res <- summary(reg, vcov = ~INSTNR)
print(res)
write.csv(fe_res_table(res), file = "//nas1/XL2$/speciale/tables/fe_radius200_defaultradius_defaultradius200-deciles/real_neet-radius_defaultradius-nw-INSTNR-all_ses_q90-_defaultradius_defaultradius200-all-deciles.csv")


