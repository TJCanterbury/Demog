#!/usr/bin/env Rscript

library(tidyr)
library(dplyr)
library(ggplot2)
require(gridExtra)

# Import data
print(path)
readfile <- read.csv(paste0(path, "summaries.csv"))

# Benefit of strategic diversity
png(paste0(path, "ESS.png"), width = 720, height = 480)


df_long1 <- readfile %>%
  select(s, pi, m, xstar, rho_0,rho_1,eta) 

df_long1
plot(df_long1)

dev.off()