#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyr)
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  library(patchwork)
  library(akima)
  library(fields)
  library(ggpubr)
})

# args <- commandArgs(trailingOnly = TRUE)
# path <- args[1]

run_policy_plots <- function(path) {

  readfile <- read.csv(paste0(path, "Policies.csv"))

  # Perceived low population density
  df <- readfile %>%
    filter(pi == 0.0) %>%
    arrange(s, m)
  s_vals <- sort(unique(df$s))
  m_vals <- sort(unique(df$m))

  z <- matrix(
    NA_real_,
    nrow = length(s_vals),
    ncol = length(m_vals)
  )
  z[cbind(
    match(df$s, s_vals),
    match(df$m, m_vals)
  )] <- df$xstar / 4

  print(z)
  pdf(paste0(path, "/Policy_lowdensity", ".pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    s_vals, m_vals, z,
    col = hcl.colors(100, "Inferno"),
    zlim = c(0,1),
    xlab = bquote("Life stage "(s)),
    ylab = bquote("Expected reproductive quality "(m)),
    legend.lab = expression("Investment threshold"),
    cex.axis = 1.5,
    cex.lab = 1.5,
    legend.cex = 1.5,
    legend.line = 4,
    asp = 1,
    axis.args = list(cex.axis = 1.5),
    main = "ESS developmental policy when individuals believe densities are low"
  )
  dev.off()

  # Perceived high population density
  df <- readfile %>%
    filter(pi == 1.0)

  s_vals <- sort(unique(df$s))
  m_vals <- sort(unique(df$m))

  z <- matrix(
    NA_real_,
    nrow = length(s_vals),
    ncol = length(m_vals)
  )

  z[cbind(
    match(df$s, s_vals),
    match(df$m, m_vals)
  )] <- df$xstar/4

  pdf(paste0(path, "/Policy_highdensity", ".pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    s_vals, m_vals, z,
    col  = hcl.colors(100, "Inferno"),
    zlim = c(0,1),
    xlab = bquote("Life stage "(s)),
    ylab = bquote("Expected reproductive quality "(m)),
    legend.lab = expression("Investment threshold"),
    cex.axis = 1.5,
    cex.lab  = 1.5,
    legend.cex = 1.5,
    legend.line = 4,
    asp = 1,
    axis.args = list(cex.axis = 1.5),
    main = "ESS developmental policy when individuals believe densities are high"
  )
  dev.off()

  # Perceived uncertain population density
  df <- readfile %>%
    filter(pi == 0.5)
  s_vals <- sort(unique(df$s))
  m_vals <- sort(unique(df$m))

  z <- matrix(
    NA_real_,
    nrow = length(s_vals),
    ncol = length(m_vals)
  )

  z[cbind(
    match(df$s, s_vals),
    match(df$m, m_vals)
  )] <- df$xstar / 4


  pdf(paste0(path, "/Policy_uncertaindensity", ".pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    s_vals, m_vals, z,
    col  = hcl.colors(100, "Inferno"),
    zlim = c(0,1),
    xlab = bquote("Life stage "(s)),
    ylab = bquote("Expected reproductive quality "(m)),
    legend.lab = expression("Investment threshold"),
    cex.axis = 1.5,
    cex.lab  = 1.5,
    legend.cex = 1.5,
    legend.line = 4,
    asp = 1,
    axis.args = list(cex.axis = 1.5),
    main = "ESS developmental policy when individuals are uncertain"
  )
  dev.off()

  # file.copy(
  #   paste0(path, "../", "baselineH", ".pdf"),
  #   paste0(path, "../../", "baselineL", ".pdf"),
  #   overwrite = TRUE
  # )
}

run_policy_plots(path)