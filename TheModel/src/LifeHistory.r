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

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]

run_LH_plots <- function(path) {

  readfile <- read.csv(paste0(path, "LifeHistory.csv"))

  # low population density
  df <- readfile %>%
    filter(n == 0)
  t_vals <- sort(unique(df$t))
  e_vals <- sort(unique(df$e))
  s_vals <- sort(unique(df$s))
  i_vals <- sort(unique(df$i))

  z <- matrix(
    NA_real_,
    nrow = length(t_vals),
    ncol = length(e_vals)
  )

  for (ti in seq_along(t_vals)){
    for (ei in seq_along(e_vals)){
      slice <- df %>%
        filter(t == t_vals[ti], e == e_vals[ei])
      z[ti, ei] <- sum(slice$"rho")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti,])
    for (ei in seq_along(e_vals)){
      z[ti, ei] <- z[ti, ei] / tsum
    }
  }

  pdf(paste0(path, "/LH_lown.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, e_vals, z,
    col = hcl.colors(100, "Inferno"),
    xlab = bquote("Age in growth cycles "(t)),
    ylab = bquote("Energy reserves "(e)),
    legend.lab = expression("Frequency of state e"),
    cex.axis = 1.5,
    cex.lab = 1.5,
    legend.cex = 1.5,
    legend.line = 4,
    axis.args = list(cex.axis = 1.5),
    main = "Life history projection in low population densities"
  )
  dev.off()

  # high population density
  df <- readfile %>%
    filter(n == 1.0) %>%
    arrange(t, e)
  t_vals <- sort(unique(df$t))
  e_vals <- sort(unique(df$e))

  z <- matrix(
    NA_real_,
    nrow = length(t_vals),
    ncol = length(e_vals)
  )

  for (ti in seq_along(t_vals)){
    for (mi in seq_along(e_vals)){
      slice <- df %>%
        filter(t == t_vals[ti], e == e_vals[mi])
      z[ti, mi] <- sum(slice$"rho")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti, ])
    for (ei in seq_along(e_vals)){
      z[ti, ei] <- z[ti, ei] / tsum
    }
  }

  pdf(paste0(path, "/LH_highn.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, e_vals, z,
    col = hcl.colors(100, "Inferno"),
    xlab = bquote("Age in growth cycles "(t)),
    ylab = bquote("Energy reserves "(e)),
    legend.lab = expression("Frequency of state e"),
    cex.axis = 1.5,
    cex.lab = 1.5,
    legend.cex = 1.5,
    legend.line = 4,
    axis.args = list(cex.axis = 1.5),
    main = "Life history projection in high population densities"
  )
  dev.off()
  
  # Low population density and life stages
  df <- readfile %>%
    filter(n == 0.0) %>%
    arrange(t, s)
  t_vals <- sort(unique(df$t))
  s_vals <- sort(unique(df$s))

  z <- matrix(
    NA_real_,
    nrow = length(t_vals),
    ncol = length(s_vals)
  )

  for (ti in seq_along(t_vals)){
    for (si in seq_along(s_vals)){
      slice <- df %>%
        filter(t == t_vals[ti], s == s_vals[si])
      z[ti, si] <- sum(slice$"rho")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti, ])
    for (si in seq_along(s_vals)){
      z[ti, si] <- z[ti, si] / tsum
    }
  }

  pdf(paste0(path, "/size_lown.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, s_vals, z,
    col = hcl.colors(100, "Inferno"),
    xlab = bquote("Age in growth cycles "(t)),
    ylab = bquote("Life stage "(s)),
    legend.lab = expression("Frequency of state s"),
    cex.axis = 1.5,
    cex.lab = 1.5,
    legend.cex = 1.5,
    legend.line = 4,
    axis.args = list(cex.axis = 1.5),
    main = "Life history projection in low population densities"
  )
  dev.off()

  # high population density and life stages
  df <- readfile %>%
    filter(n == 1.0) %>%
    arrange(t, s)
  t_vals <- sort(unique(df$t))
  s_vals <- sort(unique(df$s))

  z <- matrix(
    NA_real_,
    nrow = length(t_vals),
    ncol = length(s_vals)
  )

  for (ti in seq_along(t_vals)){
    for (si in seq_along(s_vals)){
      slice <- df %>%
        filter(t == t_vals[ti], s == s_vals[si])
      z[ti, si] <- sum(slice$"rho")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti, ])
    for (si in seq_along(s_vals)){
      z[ti, si] <- z[ti, si] / tsum
    }
  }

  pdf(paste0(path, "/size_highn.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, s_vals, z,
    col = hcl.colors(100, "Inferno"),
    xlab = bquote("Age in growth cycles "(t)),
    ylab = bquote("Life stage "(s)),
    legend.lab = expression("Frequency of state s"),
    cex.axis = 1.5,
    cex.lab = 1.5,
    legend.cex = 1.5,
    legend.line = 4,
    axis.args = list(cex.axis = 1.5),
    main = "Life history projection in high population densities"
  )
  dev.off()
}

run_LH_plots(path)