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

run_LH_plots <- function(path) {

  readfile <- read.csv(paste0(path, "LifeHistory.csv"))

  # low population density
  df <- readfile %>%
    filter(i == 0)
  t_vals <- sort(unique(df$t))
  m_vals <- sort(unique(df$m))
  s_vals <- sort(unique(df$s))
  pi_vals <- sort(unique(df$pi))

  z <- matrix(
    NA_real_,
    nrow = length(t_vals),
    ncol = length(m_vals)
  )

  for (ti in seq_along(t_vals)){
    for (mi in seq_along(m_vals)){
      slice <- df %>%
        filter(t == t_vals[ti], m == m_vals[mi])
      z[ti, mi] <- sum(slice$"rhoi")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti,])
    for (mi in seq_along(m_vals)){
      z[ti, mi] <- z[ti, mi] / tsum
    }
  }

  pdf(paste0(path, "/LH_lowi.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, m_vals, z,
    col = hcl.colors(100, "Inferno"),
    xlab = bquote("Age in growth cycles "(t)),
    ylab = bquote("Expected reproductive quality "(m)),
    legend.lab = expression("Frequency of state m"),
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
    filter(i == 1.0) %>%
    arrange(t, m)
  t_vals <- sort(unique(df$t))
  m_vals <- sort(unique(df$m))

  z <- matrix(
    NA_real_,
    nrow = length(t_vals),
    ncol = length(m_vals)
  )

  for (ti in seq_along(t_vals)){
    for (mi in seq_along(m_vals)){
      slice <- df %>%
        filter(t == t_vals[ti], m == m_vals[mi])
      z[ti, mi] <- sum(slice$"rhoi")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti, ])
    for (mi in seq_along(m_vals)){
      z[ti, mi] <- z[ti, mi] / tsum
    }
  }

  pdf(paste0(path, "/LH_highi.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, m_vals, z,
    col = hcl.colors(100, "Inferno"),
    xlab = bquote("Age in growth cycles "(t)),
    ylab = bquote("Expected reproductive quality "(m)),
    legend.lab = expression("Frequency of state m"),
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
    filter(i == 0.0) %>%
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
      z[ti, si] <- sum(slice$"rhoi")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti, ])
    for (si in seq_along(s_vals)){
      z[ti, si] <- z[ti, si] / tsum
    }
  }

  pdf(paste0(path, "/size_lowi.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, m_vals, z,
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
    filter(i == 1.0) %>%
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
      z[ti, si] <- sum(slice$"rhoi")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum <- sum(z[ti, ])
    for (si in seq_along(s_vals)){
      z[ti, si] <- z[ti, si] / tsum
    }
  }

  pdf(paste0(path, "/size_highi.pdf"), width = 10, height = 8)
  par(mar = c(5, 5, 5, 5), oma = c(3, 6, 1, 6.65))

  # Heatmap
  image.plot(
    t_vals, m_vals, z,
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