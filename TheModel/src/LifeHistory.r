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
# if (length(args) < 1 || is.na(args[1]) || args[1] == "") {
#   stop("No path supplied to R script")
# }
path <- args[1]

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
        z[ti,mi] <- sum(slice$"rhoi")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum = sum(z[ti,])
    for (mi in seq_along(m_vals)){
        
        z[ti,mi] <- z[ti,mi] / tsum

        print(z[ti,mi])
        
    }
  }

  pdf(paste0(path, "/lowi.pdf"), width = 10, height = 8)
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
    axis.args = list(cex.axis = 1.5)
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
        z[ti,mi] <- sum(slice$"rhoi")
    }
  }
  for (ti in seq_along(t_vals)){
    tsum = sum(z[ti,])
    for (mi in seq_along(m_vals)){
        
        z[ti,mi] <- z[ti,mi] / tsum

        print(z[ti,mi])
        
    }
  }

  pdf(paste0(path, "/highi.pdf"), width = 10, height = 8)
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
    axis.args = list(cex.axis = 1.5)
  )
  dev.off()
}

run_LH_plots(path)