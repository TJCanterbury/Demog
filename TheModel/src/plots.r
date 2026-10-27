#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
suppressPackageStartupMessages({
library(tidyr)
library(dplyr)
library(ggplot2)
library(plotly) 
})
loci <- list(
  psi    = list(col = "#a30000", label = bquote("Observation threshold "(psi))),
  lambda = list(col = "#ffffff", label = bquote("Time perception "(lambda*tau))),
  ve     = list(col = "#b9f7ff", label = bquote("Climate risk aversion "(nu))),
  vs     = list(col = "#a6ff00", label = bquote("Social sensitivity "(tau)))#,
  # prior     = list(col = "#003cff", label = bquote("Prior climate information "(eta[0])))
)

plotter <- function(path, x, d, x_label, output){
  d$lambda <- d$lambda * d$vs
  # Plotting parameters
  linewd = 2
  smoothness <- 2/3

  png(paste0(path, "../", output, ".png"), 10, 10,
  units  = "in",
  res    = 300)
  
  par(
    mfrow = c(2,1),
    mar = c(4, 4, 0, 15),
    oma = c(1, 1, 2, 1)  
  )
  
  plot(x, rep(NA, length(x)),
      type = "n", ylim = c(0,1),
      xlab = "", ylab = "Loci values")
  usr <- par("usr")
  rect(usr[1], usr[3], usr[2], usr[4],
      col = "#c0c0c0", border = NA)

  grid(col = "#f8f8f8")
  for (nm in names(loci)) {
    y <- d[[nm]]
    points(x, y, col = paste0(loci[[nm]]$col, "c9"))
    lines(lowess(x, y, f = smoothness),
          col = loci[[nm]]$col, lwd = linewd)
  }
  par(xpd = TRUE) 
  legend("right",
    inset = c(-0.5, 0),
    legend = lapply(loci, `[[`, "label"),
    col    = sapply(loci, `[[`, "col"),
    lwd    = linewd,
    bg     = "#c0c0c0")
  par(xpd = FALSE) 
  plot(x, d$stability,
    type = "p", col = "white", pch = 16, cex = 0.6,
    ylim = c(0,1),
    xlab = x_label,
    ylab = "Phenotype values"
  )
  grid(col = "#b0b0b0")
  variables <- list(d$stability, d$climate_match, d$nest_mass, d$mean_obs)
  colours <- c("#1f5300", "#000000", "#ff963f", "#9d00ff")

  for (i in 1:length(variables)) {
    points(x, variables[[i]], col = paste0(colours[i], "80"))
    model <- lowess(x, variables[[i]], f=smoothness)
    lines(model, col = colours[i], lwd = linewd)
  }
    

  par(xpd = TRUE) 
  legend("right",
    legend = c(
      bquote("Nest robustness " (rho)),
      bquote("Nest-climate match " (x)), 
      bquote("Nest mass " (m)), 
      "Observation rate"),
    inset = c(-0.5, 0),
    col = c("#1f5300", "#000000", "#ff963f", "#9d00ff"),
    lwd = linewd,
    bg = rgb(1, 1, 1, 0.7),
    text.width = strwidth(bquote("Observation threshold "(psi))),
    xpd = TRUE)

  dev.off()

  file.copy(paste0(path, "../", output, ".png"), 
    paste0(path, "../../", output, ".png"), overwrite = TRUE)
}

run_cue_noise_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  x <- d$cue_error

  plotter(path = path,
    x = x,
    d = d,
    x_label = bquote("Social cue noise"),
    output = "cue_noise")
}

run_theta_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  theta <- (d$theta_high - 0.5) / 0.5

  plotter(path = path,
    x = theta,
    d = d,
    x_label = bquote("Climate harshness "(theta)),
    output = "theta")
}

run_cost_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  x <- d$obs_c
  print("111\n")
  plotter(path = path,
    x = x,
    d = d,
    x_label = bquote("Cost of observations " (c[psi])),
    output = "cost")
}

run_G_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  x <- d$days
  
  plotter(path = path,
    x = x,
    d = d,
    x_label = bquote("Climate harshness " (G)),
    output = "G")
}

run_cli_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  x <- d$cli_sigma
  
  plotter(path = path,
    x = x,
    d = d,
    x_label = bquote("Climate cue noise " (sigma)),
    output = "cli_norho")
}

run_stoch_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  x <- d$stoch
  
  plotter(path = path,
    x = x,
    d = d,
    x_label = bquote("Climate autocorrelation"),
    output = "stoch")
}

run_p_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  x <- 1 - d$p

  plotter(path = path,
    x = x,
    d = d,
    x_label = bquote("Decision error rate "(1-p)),
    output = "p_simp")
}

run_mu_sig_plot <- function(path) {
  d <- read.csv(paste0(path, "summaries.csv"))
  x <- d$mu_sig
  
  plotter(path = path,
    x = x,
    d = d,
    x_label = bquote("Mutation size "(sigma[mu])),
    output = "mu_sig")
}