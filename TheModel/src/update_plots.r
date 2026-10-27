#!/usr/bin/env Rscript
cat("Script started\n")
args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args_all[grep("^--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

source(file.path(script_dir, "plots.r"))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript script.R <input>")
}
input <- args[1]
cat("Input:", input, "\n")

run_p_plot(paste0(input,"p/"))
run_cli_plot(paste0(input,"cli/"))
run_cost_plot(paste0(input,"cost/"))
run_cue_noise_plot(paste0(input,"cue_noise/"))
run_G_plot(paste0(input,"G/"))
run_theta_plot(paste0(input,"theta/"))
# run_mu_sig_plot(paste0(input,"mu_sig/"))