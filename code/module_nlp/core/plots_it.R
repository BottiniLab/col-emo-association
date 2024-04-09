#! /usr/bin/Rscript

library("ggplot2")
# library("ggpubr")
library("ggsignif")
library("ggrepel")
library("scales")

# Suppress existing file warnings
options(warn = -1)

# Create custom theme common to all plots
scatter_theme <- theme(
  # Panel background color
  panel.background = element_rect(fill = "white"),
  # Axis label color
  axis.text = element_text(color = "#151515", size = 18),
  # Legend settings
  legend.background = element_blank(),
  legend.key = element_blank(),
  legend.text = element_text(color = "#151515"),
  # legend.title = element_blank(),
  # Tick settings
  axis.ticks = element_line(color = "#151515"),
  axis.ticks.length = unit(0.15, "cm"), # Major tick size
  # Axis line
  axis.line.x = element_line(color = "black", linewidth = 1.2),
  axis.line.y = element_line(color = "black", linewidth = 1.2),
  # Font family
  text = element_text(family = "sans-serif", size = 20, colour = "#151515"),
)

barplot_theme <- theme(
  # Panel background color
  panel.background = element_rect(fill = "white"),
  # Axis label color
  axis.text.y = element_text(color = "#151515", size = 18),
  axis.text.x = element_text(
    angle = 30, color = "#151515", size = 15, vjust = 1,
    hjust = 1
  ),
  # Legend settings
  legend.background = element_blank(),
  legend.key = element_blank(),
  legend.text = element_text(color = "#151515", size = 15),
  legend.title = element_text(size = 15),
  # legend.title = element_blank(),
  # Tick settings
  axis.ticks = element_line(color = "#151515"),
  axis.ticks.length = unit(0.15, "cm"), # Major tick size
  # Axis line
  axis.line.x = element_line(color = "black", linewidth = 1.2),
  axis.line.y = element_line(color = "black", linewidth = 1.2),
  # Font family
  text = element_text(family = "sans-serif", size = 20, colour = "#151515"),
  plot.title = element_text(hjust = 0.5, family = "sans-serif", size = 20)
)

# Plot figure 7A

plot_colors <- function(file_name, save_colors,
                        xlab = "Valence", ylab = "Arousal", style = scatter_theme) {
  data <- read.csv(paste0("../data/nlp/coordinates/", file_name), header = T, sep = ",", dec = ".", fill = T)
  data$index <- toupper(data$index)
  colors <- data[data$Condition == "Color", ]
  fill_color <- c(
    rgb(0, 255, 0, maxColorValue = 255), rgb(255, 128, 0, maxColorValue = 255),
    rgb(255, 255, 0, maxColorValue = 255), rgb(0, 0, 255, maxColorValue = 255),
    rgb(128, 0, 255, maxColorValue = 255), rgb(255, 0, 0, maxColorValue = 255)
  )
  cols <- as.character(fill_color)
  names(cols) <- as.character(colors$index)

  scatter_color <- ggplot(colors, mapping = aes(x = z_valence, y = z_arousal)) +
    geom_point(aes(size = 6, colour = cols), show.legend = F) +
    scale_color_identity() +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    style
  save_to <- paste0("../figures/", save_colors)
  dir.create(file.path(dirname(save_to)))
  ggsave(save_to, plot = scatter_color, width = 6, height = 6)
}

plot_emotions <- function(file_name, save_colors, save_emotions,
                          xlab = "Valence", ylab = "Arousal", style = scatter_theme) {
  data <- read.csv(paste0("../data/nlp/coordinates/", file_name), header = T, sep = ",", dec = ".", fill = T)
  emotions <- data[data$Condition == "Emotion", ]
  labels <- c(
    "SURPRISED", "EXCITED", "SERENE", "HAPPY", "SATISFIED",
    "CALM", "TIRED", "BORED", "DEPRESSED", "SAD", "FRUSTRATED", "AFRAID", "ANGRY",
    "STRESSED", "ASTONISHED", "SLEEPY", "ALARMED", "DISGUSTED"
  )


  scatter_emotions <- ggplot(emotions, mapping = aes(x = z_valence, y = z_arousal, label = labels)) +
    geom_point(aes(size = 6, colour = "#89b4fa"), show.legend = F) +
    scale_color_identity() +
    scale_x_continuous(name = xlab) +
    scale_y_continuous(name = ylab) +
    geom_text_repel(size = 5) +
    style
  save_to <- paste0("../figures/", save_emotions)
  dir.create(file.path(dirname(save_to)))
  ggsave(save_to, plot = scatter_emotions, width = 6, height = 6)
}

# Plot figure 7E
plot_both <- function(file_name, is_geo = F, save_both,
                      xlab = "Valence", ylab = "Arousal", style = scatter_theme) {
  data <- read.csv(paste0("../data/nlp/coordinates/", file_name), header = T, sep = ",", dec = ".", fill = T)
  labels <- c(
    "GREEN", "ORANGE", "YELLOW", "BLUE", "PURPLE", "RED", "SURPRISED", "EXCITED", "SERENE", "HAPPY", "SATISFIED",
    "CALM", "TIRED", "BORED", "DEPRESSED", "SAD", "FRUSTRATED", "AFRAID", "ANGRY",
    "STRESSED", "ASTONISHED", "SLEEPY", "ALARMED", "DISGUSTED"
  )
  if (!is_geo) {
    cols <- as.character(data$fill_color)
    names(cols) <- as.character(data$index)
    both <- ggplot(data, mapping = aes(x = z_valence, y = z_arousal, label = labels)) +
      geom_point(aes(size = 6, colour = fill_color), show.legend = F) +
      scale_color_identity() +
      geom_text(size = 0) +
      scale_x_continuous(name = xlab) +
      scale_y_continuous(name = ylab) +
      geom_text_repel(size = 5) +
      style

    save_to <- paste0("../figures/", save_both)
    dir.create(file.path(dirname(save_to)))
    ggsave(save_to, plot = both, width = 6, height = 6)
  }
}

rdm_corr <- function(csv_name, title, file_name, y_position = 0.55, ylim = 0.65,
                     plabels = c("", "", ""), style = barplot_theme) {
  data <- read.csv(paste0("../data/nlp/correlations/", csv_name[1]))
  data2 <- read.csv(paste0("../data/nlp/correlations/", csv_name[2]))
  if (data2$p_val[1] < 0.001) {
    pval_ft <- "p < 0.001"
  } else {
    pval_ft <- paste0("p = ", round(data2$p_val[1], 3))
  }

  if (data2$p_val[2] < 0.001) {
    pval_ft2d <- "p < 0.001"
  } else {
    pval_ft2d <- paste0("p = ", round(data2$p_val[2], 3))
  }

  if (data2$p_val[3] < 0.001) {
    pval_control <- "p < 0.001"
  } else {
    pval_control <- paste0("p = ", round(data2$p_val[3], 3))
  }
  # Generate plots
  corrs <- ggplot(data, aes(x = factor(X), y = coefficient, fill = as.factor(round(coefficient, 3)))) +
    scale_x_discrete(limits = factor(data$X)) +
    ylim(ylim[1], ylim[2]) +
    geom_bar(stat = "identity") +
    scale_fill_manual(
      values = c("#89dceb", "#74c7ec", "#89b4fa", "#b4befe"),
      name = "rho-coefficients", limits = factor(round(data$coefficient, 3))
    ) +
    geom_text(aes(y = coefficient + 0.02 * sign(coefficient), label = plabels),
      size = 6
    ) +
    geom_signif(
      comparisons = list(c("fasttext", data$X[1])),
      annotations = pval_ft, textsize = 4,
      y_position = y_position, tip_length = 0, vjust = -0.5
    ) +
    geom_signif(
      comparisons = list(c("fasttext2d", data$X[1])),
      annotations = pval_ft2d, textsize = 4,
      y_position = y_position + 0.07, tip_length = 0, vjust = -0.5
    ) +
    geom_signif(
      comparisons = list(c("controlspace", data$X[1])),
      annotations = pval_control, textsize = 4,
      y_position = y_position + 0.14, tip_length = 0, vjust = -0.5
    ) +
    ggtitle(title) +
    xlab("Condition") +
    ylab("Coefficients") +
    style

  save_to <- paste0("../figures/", file_name)
  dir.create(file.path(dirname(save_to)))
  ggsave(save_to, plot = corrs, width = 6, height = 6)
}

# Select function from command line
args <- commandArgs(trailingOnly = TRUE)

switch(args,
  "plot_colors" = plot_colors(
    file_name = "osgood_coords_it.csv",
    save_colors = "supp3A.svg"
  ),
  "plot_emotions" = plot_emotions(
    file_name = "osgood_coords_it.csv",
    save_emotions = "supp3B.svg"
  ),
  "plot_both" = plot_both(
    file_name = "osgood_coords_it.csv",
    save_both = "supp3C.svg"
  ),
  "rdm_osgood_col" = rdm_corr(
    c(
      "it/sighted/colors_spearman1.csv",
      "it/sighted/colors_spearman2.csv"
    ),
    title = "Colors",
    y_position = 0.6, ylim = c(-0.2, 0.9),
    file_name = "supp4I.svg",
    plabels = c("", "", "", "")
  ),
  "rdm_osgood_em" = rdm_corr(
    c(
      "it/sighted/emotions_spearman1.csv",
      "it/sighted/emotions_spearman2.csv"
    ),
    title = "Emotions",
    y_position = 0.6, ylim = c(-0.1, 0.9),
    file_name = "supp4L.svg",
    plabels = c("***", "***", "", "")
  ),
  "rdm_osgood_both" = rdm_corr(
    c(
      "it/sighted/osgoodspace1.csv",
      "it/sighted/osgoodspace2.csv"
    ),
    title = "Color-Emotion Associations",
    y_position = 0.6, ylim = c(-0.1, 0.9),
    file_name = "supp5E.svg",
    plabels = c("**", "", "", "")
  ),
  "rdm_osgood_col_blind" = rdm_corr(
    c(
      "it/blind/colors_spearman1.csv",
      "it/blind/colors_spearman2.csv"
    ),
    title = "Colors",
    y_position = 0.6, ylim = c(-0.2, 0.9),
    file_name = "supp6I.svg",
    plabels = c("", "", "", "")
  ),
  "rdm_osgood_em_blind" = rdm_corr(
    c(
      "it/blind/emotions_spearman1.csv",
      "it/blind/emotions_spearman2.csv"
    ),
    title = "Emotions",
    y_position = 0.6, ylim = c(-0.1, 0.9),
    file_name = "supp6L.svg",
    plabels = c("***", "***", "", "")
  ),
  "rdm_osgood_both_blind" = rdm_corr(
    c(
      "it/blind/osgoodspace1.csv",
      "it/blind/osgoodspace2.csv"
    ),
    title = "Color-Emotions Associations",
    y_position = 0.6, ylim = c(-0.1, 0.9),
    file_name = "supp7E.svg",
    plabels = c("**", "", "", "")
  ),
)
