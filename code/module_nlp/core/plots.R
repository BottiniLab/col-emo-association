#! /usr/bin/Rscript

library("ggplot2")
library("ggpubr")
library("ggsignif")
library("ggrepel")
library("scales")

# Suppress existing file warnings
options(warn = -1)

# Create custom theme common to all plots
ce_theme <- theme(
  # Panel background color
  panel.background = element_rect(fill = "white"),
  # Axis label color
  axis.text = element_text(color = "#151515", size = 10),
  # Legend settings
  legend.background = element_blank(),
  legend.key = element_blank(),
  legend.text = element_text(color = "#151515"),
  legend.title = element_blank(),
  # Tick settings
  axis.ticks = element_line(color = "#151515"),
  # Direction of ticks
  axis.ticks.length = unit(0.15, "cm"),  # Major tick size
  # Axis line
  axis.line.x = element_line(color = "black", linewidth = 0.5),
  axis.line.y = element_line(color = "black", linewidth = 0.5),
  # Font family
  text = element_text(family = "Gotham, Helvetica, Helvetica Neue, Arial, Liberation Sans, DejaVu Sans, Bitstream Vera Sans, sans-serif", 
                      size = 10, colour = "#151515"),
  # Grid lines
  panel.grid.major = element_line(color = "white"),
  panel.grid.minor = element_line(color = "white"),
)

# Plot figures 7A and 7B
plot_osgood <- function(file_name, save_colors, save_emotions,
                      xlab = 'Valence', ylab = 'Arousal', style = ce_theme) {
  
  data <- read.csv(paste0("../data/nlp/coordinates/", file_name), header = T, sep = ",", dec = ".", fill = T)
  data$index <- toupper(data$index)
  colors <- data[data$Condition == "Color", ]
  emotions <- data[data$Condition == "Emotion", ]
  
  scatter_color <- ggplot(colors, mapping = aes(x = z_valence, y = z_arousal, label = index)) +
    geom_point(aes(size = 6, colour = "#CBA6F7"), show.legend = F) +
    scale_color_identity() +
    geom_text(size = 0) +
    scale_x_continuous(name=xlab)+
    scale_y_continuous(name=ylab)+
    geom_text_repel() +
    style
  # scale_x_continuous(breaks = seq(min(colors$z_valence), max(colors$z_valence), by = 0.25)) +
  # scale_y_continuous(breaks = seq(min(colors$z_arousal), max(colors$z_arousal), by = 0.25))
  save_to <- paste0("../figures/", save_colors)
  dir.create(file.path(dirname(save_to)))
  ggsave(save_to, plot=scatter_color, width = 6, height = 6)
  
  scatter_emotions <- ggplot(emotions, mapping = aes(x = z_valence, y = z_arousal, label = index)) +
    geom_point(aes(size = 6, colour = "#CBA6F7"), show.legend = F) +
    scale_color_identity() +
    geom_text(size = 0) +
    scale_x_continuous(name=xlab)+
    scale_y_continuous(name=ylab)+
    geom_text_repel() +
    style
  save_to <- paste0("../figures/", save_emotions)
  dir.create(file.path(dirname(save_to)))
  ggsave(save_to, plot=scatter_emotions, width = 6, height = 6)
}

# Plot figure 7E 
plot_both <- function(file_name, is_geo = F, save_both,
                      xlab = 'Valence', ylab = 'Arousal', style = ce_theme) {

  data <- read.csv(paste0("../data/nlp/coordinates/", file_name), header = T, sep = ",", dec = ".", fill = T)
  data$index <- toupper(data$index)
  if (!is_geo) {
  cols <- as.character(data$fill_color)
  names(cols) <- as.character(data$index)
  both <- ggplot(data, mapping = aes(x = z_valence, y = z_arousal, label = index)) +
    geom_point(aes(size = 6, colour = fill_color), show.legend = F) +
    scale_color_identity() +
    geom_text(size = 0) +
    scale_x_continuous(name=xlab)+
    scale_y_continuous(name=ylab)+
    geom_text_repel() +
    style
  
  save_to <- paste0("../figures/", save_both)
  dir.create(file.path(dirname(save_to)))
  ggsave(save_to, plot=both, width = 6, height = 6)
  }
}

# Select function from command line
args = commandArgs(trailingOnly = TRUE)

switch (args,
  "plot_osgood" = plot_osgood(file_name = "osgood_coords.csv",
                          save_colors = "7A.svg",
                          save_emotions = "7B.svg"),
  "plot_both" = plot_both(file_name = "osgood_coords.csv",
                          save_both = "7E.svg")
)
