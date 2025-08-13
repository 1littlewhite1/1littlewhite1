library(xlsx)
library(ggplot2)
library(mgcv)
library(dplyr)
library(readxl)
shap_values <- read_excel("D:/shap_values.xlsx")
feature_data <- read_excel("D:/features.xlsx")

feature_names <- colnames(shap_values)[-1] 

for (feature in feature_names) {
  data <- data.frame(variable = feature_data[[feature]], shap_values = shap_values[[feature]])
  
  temp_data <- scale(data$variable, center = TRUE, scale = TRUE)
  temp_index <- which((temp_data > 3) | (temp_data < -3))
  if (length(temp_index) > 0) {
    data <- data[-temp_index,]
  }
  
  gam_model <- gam(shap_values ~ s(variable), data = data)

  gam_summary <- summary(gam_model)
  r2 <- round(gam_summary$r.sq, 3)
  p_value <- round(gam_summary$s.pv, 3)
  
  target_function <- function(x) {
    predict(gam_model, newdata = data.frame(variable = x)) - 0
  }
  
  root <- tryCatch({
    uniroot(target_function, range(data$variable))$root
  }, error = function(e) {
    NA
  })
  
  xscale <- max(data$variable) - min(data$variable)
  yscale <- max(data$shap_values) - min(data$shap_values)
  
  p <- ggplot(data, aes(x = variable, y = shap_values)) +
    geom_point(size = 1, color = '#F7BB91') +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") + 
    geom_smooth(method = "gam", formula = y ~ s(x)) +
    annotate(geom = "text", x = 0.9 * max(data$variable), y = 0.8 * max(data$shap_values),
             label = as.character(as.expression(substitute(italic(R)^2 ~ '=' ~ b, list(b = r2)))),
             parse = TRUE, size = 5) +
    annotate(geom = "text", x = 0.9 * max(data$variable), y = 0.45 * max(data$shap_values),
             label = as.character(as.expression(substitute(italic(p) ~ '<' ~ 0.05))),
             parse = TRUE, size = 5) +
    labs(x = feature, y = "SHAP values", title = paste("GAM of", feature)) +
    theme_bw() +
    theme(text = element_text(size = 20, colour = "black"),
          axis.text = element_text(colour = "black"),
          plot.title = element_text(size = 1, hjust = 0.5, color = "black"))
  if (!is.na(root)) {
    p <- p + geom_vline(xintercept = root, linetype = "dashed", color = "#0F7FEF") +
      annotate(geom = "text", x = root, y = 0, label = paste("x =", round(root, 2)),
               vjust = -1, color = "black", size = 5)
  }
  if (!is.na(root)) {
    print(paste("SHAP 值为 0 时的", feature, "特征值:", root))
  } else {
    print(paste("SHAP 值为 0 时的", feature, "特征值无法找到"))
  }
  #ggsave(paste0('返修相关/Q/plot', feature, '.pdf'), plot = p, width = 5, height = 3.5, units = 'in', dpi = 300)
}
  print(p)


