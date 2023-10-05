
### -------- Install Required Packages -------- ###
#install.packages("tidyverse")
#install.packages("caret")
#install.packages("Hmisc")
#install.packages("tictoc")
#install.packages("keras")
#install.packages("ROSE")
#install.packages("pROC")
#install.packages("tfruns")
#install.packages("ggplot2")
#install.packages("tensorflow")

# RESET ENV ON RE-RUNS
#rm(list=ls())

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORTANT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

# THIS CODE REQUIRES CODE TO BE RAN IN THE WORKING DIRECOTRY OF CHOICE, BUT THE FOLLOWING
# FOLDERS ARE REQUIRED WITHIN THIS DIRECTORY

# /results/
# /standard_results/
# /oversamp_results/
# /undersamp_results/

# Complimentary File also needed for HYPERPARAM TUNING: `neural_network.R`

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

### -------- Load Required Packages -------- ###
library(tidyverse)
library(dplyr)
library(caret)
library(Hmisc)
library(keras)
library(tictoc)
library(tidyr)
library(pROC)
library(tfruns)
library(ggplot2)
library(ROSE)
library(tensorflow)
#install_keras(method="conda")



### Load Pre-Processed CSV ###

f1_data <- read.csv('formula_1_cleansed.csv')

### ~~~~~~~~~~~~~~~~~~~~~~~ Extra Pre-Processing ~~~~~~~~~~~~~~~~~~~~~~~ ###

# Filter for all races from 2005 onwards - want to use records with speed stats
f1_data <- f1_data[f1_data$year >= 2005, ]

# Add a column which indicates win or not
f1_data$win <- ifelse(f1_data$positionOrder==1, 1, 0)

# Add a column which indicates podium or not
f1_data <- f1_data %>%
  mutate(podium = ifelse(positionOrder < 4, 1, 0))

# Add a cumulative sum of race wins per season for each driver
f1_data <- f1_data %>% 
  group_by(year, driverRef) %>% 
  arrange(raceId) %>% 
  mutate(season_wins = cumsum(win))

f1_data <-  f1_data %>%
  mutate(season_wins = ifelse(season_wins > 0, season_wins-1, season_wins))

# Add a cumulative sum of race podiums per season for each driver
f1_data <- f1_data %>% 
  group_by(year, driverRef) %>% 
  arrange(raceId) %>% 
  mutate(season_podiums = cumsum(podium))

f1_data <-  f1_data %>%
  mutate(season_podiums = ifelse(season_podiums > 0, season_podiums-1, season_podiums))

# Get previous fastest speed for driver
f1_data <- f1_data %>%
  group_by(driverRef) %>% 
  arrange(year, round, driverRef) %>%
  mutate(last_top_speed = ifelse(driverRef == lag(driverRef), lag(fastestLapSpeed), 0))

# Select data that will be used in model
ml_data <- f1_data %>% 
  select(year, round, grid, driverRef, driver_nationality, driver_age, season_wins, season_podiums, last_top_speed,
         constructor_name, constructor_nationality, circuitRef, country, lat, lng, win)

# Set NAs to 0 <- NAs only exist in last_top_speed column
ml_data[is.na(ml_data)] <- 0


### ~~~~~~~~~~~~~~~~~~~~~ Preparing Data for Model ~~~~~~~~~~~~~~~~~~~~~ ###

# Normalise Numeric Data 
Scaler <- preProcess(ml_data, method = "range")
ScaledData <- predict(Scaler,ml_data)

# Apply OHE to Categorical Data
dmy <- dummyVars(" ~ .", data = ScaledData)
one_hot_df <- data.frame(predict(dmy, newdata = ScaledData))


# Remove non-required variables
rm(list=c('dmy', 'ScaledData', 'Scaler', 'f1_data'))

# set margins
par(mar=c(5.1, 4.1, 4.1, 2.1))
# Plot Win (1) to Lose (0) rows distribution
barplot(prop.table(table(one_hot_df$win)),
        col=c("lightgrey", "red"),
        ylim=c(0,1),
        main="Distribution of Race Winner Rows")

### ~~~~~~~ Splitting Data ~~~~~~~ ###
# get folds split evenly by 3 year intervals, for 6 folds in total
fold_years <- split(sample(unique(one_hot_df$year), replace=FALSE), rep(1:6, each=3))

# assign all but first fold to training
ind <- which(one_hot_df$year %in% fold_years[[1]]) 
training <- one_hot_df[-ind,]

### ~~~~~~~ Training Splits ~~~~~~ ###

# STANDARD DATA
train_x <- as.matrix(training[, -which(names(training) == "win")])
train_y <- as.matrix(training[, which(names(training) == "win")])

# OVER SAMPLING
over_train <- ovun.sample(win~., data=training, method="over")$data
over_train_x <- as.matrix(over_train[, -which(names(over_train) == "win")])
over_train_y <- as.matrix(over_train[, "win"])

table(over_train_y)

# UNDER SAMPLING
under_train <- ovun.sample(win~., data=training, method="under")$data
under_train_x <- as.matrix(under_train[, -which(names(under_train) == "win")])
under_train_y <- as.matrix(under_train[, "win"])

table(under_train_y)


# Set Hyperparameters that we want to test
hyper_params <- list('dense_units'=c(32, 64),
                     'activation_function' = 'relu',
                     'learning_rate' = c(0.001, 0.01),
                     'drop_out' = c(0.1, 0.2),
                     'epochs' = c(30, 40, 50),
                     'batch_size' = c(32, 64),
                     'validation_split' = c(0.1, 0.2))


# ~~~~~~~~~~~~~~~~~~~~~~~~~ Standard Model ~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Hyperparameter Tuning - Standard
x <- train_x
y <- train_y
standard_runs <- tuning_run("neural_network.R",
                            flags = hyper_params,
                            sample=0.05)

# Save tested Hyperparams and select best performing params
standard_params <- standard_runs[order(standard_runs$metric_binary_accuracy, decreasing = TRUE),]
write.csv(standard_params, gsub("/", "\\\\", paste0("./standard_results/","params.csv")), row.names = T)
standard_best_params <- standard_params[1,]


# Initiate & build Model (3 hidden layers)
cv_model <- keras_model_sequential() %>% 
  layer_dense(units=standard_best_params$flag_dense_units,
              activation = standard_best_params$flag_activation,
              input_shape = ncol(x)) %>%
  layer_dropout(rate=standard_best_params$flag_drop_out) %>%
  layer_dense(units=standard_best_params$flag_dense_units/2,
              activation = standard_best_params$flag_activation) %>%
  layer_dropout(rate=standard_best_params$flag_drop_out) %>%
  layer_dense(units=standard_best_params$flag_dense_units/4,
              activation = standard_best_params$flag_activation) %>%  
  layer_dropout(rate=standard_best_params$flag_drop_out) %>%
  layer_dense(units = 1, activation = "sigmoid")


# Compile
cv_model %>%
  compile(
    loss = "binary_crossentropy",  # loss function
    optimizer = optimizer_adam(learning_rate=standard_best_params$flag_learning_rate),  # Use adam optimizer
    metrics = c("binary_accuracy") # Evaluation metric is binary_accuracy as it is a binary classification problem
  )

# CROSS VALIDATION WITH OPTIMAL PARAMS (5 Folds Training, 1 Fold Test)
for(f in c(1:6)){
  cat("\n Fold:", f, "\n")
  ind <- which(one_hot_df$year %in% fold_years[[f]]) 
  train_df <- as.matrix(one_hot_df[-ind, -which(names(one_hot_df) == "win")])
  y_train <- one_hot_df[-ind, which(names(one_hot_df) == "win")]
  valid_df <- as.matrix(one_hot_df[ind, -which(names(one_hot_df) == "win")])
  y_valid <- one_hot_df[ind, which(names(one_hot_df) == "win")]
  
  model_1 <- cv_model %>% fit(
    x = as.matrix(train_df), y = y_train,
    batch_size = standard_best_params$flag_batch_size,
    epochs = standard_best_params$flag_epochs,
    validation_split = standard_best_params$flag_validation_split)

  # get predictions
  predicted <- cv_model %>% predict(valid_df)
  
  # Convert predictions to a dataframe
  predictions_df <- data.frame(predicted = predicted)
  # Add the predictions as a new column to the original dataframe
  original_df <- as.matrix(ml_data[ind, -which(names(ml_data) == "win")])
  df <- cbind(original_df, y_valid, predictions_df)
  df <- df %>%
    group_by(year, round) %>%
    mutate(win_predict = ifelse(predicted==max(predicted), 1, 0))
  
  # Calc Confusion Matrix
  conf <- confusionMatrix(as.factor(df$y_valid),as.factor(df$win_predict), positive = '1')
  # Display Matrix
  fourfoldplot(as.table(conf),color=c("red","green"),main = "Standard Sampling Confusion Matrix")
  
  # Save original datset with predictions AND confusion matrix to seperate files
  write.csv(as.table(conf), gsub("/", "\\\\", paste0("./standard_results/","win_conf_fold_",f,".csv")), row.names = T)
  write.csv(df, gsub("/", "\\\\", paste0("./standard_results/","win_predicted_fold_",f,".csv")), row.names = F)
  
  
  # Get Feature Importance
  input_layer_weights <- cv_model$layers[[1]]$get_weights()[[1]]
  feature_importances <- rowMeans(abs(input_layer_weights))
  
  # Map Feature Importance to columns
  feat_imp_df <- data.frame(cbind(colnames(train_df), feature_importances))
  colnames(feat_imp_df) <- c('feature', 'importance')
  feature_df <- feat_imp_df[order(feat_imp_df$importance, decreasing = TRUE),]
  # Save table of FI
  write.csv(feature_df, gsub("/", "\\\\", paste0("./standard_results/","featureimportance_fold_",f,".csv")), row.names = F)
  
  # Filter for top 20 (not driverRef) and plot and save
  feature_df <- feature_df %>% filter(!grepl('driverRef', feature))
  display_feat <- top_n(feature_df, 20)
  display_feat$importance <- as.numeric(display_feat$importance)
  par(mar=c(5.1, 12, 4.1, 2.1))
  display_feat %>% ggplot() +
    geom_bar(aes(x = reorder(feature, importance), y = importance), stat = "identity") +
    coord_flip() +
    labs(x = "Feature", y = "Importance", title=paste0("Standard Sample Feature Importance", f)) +
    theme_minimal()
  ggsave(paste0("./standard_results/","featureimportance_foldplot_",f,".png"), last_plot(), bg = 'white')
  par(mar=c(5.1, 4.1, 4.1, 2.1))
  }


################ LESS NOTES BELOW - SAME PROCESS FOR OVER SAMPLE AND UNDER SAMPLE ################ 

# ~~~~~~~~~~~~~~~~~~~~~~~~ Oversampling Model ~~~~~~~~~~~~~~~~~~~~~~~ #

# Hyperparameter Tuning - Over-sampling
x <- over_train_x
y <- over_train_y
over_runs <- tuning_run("neural_network.R",
                        flags = hyper_params,
                        sample=0.05)
over_params <- over_runs[order(over_runs$metric_binary_accuracy, decreasing = TRUE),]
write.csv(over_params, gsub("/", "\\\\", paste0("./oversamp_results/","params.csv")), row.names = T)
over_best_params <- over_params[1,]

# Initiate Model
cv_model <- keras_model_sequential() %>% 
  layer_dense(units=over_best_params$flag_dense_units,
              activation = over_best_params$flag_activation,
              input_shape = ncol(x)) %>%
  layer_dropout(rate=over_best_params$flag_drop_out) %>%
  layer_dense(units=over_best_params$flag_dense_units/2,
              activation = over_best_params$flag_activation) %>%
  layer_dropout(rate=over_best_params$flag_drop_out) %>%
  layer_dense(units=over_best_params$flag_dense_units/4,
              activation = over_best_params$flag_activation) %>%  
  layer_dropout(rate=over_best_params$flag_drop_out) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile
cv_model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate=over_best_params$flag_learning_rate),
    metrics = c("binary_accuracy") 
  )

# CROSS VALIDATION
for(f in c(1:6)){
  cat("\n Fold:", f, "\n")
  ind <- which(one_hot_df$year %in% fold_years[[f]])
  
  #Over Sample Train Data
  train_df <- one_hot_df[-ind, ]
  train_df <- ovun.sample(win~., data=train_df, method="over")$data
  y_train <- train_df[, which(names(train_df) == "win")]
  train_df <- as.matrix(train_df[, -which(names(train_df) == "win")])

  valid_df <- as.matrix(one_hot_df[ind, -which(names(one_hot_df) == "win")])
  y_valid <- one_hot_df[ind, which(names(one_hot_df) == "win")]
  
  model_1 <- cv_model %>% fit(
    x = as.matrix(train_df), y = y_train,
    batch_size = over_best_params$flag_batch_size,
    epochs = over_best_params$flag_epochs,
    validation_split = over_best_params$flag_validation_split)
  
  predicted <- cv_model %>% predict(valid_df)
  
  
  # Convert predictions to a dataframe
  predictions_df <- data.frame(predicted = predicted)
  # Add the predictions as a new column to the original dataframe
  original_df <- as.matrix(ml_data[ind, -which(names(ml_data) == "win")])
  df <- cbind(original_df, y_valid, predictions_df)
  
  df <- df %>%
    group_by(year, round) %>%
    mutate(win_predict = ifelse(predicted==max(predicted), 1, 0))
  
  conf <- confusionMatrix(as.factor(df$y_valid),as.factor(df$win_predict), positive = '1')
  
  fourfoldplot(as.table(conf),color=c("red","green"),main = "Under Sampling Confusion Matrix")
  
  write.csv(as.table(conf), gsub("/", "\\\\", paste0("./oversamp_results/","win_conf_fold_",f,".csv")), row.names = T)
  write.csv(df, gsub("/", "\\\\", paste0("./oversamp_results/","win_predicted_fold_",f,".csv")), row.names = F)
  
  
  input_layer_weights <- cv_model$layers[[1]]$get_weights()[[1]]
  feature_importances <- rowMeans(abs(input_layer_weights))
  
  feat_imp_df <- data.frame(cbind(colnames(train_df), feature_importances))#, row.names = colnames(train_df))
  colnames(feat_imp_df) <- c('feature', 'importance')
  feature_df <- feat_imp_df[order(feat_imp_df$importance, decreasing = TRUE),]
  write.csv(feature_df, gsub("/", "\\\\", paste0("./oversamp_results/","featureimportance_fold_",f,".csv")), row.names = F)
  
  feature_df <- feature_df %>% filter(!grepl('driverRef', feature))
  
  display_feat <- top_n(feature_df, 20)
  
  display_feat$importance <- as.numeric(display_feat$importance)
  par(mar=c(5.1, 12, 4.1, 2.1))
  display_feat %>% ggplot() +
    geom_bar(aes(x = reorder(feature, importance), y = importance), stat = "identity") +
    coord_flip() +
    labs(x = "Feature", y = "Importance", title=paste0("Over Sample Feature Importance", f)) +
    theme_minimal()
  ggsave(paste0("./oversamp_results/","featureimportance_foldplot_",f,".png"), last_plot(), bg = 'white')
  par(mar=c(5.1, 4.1, 4.1, 2.1))
}



# ~~~~~~~~~~~~~~~~~~~~~~ Undersampling Model ~~~~~~~~~~~~~~~~~~~~~~~ #

# Hyperparameter Tuning - Under-sampling
x <- under_train_x
y <- under_train_y
under_runs <- tuning_run("neural_network.R",
                         flags = hyper_params,
                         sample=0.05)
under_params <- under_runs[order(under_runs$metric_binary_accuracy, decreasing = TRUE),]
write.csv(under_params, gsub("/", "\\\\", paste0("./undersamp_results/","params.csv")), row.names = T)
under_best_params <- under_params[1,]


# Initiate Model
cv_model <- keras_model_sequential() %>% 
  layer_dense(units=under_best_params$flag_dense_units,
              activation = under_best_params$flag_activation,
              input_shape = ncol(x)) %>%
  layer_dropout(rate=under_best_params$flag_drop_out) %>%
  layer_dense(units=under_best_params$flag_dense_units/2,
              activation = under_best_params$flag_activation) %>%
  layer_dropout(rate=under_best_params$flag_drop_out) %>%
  layer_dense(units=under_best_params$flag_dense_units/4,
              activation = under_best_params$flag_activation) %>%  
  layer_dropout(rate=under_best_params$flag_drop_out) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Plot the architecture of the neural network
plot_model(cv_model, show_shapes = TRUE, show_layer_names = TRUE)
# Compile
cv_model %>%
  compile(
    loss = "binary_crossentropy", 
    optimizer = optimizer_adam(learning_rate=under_best_params$flag_learning_rate),   
    metrics = c("binary_accuracy")   
  )

# CROSS VALIDATION
for(f in c(1:6)){
  cat("\n Fold:", f, "\n")
  ind <- which(one_hot_df$year %in% fold_years[[f]]) 
  
  #Over Sample Train Data
  train_df <- one_hot_df[-ind, ]
  train_df <- ovun.sample(win~., data=train_df, method="under")$data
  y_train <- train_df[, which(names(train_df) == "win")]
  train_df <- as.matrix(train_df[, -which(names(train_df) == "win")])
  
  
  valid_df <- as.matrix(one_hot_df[ind, -which(names(one_hot_df) == "win")])
  y_valid <- one_hot_df[ind, which(names(one_hot_df) == "win")]
  
  model_1 <- cv_model %>% fit(
    x = as.matrix(train_df), y = y_train,
    batch_size = under_best_params$flag_batch_size,
    epochs = under_best_params$flag_epochs,
    validation_split = under_best_params$flag_validation_split
    validation_data = list(valid_df, y_valid))

  predicted <- cv_model %>% predict(valid_df)
  
  
  # Convert predictions to a dataframe
  predictions_df <- data.frame(predicted = predicted)
  # Add the predictions as a new column to the original dataframe
  original_df <- as.matrix(ml_data[ind, -which(names(ml_data) == "win")])
  df <- cbind(valid_df, y_valid, predictions_df)
  
  df <- df %>%
    group_by(year, round) %>%
    mutate(win_predict = ifelse(predicted==max(predicted), 1, 0))
  
  conf <- confusionMatrix(as.factor(df$y_valid),as.factor(df$win_predict), positive = '1')
  
  fourfoldplot(as.table(conf),color=c("red","green"),main = "Under Sampling Confusion Matrix")
  
  write.csv(as.table(conf), gsub("/", "\\\\", paste0("./undersamp_results/","win_conf_fold_",f,".csv")), row.names = T)
  write.csv(df, gsub("/", "\\\\", paste0("./undersamp_results/","win_predicted_fold_",f,".csv")), row.names = F)
  
  
  input_layer_weights <- cv_model$layers[[1]]$get_weights()[[1]]
  feature_importances <- rowMeans(abs(input_layer_weights))
  
  feat_imp_df <- data.frame(cbind(colnames(train_df), feature_importances))#, row.names = colnames(train_df))
  colnames(feat_imp_df) <- c('feature', 'importance')
  feature_df <- feat_imp_df[order(feat_imp_df$importance, decreasing = TRUE),]
  write.csv(feature_df, gsub("/", "\\\\", paste0("./undersamp_results/","featureimportance_fold_",f,".csv")), row.names = F)
  
  feature_df <- feature_df %>% filter(!grepl('driverRef', feature))
  
  display_feat <- top_n(feature_df, 20)
  
  display_feat$importance <- as.numeric(display_feat$importance)
  par(mar=c(5.1, 12, 4.1, 2.1))
  ggplot(display_feat) +
    geom_bar(aes(x = reorder(feature, importance), y = importance), stat = "identity") +
    coord_flip() +
    labs(x = "Feature", y = "Importance", title=paste0("Under Sample Feature Importance ", f)) +
    theme_minimal()
  ggsave(paste0("./undersamp_results/","featureimportance_foldplot_",f,".png"), last_plot(), bg = 'white')
  par(mar=c(5.1, 4.1, 4.1, 2.1))
}




# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Visualizing Results ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# For each sampling method
# Load Each Folds Conf Mat, Plot and Save and store ALL SENSITIVITY's
stand_sens <- numeric(6)
under_sens <- numeric(6)
over_sens <- numeric(6)
for(f in c(1:6)){
  
  # ~~~~~~~~~~~~~ STANARD SAMPLING ~~~~~~~~~~~~~ #
  standard <- read.csv(paste0(".\\standard_results\\win_conf_fold_", f, ".csv"),
                      colClasses=c("NULL",NA,NA),
                      col.names = c("", "Actual_0", "Actual_1"),
                      row.names = c("Predicted_0", "Predicted_1"))
  plot_tbl <- t(as.matrix(standard))
  tiff(paste0("Standard Sampling Confusion Matrix", f, '.png'))
  fourfoldplot(as.table(plot_tbl),color=c("red","green"),main = paste0("Standard Sampling Confusion Matrix", f)) + 
    text(-0.4,0.4, "TN", cex=1) + 
    text(0.4, -0.4, "TP", cex=1) + 
    text(0.4,0.4, "FN", cex=1) + 
    text(-0.4, -0.4, "FP", cex=1)
  dev.off()
  
  
  standard <- standard %>%
    bind_rows(summarise(.,
                        across(where(is.numeric), sum)))
  
  standard <- rbind(standard, c("sum_of_cols", colSums(standard[,1:2])))
  stand_sens[f] <- round(as.numeric(standard['Predicted_1','Actual_1'])/as.numeric(standard['...3','Actual_1']), digits = 2)
  
  # ~~~~~~~~~~~~~~ UNDER SAMPLING ~~~~~~~~~~~~~~ #
  under_mat <- read.csv(paste0(".\\undersamp_results\\win_conf_fold_", f, ".csv"),
                        colClasses=c("NULL",NA,NA),
                        col.names = c("", "Actual_0", "Actual_1"),
                        row.names = c("Predicted_0", "Predicted_1"))
  plot_tbl <- t(as.matrix(under_mat))
  tiff(paste0("Under Sampling Confusion Matrix", f, '.png'))
  fourfoldplot(as.table(plot_tbl),color=c("red","green"),main = paste0("Under Sampling Confusion Matrix", f)) + 
    text(-0.4,0.4, "TN", cex=1) + 
    text(0.4, -0.4, "TP", cex=1) + 
    text(0.4,0.4, "FN", cex=1) + 
    text(-0.4, -0.4, "FP", cex=1)
  dev.off()
  under_mat <- under_mat %>%
    bind_rows(summarise(.,
                        across(where(is.numeric), sum)))
  
  under_mat <- rbind(under_mat, c("sum_of_cols", colSums(under_mat[,1:2])))
  under_sens[f] <- round(as.numeric(under_mat['Predicted_1','Actual_1'])/as.numeric(under_mat['...3','Actual_1']),digits = 2)
  
  # ~~~~~~~~~~~~~~~ OVER SAMPLING ~~~~~~~~~~~~~~ #
  over_mat <- read.csv(paste0(".\\oversamp_results\\win_conf_fold_", f, ".csv"),
                       colClasses=c("NULL",NA,NA),
                       col.names = c("", "Actual_0", "Actual_1"),
                       row.names = c("Predicted_0", "Predicted_1"))
  plot_tbl <- t(as.matrix(over_mat))
  tiff(paste0("Over Sampling Confusion Matrix", f, '.png'))
  fourfoldplot(as.table(plot_tbl),color=c("red","green"),main = paste0("Over Sampling Confusion Matrix", f)) + 
    text(-0.4,0.4, "TN", cex=1) + 
    text(0.4, -0.4, "TP", cex=1) + 
    text(0.4,0.4, "FN", cex=1) + 
    text(-0.4, -0.4, "FP", cex=1)
  dev.off()
  over_mat <- over_mat %>%
    bind_rows(summarise(.,
                        across(where(is.numeric), sum)))
  
  over_mat <- rbind(over_mat, c("sum_of_cols", colSums(over_mat[,1:2])))
  over_sens[f] <- round(as.numeric(over_mat['Predicted_1','Actual_1'])/as.numeric(over_mat['...3','Actual_1']), digits = 2)
  
}

# Save Sensitivity's to dataframe
final_sens <- as.data.frame(cbind(stand_sens, over_sens, under_sens))
final_sens <- final_sens %>% 
  summarise(across(where(is.numeric), ~ c(., mean(.)))) %>% mutate(across(is.numeric, round, digits=2))
final_sens <- as.matrix(final_sens)

# Plot Sensitivities and SAVE
par(mar=c(5.1, 4.1, 4.1, 2.1))
colours <- c('#002e61', '#F55F55', '#00ddae')
tiff("Sensitivity by Fold and Sample Method.png")
plot <- barplot(t(final_sens),main='Sensitivity by Fold and Sample Method',ylab='Sensitivity', xlab='Fold',beside = TRUE,
        col=colours, ylim=c(0,max(final_sens)*1.5),
        names.arg = c("1", "2", "3", "4", "5", "6", "Avg"))
box()
legend('topright',legend=c('Standard','Over-sample', 'Under-sample'),fill=colours)
text(25.5, t(final_sens)[1,7] + 0.05, labels = t(final_sens)[1,7], cex = 1)
dev.off()

# DISPLAY PREDICTIONS v ACTUAL IN DATAFRAME AND SAVE TO CSV

# Load Data
standard_df <- read.csv(".\\standard_results\\win_predicted_fold_6.csv")

# Get Actual Winners
actual_winners <- standard_df %>%
  group_by(year, round) %>%
  filter(win_predict==1) %>%
  select(year, round, circuitRef, driverRef)
# Get Predicted Winners
predicted_winners<- standard_df %>%
  group_by(year, round) %>%
  filter(y_valid==1) %>%
  select(year, round, circuitRef, driverRef)

# Combine dataframes and WRITE TO CSV
compare_predictions_df <- cbind(actual_winners, predicted_winners$driverRef)
colnames(compare_predictions_df) <- c("year", "round", "circuit", "Actual", "Predicted")
write.csv(compare_predictions_df, "best_performing_fold_standard_sample.csv", row.names = F)
