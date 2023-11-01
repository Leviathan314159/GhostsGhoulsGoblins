# This R file is analysis for the Ghosts, Ghouls, and Goblins dataset

# Libraries ---------------------
library(tidymodels)
library(vroom)
library(tidyverse)
library(embed)
library(ranger)
library(discrim)
library(themis) # for SMOTE

# Read in the data -----------------------

base_folder <- "GhostsGhoulsGoblins/"
ggg_train <- vroom(paste0(base_folder, "train.csv"))
ggg_test <- vroom(paste0(base_folder, "test.csv"))
ggg_missing_train <- vroom(paste0(base_folder, "trainWithMissingValues.csv"))
glimpse(ggg_missing_train)

# Recipes ------------------------

k_val <- 8
num_trees <- 1000

# ggg_predict_missing_recipe <- recipe(type ~ ., data = ggg_missing_train) |> 
#   step_impute_knn(bone_length ~ ., data = ggg_missing_train, impute_with = imp_vars(all_predictors()), neighbors = k_val) |> 
#   step_impute_knn(rotting_flesh ~ ., data = ggg_missing_train, impute_with = imp_vars(all_predictors()), neighbors = k_val) |> 
#   step_impute_knn(hair_length ~ ., data = ggg_missing_train, impute_with = imp_vars(all_predictors()), neighbord = k_val)

# Impute missing values
ggg_impute_knn <- recipe(type ~ ., data = ggg_missing_train) |> 
  step_mutate_at(all_nominal_predictors(), fn = factor) |> 
  step_impute_knn(all_predictors(), impute_with = imp_vars(all_predictors()), neighbors = k_val)

ggg_impute_rf <- recipe(type ~ ., data = ggg_missing_train) |> 
  step_mutate_at(all_nominal_predictors(), fn = factor) |> 
  step_impute_bag(all_predictors(), impute_with = imp_vars(all_predictors()), trees = num_trees)

ggg_impute_knn_baked <- prep(ggg_impute_knn) |> bake(new_data = ggg_missing_train)
glimpse(ggg_impute_knn_baked)

ggg_impute_rf_baked <- prep(ggg_impute_rf) |> bake(new_data = ggg_missing_train)
glimpse(ggg_impute_rf_baked)

# Calculate the rmse for the imputations
rmse_vec(ggg_train[is.na(ggg_missing_train)], ggg_impute_knn_baked[is.na(ggg_missing_train)])
rmse_vec(ggg_train[is.na(ggg_missing_train)], ggg_impute_rf_baked[is.na(ggg_missing_train)])

# Find the best imputation value
k_val_vec <- 3:50
rmse_values <- 3:50
for (i in 1:length(k_val_vec)) {
  temp_recipe <- recipe(type ~ ., data = ggg_missing_train) |> 
    step_mutate_at(all_nominal_predictors(), fn = factor) |> 
    step_impute_knn(all_predictors(), impute_with = imp_vars(all_predictors()), neighbors = k_val_vec[i])
  temp_baked <- prep(temp_recipe) |> bake(new_data = ggg_missing_train)
  
  rmse_values[i] <- rmse_vec(ggg_train[is.na(ggg_missing_train)], temp_baked[is.na(ggg_missing_train)])
}
k_val_frame <- data.frame(k_val_vec, rmse_values)
ggplot(data = k_val_frame, aes(x = k_val_vec, y = rmse_values)) + 
  geom_line(color = "firebrick", linewidth=1) + xlab("K Values") + ylab("RMSE Values")
ggsave(paste0(base_folder, "Imputation RMSE by Number of Neighbors.png"))
k_val_frame |> slice_min(order_by = rmse_values, n = 3)

# The best imputation value is k = 10, rmse = 0.13773