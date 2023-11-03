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
glimpse(ggg_train)

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

# Recipe for Naive Bayes
naive_recipe <- recipe(type ~ ., data = ggg_train) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(all_numeric_predictors())


# Naive Bayes ----------------------------

# Set the model
naive_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
  set_mode("classification") |>
  set_engine("naivebayes")

# Set workflow
naive_wf <- workflow() |>
  add_recipe(naive_recipe) |>
  add_model(naive_model)

# Tuning
# Set up the grid with the tuning values
naive_grid <- grid_regular(Laplace(), smoothness())

# Set up the K-fold CV
naive_folds <- vfold_cv(data = ggg_train, v = 20, repeats = 1)

# Find best tuning parameters
naive_cv_results <- naive_wf |>
  tune_grid(resamples = naive_folds,
            grid = naive_grid,
            metrics = metric_set(accuracy))

# Select best tuning parameters
naive_best_tune <- naive_cv_results |> select_best("accuracy")
naive_final_wf <- naive_wf |>
  finalize_workflow(naive_best_tune) |>
  fit(data = ggg_train)

# Make predictions
naive_predictions <- predict(naive_final_wf, new_data = ggg_test, type = "class")
naive_predictions

# Prepare data for export
naive_export <- data.frame("id" = ggg_test$id,
                           "type" = naive_predictions$.pred_class)

# Write the data
vroom_write(naive_export, paste0(base_folder, "naive_bayes.csv"), delim = ",")

