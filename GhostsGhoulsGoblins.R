# This R file is analysis for the Ghosts, Ghouls, and Goblins dataset

# Libraries ---------------------
library(tidymodels)
library(vroom)
library(tidyverse)
library(embed)
library(ranger)
library(discrim)
library(themis) # for SMOTE
# library(bonsai)
# library(lightgbm)

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

# # Impute missing values
# ggg_impute_knn <- recipe(type ~ ., data = ggg_missing_train) |> 
#   step_mutate_at(all_nominal_predictors(), fn = factor) |> 
#   step_impute_knn(all_predictors(), impute_with = imp_vars(all_predictors()), neighbors = k_val)
# 
# ggg_impute_rf <- recipe(type ~ ., data = ggg_missing_train) |> 
#   step_mutate_at(all_nominal_predictors(), fn = factor) |> 
#   step_impute_bag(all_predictors(), impute_with = imp_vars(all_predictors()), trees = num_trees)
# 
# ggg_impute_knn_baked <- prep(ggg_impute_knn) |> bake(new_data = ggg_missing_train)
# glimpse(ggg_impute_knn_baked)
# 
# ggg_impute_rf_baked <- prep(ggg_impute_rf) |> bake(new_data = ggg_missing_train)
# glimpse(ggg_impute_rf_baked)
# 
# # Calculate the rmse for the imputations
# rmse_vec(ggg_train[is.na(ggg_missing_train)], ggg_impute_knn_baked[is.na(ggg_missing_train)])
# rmse_vec(ggg_train[is.na(ggg_missing_train)], ggg_impute_rf_baked[is.na(ggg_missing_train)])
# 
# # Find the best imputation value
# k_val_vec <- 3:50
# rmse_values <- 3:50
# for (i in 1:length(k_val_vec)) {
#   temp_recipe <- recipe(type ~ ., data = ggg_missing_train) |> 
#     step_mutate_at(all_nominal_predictors(), fn = factor) |> 
#     step_impute_knn(all_predictors(), impute_with = imp_vars(all_predictors()), neighbors = k_val_vec[i])
#   temp_baked <- prep(temp_recipe) |> bake(new_data = ggg_missing_train)
#   
#   rmse_values[i] <- rmse_vec(ggg_train[is.na(ggg_missing_train)], temp_baked[is.na(ggg_missing_train)])
# }
# k_val_frame <- data.frame(k_val_vec, rmse_values)
# ggplot(data = k_val_frame, aes(x = k_val_vec, y = rmse_values)) + 
#   geom_line(color = "firebrick", linewidth=1) + xlab("K Values") + ylab("RMSE Values")
# ggsave(paste0(base_folder, "Imputation RMSE by Number of Neighbors.png"))
# k_val_frame |> slice_min(order_by = rmse_values, n = 3)

# The best imputation value is k = 10, rmse = 0.13773

# Recipe for Naive Bayes
naive_recipe <- recipe(type ~ ., data = ggg_train) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) |> 
  step_normalize(all_numeric_predictors())

# Recipe for MLP
mlp_recipe <- recipe(type ~ ., data = ggg_train) |> 
  update_role(id, new_role = "id") |> 
  step_mutate(color = as.factor(color)) |> 
  step_range(all_numeric_predictors(), min = 0, max = 1) # Scale to [0, 1]

# Recipe for Boosted Trees
boosted_recipe <- recipe(type ~ ., data = ggg_train) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(all_numeric_predictors())

# Recipe for BART
bart_recipe <- recipe(type ~ ., data = ggg_train) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_normalize(all_numeric_predictors())

# Recipe for PCA Random Forest
# Principal Component Analysis Recipes
threshold_value <- 0.7
pca_tree_recipe <- recipe(type ~ ., data = ggg_train) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) |> 
  step_normalize(all_predictors()) |> 
  step_pca(all_predictors(), threshold = threshold_value)

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

# MLP ------------------------------
# # Set the model
# mlp_model <- mlp(hidden_units = tune(), 
#                        epochs = 100) |> #,
#                        # activation = "relu") |>
#   set_engine("nnet") |> # If keras doesn't work, change engine to nnet
#   set_mode("classification")
# # Set workflow
# mlp_wf <- workflow() |>
#   add_recipe(mlp_recipe) |>
#   add_model(mlp_model)
# 
# # Tuning
# # Set up the grid with the tuning values
# mlp_grid <- grid_regular(hidden_units(range = c(1, 15)),
#                          levels = 15)
# 
# # Set up the K-fold CV
# mlp_folds <- vfold_cv(data = ggg_train, v = 15, repeats = 1)
# 
# # Find best tuning parameters
# mlp_cv_results <- mlp_wf |>
#   tune_grid(resamples = mlp_folds,
#             grid = mlp_grid,
#             metrics = metric_set(accuracy))
# 
# # Plot the graph of tuned metrics
# mlp_cv_results |> collect_metrics() |> filter(.metric == "accuracy") |> 
#   ggplot(aes(x = hidden_units, y = mean)) + geom_line()
# ggsave(paste0(base_folder, "Accuracy_vs_num_units.png"))
# 
# # Select best tuning parameters
# mlp_best_tune <- mlp_cv_results |> select_best("accuracy")
# mlp_final_wf <- mlp_wf |>
#   finalize_workflow(mlp_best_tune) |>
#   fit(data = ggg_train)
# 
# # Make predictions
# mlp_predictions <- predict(mlp_final_wf, new_data = ggg_test, type = "class")
# mlp_predictions
# 
# # Prepare data for export
# mlp_export <- data.frame("id" = ggg_test$id,
#                            "type" = mlp_predictions$.pred_class)
# 
# # Write the data
# vroom_write(mlp_export, paste0(base_folder, "mlp.csv"), delim = ",")

# Boosted Trees -----------------------------
# 
# # Set the model
# boosted_model <- boost_tree(tree_depth = tune(),
#                             trees = tune(),
#                             learn_rate = tune()) |>
#   set_mode("classification") |>
#   set_engine("lightgbm")
# 
# # Set workflow
# boosted_wf <- workflow() |>
#   add_recipe(boosted_recipe) |>
#   add_model(boosted_model)
# 
# # Tuning
# # Set up the grid with the tuning values
# boosted_grid <- grid_regular(tree_depth(), trees(), learn_rate())
# 
# # Set up the K-fold CV
# boosted_folds <- vfold_cv(data = ggg_train, v = 10, repeats = 1)
# 
# # Find best tuning parameters
# boosted_cv_results <- boosted_wf |>
#   tune_grid(resamples = boosted_folds,
#             grid = boosted_grid,
#             metrics = metric_set(accuracy))
# 
# # Select best tuning parameters
# boosted_best_tune <- boosted_cv_results |> select_best("accuracy")
# boosted_final_wf <- boosted_wf |>
#   finalize_workflow(boosted_best_tune) |>
#   fit(data = ggg_train)
# 
# # Make predictions
# boosted_predictions <- predict(boosted_final_wf, new_data = ggg_test, type = "class")
# boosted_predictions
# 
# # Prepare data for export
# boosted_export <- data.frame("id" = ggg_test$id,
#                            "type" = boosted_predictions$.pred_class)
# 
# # Write the data
# vroom_write(boosted_export, paste0(base_folder, "boosted_trees.csv"), delim = ",")

# BART ----------------------------
# # Set the model
# bart_model <- parsnip::bart(trees = tune()) |>
#   set_mode("classification") |>
#   set_engine("dbarts")
# 
# # Set workflow
# bart_wf <- workflow() |>
#   add_recipe(bart_recipe) |>
#   add_model(bart_model)
# 
# # Tuning
# # Set up the grid with the tuning values
# bart_grid <- grid_regular(trees())
# 
# # Set up the K-fold CV
# bart_folds <- vfold_cv(data = ggg_train, v = 10, repeats = 1)
# 
# # Find best tuning parameters
# bart_cv_results <- bart_wf |>
#   tune_grid(resamples = bart_folds,
#             grid = bart_grid,
#             metrics = metric_set(accuracy))
# 
# # Select best tuning parameters
# bart_best_tune <- bart_cv_results |> select_best("accuracy")
# bart_final_wf <- bart_wf |>
#   finalize_workflow(bart_best_tune) |>
#   fit(data = ggg_train)
# 
# # Make predictions
# bart_predictions <- predict(bart_final_wf, new_data = ggg_test, type = "class")
# bart_predictions
# 
# # Prepare data for export
# bart_export <- data.frame("id" = ggg_test$id,
#                           "type" = bart_predictions$.pred_class)
# 
# # Write the data
# vroom_write(bart_export, paste0(base_folder, "bart.csv"), delim = ",")

# Random Forest ------------------
pca_forest <- rand_forest(mtry = tune(),
                             min_n = tune(),
                             trees = 1000) |>
  set_engine("ranger") |>
  set_mode("classification")

# Create a workflow using the model and recipe
pca_forest_wf <- workflow() |>
  add_model(pca_forest) |>
  add_recipe(pca_tree_recipe)

# Set up the grid with the tuning values
pca_forest_grid <- grid_regular(mtry(range = c(1, (length(ggg_train)-1))), min_n(), levels = 5)

# Set up the K-fold CV
pca_forest_folds <- vfold_cv(data = ggg_train, v = 10, repeats = 1)

# Find best tuning parameters
forest_cv_results <- pca_forest_wf |>
  tune_grid(resamples = pca_forest_folds,
            grid = pca_forest_grid,
            metrics = metric_set(accuracy))

# Finalize the workflow using the best tuning parameters and predict
# The best parameters were mtry = 9 and min_n = 2

# Find out the best tuning parameters
best_forest_tune <- forest_cv_results |> select_best("roc_auc")

# Use the best tuning parameters for the model
forest_final_wf <- pca_forest_wf |>
  finalize_workflow(best_forest_tune) |>
  fit(data = ggg_train)

pca_forest_predictions <- predict(forest_final_wf,
                                     new_data = ggg_test,
                                     type = "class")
pca_forest_predictions

forest_export <- data.frame("id" = ggg_test$id,
                           "type" = pca_forest_predictions$.pred_class)

# Write the data
vroom_write(forest_export, paste0(base_folder, "random_forest_classification.csv"), delim =",")
