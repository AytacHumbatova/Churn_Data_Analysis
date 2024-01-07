# 1. Import dataset.
# 2. Remove unneeded columns.
# 3. Build Churn model.
# 4. Compare model results for training and test sets.
# 5. Evaluate and explain model results using ROC & AUC curves.
# 6. Name your final homework Script as “Churn_Modelling”.
# 7. Create repository named “Churn_Data_Analysis” in your Github account and push your homework Script to this repository.
# 8. Fork other users’ repositories, make pull requests (at least one, making three pull requests is desirable).

library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)
library(yardstick)
library(ggsci)

df <- read_csv("Churn_Modelling (1).csv")
glimpse(df)
skim(df)

df$Exited <- df$Exited %>% as.factor()
df$Exited %>% table() %>% prop.table()

#Excluding unimportant variables by IV(information value) (.0.02) 
ivars <- df %>% 
  iv(y = 'Exited') %>% 
  as.tibble() %>% 
  mutate(info_value = round(info_value, 3)) %>% 
  arrange(desc(info_value))

ivars <- ivars %>% filter(info_value > 0.02)

ivars <- ivars[[1]]

df <- df %>% select(Exited, ivars) 

#While appliying woebin() there's an error about surname column which has more tahn 50 unique variables, 
#so I have excluded that column.
df <- df %>% select(-4)

#Split data into train and test sets using seed=123
dt_list <- df %>% split_df('Exited', ratio = 0.8, seed =123) 

#Applying binning according to Weight of Evidence principle
bins <- df %>% woebin('Exited')

train_woe <- dt_list$train %>% woebin_ply(bins)
test_woe <- dt_list$test %>% woebin_ply(bins)

names <- names(train_woe)
names <- gsub('_woe', '', names) 

names(train_woe) <- names
names(test_woe) <- names

#Building a logistic regression model
target <- 'Exited'
features <- train_woe %>% select(-Exited) %>% names()

h2o.init()

train_h2o <- train_woe %>% select(target, features) %>% as.h2o()
test_h2o <- test_woe %>% select(target, features) %>% as.h2o()

model <- h2o.glm(
  x=features, y=target, family='binomial',
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

model %>% show()

if(model@model$coefficients_table %>% as.data.frame() %>% 
      select(names, p_value) %>% 
      mutate(p_value = round(p_value, 3)) %>% 
      arrange(desc(p_value)) %>% .[1,2]>=0.05){
  model@model$coefficients_table%>% as.data.frame() %>% 
    select(names, p_value) %>% 
    mutate(p_value = round(p_value, 3)) %>% 
    arrange(desc(p_value)) %>% .[1,1] -> v
  features <- features[features!=v]
}

h2o.init()

train_h2o <- train_woe %>% select(target, features) %>% as.h2o()
test_h2o <- test_woe %>% select(target, features) %>% as.h2o()

model <- h2o.glm(
  x=features, y=target, family='binomial',
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

model %>% show()

#Evaluation
#predictions
pred <- model %>% h2o.predict(newdata = test_h2o) %>% as.data.frame() %>% 
  select(p1, predict) %>% mutate(p1 = round(p1, 3))

#f1 score
model %>% h2o.performance(newdata = test_h2o) %>% 
  h2o.find_threshold_by_max_metric('f1')

#ROC curve
eva <- perf_eva(
  pred=pred %>% pull(p1),
  label=dt_list$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c('auc', 'gini'),
  show_plot = 'roc',
  confusion_matrix = TRUE)

eva$binomial_metric

eva$confusion_matrix









