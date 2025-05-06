#======================================================================= 
# 
# Exploratory Data Analysis - CardioGoodFitness 
# 
#=======================================================================

# Environment set up and data import

# Invoking libraries
library(readr) # To import csv files
library(ggplot2) # To create plots
library(corrplot) # To plot correlation plot between numerical variables
library(gridExtra) # To plot multiple ggplot graphs in a grid
library(DataExplorer) # visual exploration of data
library(caTools) # Split Data into Test and Train Set
library(caret) # for confusion matrix function
library(randomForest) # to build a random forest model
library(rpart) # to build a decision model
library(rattle) 
library(gbm) # basic implementation using AdaBoost
library(xgboost) # to build a XGboost model
library(DMwR) # for sMOTE
library(knitr) # Necessary to generate source codes from a .Rmd File
library(markdown) # To convert to HTML
library(rmarkdown) # To convret analyses into high quality documents

# Set working directory 
setwd("C:/Users/egwuc/Desktop/PGP-DSBA-UT Austin/Machine Learning/Week 5 - Project/")

# Read input file
cars_dataset <- read.csv("Cars-dataset.csv")

# Global options settings
options(scipen = 999) # turn off scientific notation like 1e+06

# Check dimension of dataset 
dim(cars_dataset)

# Check first 6 rows(observations) of dataset
head(cars_dataset)
tail(cars_dataset)

# Check structure of dataset
str(cars_dataset)

# Get summary of dataset
summary(cars_dataset)

# How many missing vaues do we have?
sum(is.na(cars_dataset)) 

# What columns contain missing values?
colSums(is.na(cars_dataset))

# Impute the missing value with the column mean/median
data1 = cars_dataset
data1$MBA[is.na(data1$MBA)] <- median(data1$MBA, na.rm = T)
dim(data1)
cars_dataset <- data1
sum(is.na(cars_dataset))

# Change Engineer, MBA and license to factor variable
cars_dataset$Engineer <- as.factor(cars_dataset$Engineer)
cars_dataset$MBA <- as.factor(cars_dataset$MBA)
cars_dataset$license <- as.factor(cars_dataset$license)

# View the dataset 
View(cars_dataset)

# Distribution of the dependent variable
prop.table(table(cars_dataset$Transport))*100

plot_histogram_n_boxplot = function(variable, variableNameString, binw){
  
  a <- ggplot(data = cars_dataset, aes(x = variable)) +
    labs(x = variableNameString, y = 'count')+
    geom_histogram(fill = 'green', col = 'white', binwidth = binw) +
    geom_vline(aes(xintercept = mean(variable)),
               color = "black", linetype = "dashed", size = 0.5)
  
  b <- ggplot(data = cars_dataset, aes('',variable))+ 
    geom_boxplot(outlier.colour = 'red', col = 'red', outlier.shape = 19)+
    labs(x = '', y = variableNameString) + coord_flip()
  grid.arrange(a,b,ncol = 2)
}

plot_histogram_n_boxplot(cars_dataset$Age, 'Age', 2)

plot_histogram_n_boxplot(cars_dataset$Work.Exp, 'Work Experience', 2)

plot_histogram_n_boxplot(cars_dataset$Salary, 'Salary', 5)

plot_histogram_n_boxplot(cars_dataset$Distance, 'Distance', 2)

ggplot(cars_dataset, aes(x = Gender, fill = Transport)) + 
  geom_bar(position = "dodge") + 
  labs(y = "Count", 
       fill = "Transport",
       x = "Gender",
       title = "Gender by Transport") +
  theme_minimal()

ggplot(cars_dataset, aes(x = Engineer, fill = Transport)) + 
  geom_bar(position = "dodge") + 
  labs(y = "Count", 
       fill = "Transport",
       x = "Engineer",
       title = "Engineer by Transport") +
  theme_minimal()

ggplot(cars_dataset, aes(x = MBA, fill = Transport)) + 
  geom_bar(position = "dodge") + 
  labs(y = "Count", 
       fill = "Transport",
       x = "MBA",
       title = "MBA by Transport") +
  theme_minimal()

ggplot(cars_dataset, aes(x = license, fill = Transport)) + 
  geom_bar(position = "dodge") + 
  labs(y = "Count", 
       fill = "Transport",
       x = "License",
       title = "License by Transport") +
  theme_minimal()

# Numeric variables in the data
num_vars = sapply(cars_dataset, is.numeric)

# Correlation Plot
corrplot(cor(cars_dataset[,num_vars]), method = 'number')

# Distribution of the Transport variable
prop.table(table(cars_dataset$Transport))*100

# Adding a new column titled "Carusage"
# Given we want to determine employees who use a car or not, we will use 
# "Car" to represent "Car" and "Not Car" to represent "2Wheeler" and "Public Transport".
cars_dataset$Carusage <- ifelse(cars_dataset$Transport == "Car", "Car", "Not.Car")
table(cars_dataset$Carusage)
prop.table(table(cars_dataset$Carusage))*100

# The Carusage variable needs to be converted to a factor variable  
cars_dataset$Carusage <- as.factor(cars_dataset$Carusage)
summary(cars_dataset)

# Remove the Transport variable
cars_dataset <- cars_dataset[,-9]
view(cars_dataset)

# Split the data into train and test 
set.seed(123)
carsdataset_index <- createDataPartition(cars_dataset$Carusage, p = 0.70, list = FALSE)

carsdataset_train <- cars_dataset[carsdataset_index,]
carsdataset_test <- cars_dataset[-carsdataset_index,]

prop.table(table(cars_dataset$Carusage))*100
prop.table(table(carsdataset_train$Carusage))*100
prop.table(table(carsdataset_test$Carusage))*100

# Apply SMOTE on the Train dataset
table(carsdataset_train$Carusage)
prop.table(table(carsdataset_train$Carusage))*100

smote_carsdataset_train <- SMOTE(Carusage ~ ., data = carsdataset_train,
                     perc.over = 500,
                     perc.under = 200,
                     k = 5)

table(smote_carsdataset_train$Carusage)
prop.table(table(smote_carsdataset_train$Carusage))*100

# perc.over	
# how many extra cases from the minority class are generated (known as over-sampling)

# smoted_minority_class = perc.over/100 * minority_class_cases + minority_class_cases

# perc.under	
# how many extra cases from the majority classes are selected for each case generated from the minority class (known as under-sampling)

# k: number of nearest neighbours that are used to generate the new examples of the minority class.

# Define the training control
fitControl <- trainControl(
              method = 'repeatedcv',           # k-fold cross validation
              number = 3,                      # number of folds or k
              repeats = 1,                     # repeated k-fold cross-validation
              allowParallel = TRUE,
              classProbs = TRUE,
              summaryFunction = twoClassSummary # should class probabilities be returned
    ) 

knn_model <- train(Carusage ~ ., data = smote_carsdataset_train,
                   preProcess = c("center", "scale"),
                   method = "knn",
                   tuneLength = 3,
                   trControl = fitControl)
knn_model

knn_prediction_test <- predict(knn_model, newdata = carsdataset_test, type = "raw")
confusionMatrix(knn_prediction_test, carsdataset_test$Carusage)

varImp(object = knn_model)
plot(varImp(object = knn_model))

nb_model <- train(Carusage ~ ., data = smote_carsdataset_train,
                 method = "naive_bayes",
                 trControl = fitControl)

summary(nb_model)

nb_prediction_test <- predict(nb_model, newdata = carsdataset_test, type = "raw")
confusionMatrix(nb_prediction_test, carsdataset_test$Carusage)

varImp(object = nb_model)
plot(varImp(object = nb_model))

slr_model <- train(Carusage ~ ., data = smote_carsdataset_train,
                 method = "glm",
                 family = "binomial",
                 trControl = fitControl)

summary(slr_model)

slr_prediction_test <- predict(slr_model, newdata = carsdataset_test, type = "raw")
confusionMatrix(slr_prediction_test, carsdataset_test$Carusage)

# se"N"sitivity : True "P"ositive rate
# s"P"ecificity : True "N"egative rate

varImp(object = slr_model)
plot(varImp(object = slr_model))

rf_model <- train(Carusage ~ ., data = smote_carsdataset_train,
                     method = "rf",
                     ntree = 30,
                     maxdepth = 5,
                     tuneLength = 10,
                     trControl = fitControl)

rf_prediction_test <- predict(rf_model, newdata = carsdataset_test, type = "raw")
confusionMatrix(rf_prediction_test, carsdataset_test$Carusage)

varImp(object = rf_model)
plot(varImp(object = rf_model))

gbm_model <- train(Carusage ~ ., data = smote_carsdataset_train,
                     method = "gbm",
                     trControl = fitControl,
                     verbose = FALSE)

gbm_prediction_test <- predict(gbm_model, newdata = carsdataset_test, type = "raw")
confusionMatrix(gbm_prediction_test, carsdataset_test$Carusage)

varImp(object = gbm_model)
plot(varImp(object = gbm_model))

cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)

    xgb.grid <- expand.grid(nrounds = 500,
                            eta = c(0.01),
                            max_depth = c(2,4),
                            gamma = 0,               #default=0
                            colsample_bytree = 1,    #default=1
                            min_child_weight = 1,    #default=1
                            subsample = 1            #default=1
    )

    xgb_model <-train(Carusage~.,
                     data=smote_carsdataset_train,
                     method="xgbTree",
                     trControl=cv.ctrl,
                     tuneGrid=xgb.grid,
                     verbose=T,
                     nthread = 2
    )

xgb_prediction_test <- predict(xgb_model, newdata = carsdataset_test, type = "raw")
confusionMatrix(xgb_prediction_test, carsdataset_test$Carusage)

varImp(object = xgb_model)
plot(varImp(object = xgb_model))

models_to_compare <- list(KNN = knn_model,
                   Naive_Bayes = nb_model,
                   Logistic_Regression = slr_model,
                   Random_Forest = rf_model,
                   Gradient_Boosting = gbm_model,
                   Xtreme_Gradient_Boosting = xgb_model)
resamp <- resamples(models_to_compare)
resamp
summary(resamp)

Name = c("KNN", "Naive_Bayes", "Logistic_Regression", "Random_Forest", "Gradient_Boosting", "Xtreme_Gradient_Boosting")
Accuracy = c(0.97, 0.97, 0.98, 1.0, 0.99, 0.99)
Sensitivity=c(0.80, 0.90, 0.90, 1.0, 0.90, 0.90)
Specificity=c(0.99, 0.98, 0.99, 1.0, 1.0, 1.0)
output = data.frame(Name, Accuracy, Sensitivity, Specificity)
output

#======================================================================= 
# 
# T H E - E N D 
# 
#=======================================================================

# Generate the .R file from this .Rmd to hold the source code 

purl("Cars Case Study.Rmd", documentation = 0)
