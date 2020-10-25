###########################################################################
############### PROJECT STATISTICAL LEARNING APPENDIX 1 ##################
##########################################################################

# Nicole Maria Formenti, 941481

######################## DATA CLEANSING ##############################

#datasets downloaded from 
#importing datasets 
clev = read.csv('processed.cleveland.data',header=F)
swiss = read.csv('processed.switzerland.data',header=F)
va = read.csv('processed.va.data',header=F)  #Long beach VA
hung = read.csv('processed.hungarian.data',header=F)
names = c('age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',
          'slope','ca','thal','num') 
colnames(hung) = names
colnames(clev) = names
colnames(va) = names
colnames(swiss) = names 

#dicotomize the response variable
clev$num = ifelse(clev$num >0,1,0)
va$num = ifelse(va$num >0,1,0)
swiss$num = ifelse(swiss$num >0,1,0)

#creating a single dataset
heart_data = rbind(swiss, clev, va, hung)

#fixing data type
heart_data$num = as.factor(heart_data$num)

#checking nan values
library(dplyr)
dim(heart_data) #920 14
for(i in seq(1:length(heart_data))) {
  heart_data[,i] = na_if(heart_data[,i], '?')} 
#how many nan values there are in each column
for(i in seq(1:length(heart_data))) {
  s = sum(is.na(heart_data[,i]))/length(heart_data[,i])
  cat('Percentage of Nan values in', colnames(heart_data)[i], ':', s*100,'\n')}

#get rid of thal and ca columns
heart_data = heart_data[,-c(12,13)]

#get rid of rows with nan values
heart_data_clean = heart_data[complete.cases(heart_data),] 

#fix data type
heart_data_clean$fbs = factor(heart_data_clean$fbs)
heart_data_clean$restecg = factor(heart_data_clean$restecg)
heart_data_clean$exang = factor(heart_data_clean$exang)
heart_data_clean$slope = factor(heart_data_clean$slope)
heart_data_clean$thalach = as.integer(heart_data_clean$thalach)
heart_data_clean$oldpeak = as.integer(heart_data_clean$oldpeak)
heart_data_clean$trestbps = as.integer(heart_data_clean$trestbps)
heart_data_clean$chol = as.integer(heart_data_clean$chol)
heart_data_clean$sex = as.factor(heart_data_clean$sex)
heart_data_clean$cp = as.factor(heart_data_clean$cp)
heart_data_clean$num = as.factor(heart_data_clean$num)

row.names(heart_data_clean) = seq(1,dim(heart_data_clean)[1])

#statistics of data
summary(heart_data_clean)
#boxplot of data to check outliers
library(ggplot2)
library(ggrepel)

ggplot(heart_data_clean, aes(y=trestbps, x=factor(0)))+
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=4, fill='#AED6F1')+
  geom_jitter(width=0.25)+
  labs(title='TRESTBPS', x=' ', y='trestbps')+
  theme_gray(base_size = 10)+
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5))

ggplot(heart_data_clean, aes(y=chol, x=factor(0)))+
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=4, fill='#D7BDE2')+
  geom_jitter(width=0.25)+
  labs(title='CHOL', x=' ', y='chol')+
  theme_gray(base_size = 10)+
  theme(plot.title = element_text(size = 20, face = "bold", hjust = 0.5))


#removing outliers
heart_data_clean = heart_data_clean[(heart_data_clean$chol>100) & (heart_data_clean$chol<500),]

#checking dimensions and fixing rows number
dim(heart_data_clean) #335 12
rownames(heart_data_clean) = 1:dim(heart_data_clean[1])[1]

#label balance
table(heart_data_clean$num) #135 200


#################### LOGISTIC REGRESSION ####################
library(glmnet)
library(boot)

#for reproducibility
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
#split training and dataset
train = sample(1:nrow(heart_data_clean),335*0.7)
X_test = heart_data_clean[-train,]
y_test = heart_data_clean$num[-train]

#fit logistig regression
logi_regr = glm(num~., data=heart_data_clean, subset=train, family=binomial)
summary(logi_regr)

#predict probabilities for data points in the validation set
prob = predict(logi_regr, X_test, type='response') 
prob[1:10]

#predict classes for data points in the validation set
class = ifelse(prob>=0.50, '1', '0')
#confusion matrix
t = table(class, y_test)
#compute correctly classify instances according to recall rate
recall = t[4]/(t[4]+t[3])
recall #95.16

#perform 10-fold cross validation for the average recall rate
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
v = trunc(seq(1,335,length.out=10))
recall_logi = c()

for(i in seq(1,(length(v)-1))) {
  test = seq(v[i],v[i+1])
  train = setdiff(1:335,test)
  X_train = heart_data_clean[train,] 
  y_train = heart_data_clean$num[train] 
  
  logi_regr_full = glm(num~., data=X_train, family=binomial)
  prob = predict(logi_regr_full, heart_data_clean[test,], type='response')  
  class = ifelse(prob>=0.40, '1', '0')
  t = table(class, heart_data_clean$num[test])
  
  recall = t[4]/(t[4]+t[3]) 
  recall_logi = c(recall_logi, recall)
  
  if(is.na(recall_logi[i])) { 
    cat('null value at position:',i)
    print(t)
    #to check for confusion matrix that may have one row or
    #one column
  }
}

recall_logi[6] = 33/(33)
mean(recall_logi) #89.51


######################## BEST SUBSET SELECTION #################
library(leaps)
#fit best subset selection
best_sub = regsubsets(num~., data=heart_data_clean, nvmax=11) 
summary(best_sub)

#plot the Cp score against the different models
plot(best_sub, scale='Cp')


################## FORWARD STEPWISE SUBSET SELECTION ###############
forw = regsubsets(num~., data=heart_data_clean, nvmax=11, method='forward') 
plot(forw, scale='Cp')


################## BACKWARD STEPWISE SUBSET SELECTION ###############
forw = regsubsets(num~., data=heart_data_clean, nvmax=11, method='backward') 
plot(forw, scale='Cp')


############################ LASSO  #############################
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
#split in training and validation set
train = sample(1:nrow(heart_data_clean),335*0.7)
x = model.matrix(num~.-1, data=heart_data_clean) 
y = heart_data_clean$num

#fit the lasso
lasso = glmnet(x[train,],y[train], alpha=1, family='binomial') #we must specify famiy=binomial since we have a binary response (logistic regression)

#plot the values of coefficients against log(lambda)
plot(lasso, xvar='lambda', label=TRUE)
print(lasso)

#10-fold cross-validation to select the best lambda
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
cv_lasso= cv.glmnet(x[train,],y[train],family='binomial',alpha=1, nfolds=10)

#plot the binomial deviance against log(lambda)
plot(cv_lasso)
cv_lasso$lambda.min #0.01647649
cv_lasso$lambda.1se #0.05031683

#lasso coefficients
coef(cv_lasso)

#predict probabilities for data points in validation set
prob_lasso_train = predict(cv_lasso, newx=x[train,], type='response') 
prob_lasso_train[1:10]

#predict classes for data points in validation set and recall rate
class_lasso_test = predict(cv_lasso, x[-train,], type='class') 
t = table(class_lasso_test, true=y[-train]) 
recall = t[4]/(t[4]+t[3]) 
recall #91.94 

#nested-cross validation to find the average recall rate
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
v = trunc(seq(1,335,length.out=10))
recall_l = c()

for(i in seq(1,(length(v)-1))){
  test = seq(v[i],v[i+1])
  x = model.matrix(num~.-1, data=heart_data_clean) 
  y = heart_data_clean$num
  train = setdiff(1:335,test)
  X_train = heart_data_clean[train,] 
  y_train = heart_data_clean$num[train] 
  
  cv_lasso= cv.glmnet(x[train,], y_train, family='binomial',alpha=1, nfolds=10)
  class_lasso_test = predict(cv_lasso, x[test,], type='class') 
  t = table(class_lasso_test, true=y[test]) 
  
  recall = t[4]/(t[4]+t[3]) 
  recall_l = c(recall_l,recall)
  
  if(is.na(recall_l[i])) {
    cat('null value at position:',i)
    print(t)
  }
}

mean(recall_l) #86.97 


######################## RIDGE REGRESSION  #########################
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
#split in training and validation set
train = sample(1:nrow(heart_data_clean),335*0.7)
x = model.matrix(num~.-1, data=heart_data_clean) 
y = heart_data_clean$num

#fit ridge regression
ridge = glmnet(x[train,],y[train], alpha=0, family='binomial')

#plot the values of coefficients against log(lambda)
plot(ridge, xvar='lambda', label=TRUE)

#10-folds cross validation to select the best parameter lambda
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
cv_ridge= cv.glmnet(x[train,],y[train],family='binomial',alpha=0)

#plot the binomial deviance against log(lambda)
plot(cv_ridge)
cv_ridge$lambda.min #0.07471889 ####
cv_ridge$lambda.1se #0.3310511 ########

#show ridge regression coefficients
coef(cv_ridge)

#predict probabilities for validation set
prob_ridge_train = predict(cv_ridge, newx=x[train,], type='response') 
prob_ridge_train[1:10]

#predict classes for validation set and recall rate
class_rr_test = predict(cv_lasso, x[-train,], type='class') 
t = table(class_rr_test, true=y[-train])
recall = t[4]/(t[4]+t[3]) 
recall #88.71 

#nested cross-validation to find the average recall rate
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
v = trunc(seq(1,335,length.out=10))
recall_r = c()

for(i in seq(1,(length(v)-2))){
  test = seq(v[i],v[i+1])
  x = model.matrix(num~.-1, data=heart_data_clean) 
  y = heart_data_clean$num
  train = setdiff(1:335,test)
  X_train = heart_data_clean[train,] 
  y_train = heart_data_clean$num[train] 
  
  cv_rr= cv.glmnet(x[train,], y[train], family='binomial',alpha=0, nfolds=10)
  class_rr_test = predict(cv_lasso, x[test,], type='class') 
  t = table(class_rr_test, true=y[test])
  
  recall = t[4]/(t[4]+t[3]) 
  recall_r = c(recall_r, recall)
  
  if(is.na(recall_r[i])) {
    cat('null value at position:',i)
    print(t)
  }
}

mean(recall_r) #83.52


##################### CLASSIFICATION TREE ######################
library(tree)
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
#split training and validation set
train = sample(1:nrow(heart_data_clean),335*0.7)
X_test = heart_data_clean[-train,]
y_test = heart_data_clean$num[-train]

#fit a full grown tree
tree_heart = tree(num~., data=heart_data_clean, subset=train)
#show splitting rules
tree_heart

#show the plot of the tree
plot(tree_heart); text(tree_heart)

#predict classes for data points in validation set and calculate recall
tree_pred = predict(tree_heart, new=X_test, type='class')
t = table(tree_pred, y_test)
recall = t[4]/(t[4]+t[3]) 
recall #83.87 

set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
#use 10 fold cross-validation to prune the tree
cv_tree_heart = cv.tree(tree_heart, FUN=prune.misclass, K=10) 

#data frame with the number of leaves and corresponding cost-complexity
#parameter k (α) and cross validation error rate
summ=data.frame('n_leaves'=cv_tree_heart$size, 'alpha'= cv_tree_heart$k, 
'CV_error'=cv_tree_heart$dev)
#look at the plot of miscalssification rate against the number of
#leaves to select the best tree dimension
plot(cv_tree_heart)

#use validation method to see the recall for the different
#tree dimensions 
c=cut(1:335,10)
for(i in seq(7,11)) {
  pruned_tree = prune.misclass(tree_heart, best=i)
  tree_pred = predict(pruned_tree, new=X_test, type='class')
  t = table(tree_pred, y_test)
  
  recall = t[4]/(t[4]+t[3]) 
  
  cat('Recall for',i,'number of nodes:',recall,'\n')
} #90.32

#fit a new model using a pruned tree
pruned_tree = prune.misclass(tree_heart, best=7) 
#plot the tree model
plot(pruned_tree); text(pruned_tree)
#show tree splitting rules
pruned_tree

#select the optimal number of leaves according to cross validation,
#corresponding to the lowest CV error
best_nleaves=summ[summ$CV_error==min(summ$CV_error),1] 

#use nested cross validation to compute the average recall rate for tree
library(tree)
library(parallel)
#parallelization to speed up the calculations
cores=detectCores()
cl=makeCluster(cores-1) 
cl

set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
v = trunc(seq(1,335,length.out=10))
heart_data_clean$num = as.factor(heart_data_clean$num)
recall_tree = c()

for(i in seq(1,(length(v)-1))){
  test = seq(v[i],v[i+1])
  X_test = heart_data_clean[test,]
  y_test = heart_data_clean$num[test]
  train = setdiff(1:335,test)
  X_train = heart_data_clean[train,] 
  
  tree_heart = tree(num~., data=heart_data_clean, subset=train)
  cv_tree_heart = cv.tree(tree_heart, FUN=prune.misclass, K=10) 
  summ=data.frame('n_leaves'=cv_tree_heart$size, 'alpha'= cv_tree_heart$k, 'CV_error'=cv_tree_heart$dev)
  best_nleaves=min(summ[summ$CV_error==min(summ$CV_error),1])
  
  pruned_tree = prune.misclass(tree_heart, best=best_nleaves)
  tree_pred = predict(pruned_tree, new=X_test, type='class')
  t = table(tree_pred, y_test)
  
  recall = t[4]/(t[4]+t[3]) 
  
  recall_tree = c(recall_tree, recall)
  
  if(is.na(recall_tree[i])) {
    cat('null value at position:',i)
    print(t)
  }
}

mean(recall_tree) #84.94 


########################## BAGGING ################################
library(randomForest)
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
#split in training and validation set
train = sample(1:nrow(heart_data_clean),335*0.7)
X_test = heart_data_clean[-train,]
y_test = heart_data_clean$num[-train]

#fit a bagging model with 500 trees
bag_heart = randomForest(num~., data=heart_data_clean, subset=train, mtry=11, importance=TRUE, n.trees=500)

#recover confusion matrix from the model
t = bag_heart$confusion
#calculate recall rate on the training set for comparison with OOB
recall = t[4]/(t[4]+t[3]) 
recall #76.55

#plot OOB errors against the number of trees
plot(bag_heart, main='OOB error vs. n. of trees')
#plot only the average OOB error
plot(y=bag_heart$err.rate[,1], x=seq(1,500), type='l', col='red', xlab='ntrees', ylab='OOB error', main='OOB vs. number of trees') 

#look at variables importance ranking according to bagging
bag_heart$importance
varImpPlot(bag_heart, main='bagging features importance')

#predict classes for data points in the validation set and recall rate
bag_test = predict(bag_heart, new=X_test)
t = table(bag_test, true=y_test)
recall = t[4]/(t[4]+t[3]) 
recall #91.93

#use nested cross-validation to calculate average recall rate
library(randomForest)
library(parallel)
library(caret)
library(earth)
#parallelization
cores=detectCores()
cl=makeCluster(cores-1) 
cl

v = trunc(seq(1,335,length.out=10))
heart_data_clean$num = as.factor(heart_data_clean$num)
recall_bag = c()

set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')

for(i in seq(1,(length(v)-1))){
  test = seq(v[i],v[i+1])
  X_test = heart_data_clean[test,]
  y_test = heart_data_clean$num[test]
  train = setdiff(1:335,test)
  
  control = trainControl(method = "cv", number = 10, allowParallel = T, search = 'grid')
  bag_heart = train(num~., data = heart_data_clean[train,], method = "treebag", trControl = control, ntree=1000, verbose=FALSE)
  bag_test = predict(bag_heart, new=X_test)
  t = table(bag_test, true=y_test)
  
  recall = t[4]/(t[4]+t[3]) 
  recall_bag = c(recall_bag, recall)
  
  if(is.na(recall_bag[i])) {
    cat('null value at position:',i)
    print(t)
  }
}

stopCluster(cl)

mean(recall_bag) #81.88 


######################### RANDOM FOREST ##############################
library(randomForest)
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
#split in training and validation set
train = sample(1:nrow(heart_data_clean),335*0.7)
X_test = heart_data_clean[-train,]
y_test = heart_data_clean$num[-train]


#use 10-fold cross-validation to find the optimal number of 
#subsampled predictors
library(caret)
library(parallel)
#parallelization
cores=detectCores()
cl=makeCluster(cores-1) 
cl

rf_grid = expand.grid(.mtry = c(1:11))
control = trainControl(method = "cv", number = 10, allowParallel = T, search = 'grid')
rf_heart = train(num~., data=heart_data_clean[train,], method='rf', 
                    metric='Accuracy', tuneGrid=rf_grid, trControl=control)

stopCluster(cl)

#look at the optimal hyperparameter
rf_heart$bestTune 

#predict classes for validation data and calculate recall
rf_test = predict(rf_heart, new=X_test)
t = table(rf_test, true=y_test)
recall = t[4]/(t[4]+t[3]) 
recall #93.55 

#plot the variable importance 
rfor_heart = randomForest(num~., data=heart_data_clean, subset=train, 
                          mtry=3, importance=TRUE, n.trees=1000)
varImpPlot(rfor_heart, main='random forest features importance')


#use nested cross validation to find the average recall
library(randomForest)
library(caret)
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
v=trunc(seq(1,335,length.out=10))
heart_data_clean$num = as.character(heart_data_clean$num)
recall_rf2 = c()

cores=detectCores()
cl=makeCluster(cores-1) 
cl

for(i in seq(1,(length(v)-1))){
  test = seq(v[i],v[i+1])
  X_test = heart_data_clean[test,]
  y_test = heart_data_clean$num[test]
  train = setdiff(1:335,test)
  
  rf_grid = expand.grid(.mtry = c(1:11))
  control = trainControl(method = "cv", number = 10, allowParallel = T, search = 'grid')
  rf_heart = train(num~., data=heart_data_clean[train,], method='rf', 
                   metric='Accuracy', tuneGrid=rf_grid, trControl=control)
  
  rf_test = predict(rf_heart, new=X_test, type='raw')
  t = table(rf_test, true=y_test)
  recall = t[4]/(t[4]+t[3]) 
  recall_rf2 = c(recall_rf2, recall)
  
  if(is.na(recall_rf2[i])) {
    cat('null value at position:',i)
    print(t)
  }
}

stopCluster(cl) 

recall_rf2[9] = 35/(35+4)
mean(recall_rf2) #86.18 


############################ BOOSTING ############################

### 1) GRADIENT BOOSTING
library(gbm)
library(caret)
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
library(parallel)
#split in training and validation set
train = sample(1:nrow(heart_data_clean),335*0.7)
X_test = heart_data_clean[-train,]
y_test = heart_data_clean$num[-train]

heart_data_clean$num = as.character(heart_data_clean$num)
#parallelization
cores=detectCores()
cl=makeCluster(cores-1) 
cl

#use 5-fold cross-validation 3 times to find the optimal
#hyperparameters
control = trainControl(method = "repeatedcv", number = 5, repeats=3,  allowParallel = T) 
gbm_grid = expand.grid(interaction.depth = 1:4, n.trees = seq(100, 5000,by=500), shrinkage = c(0.01, 0.001), n.minobsinnode = 10) 
model_gbm = train(num~., data = heart_data_clean[train,], method = "gbm", trControl = control, tuneGrid=gbm_grid, metric='Accuracy', verbose=0)
stopCluster(cl)

#look at the optimal value for hyperparameters
best = model_gbm$bestTune
n.trees = best$n.trees

#predict classes for test set and calculate recall rate
boost_test = predict(model_gbm, newdata=X_test, n.trees=n.trees, type='raw')
t = table(boost_test, true=y_test)
recall = t[4]/(t[4]+t[3]) 
recall #93.55


#use nested-cross validation to find the average recall rate
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
heart_data_clean$num = as.character(heart_data_clean$num)
recall_b1 = c()

cores=detectCores()
cl=makeCluster(cores-1) 
cl

for(i in seq(1,(length(v)-1))){
  test = seq(v[i],v[i+1])
  X_test = heart_data_clean[test,]
  y_test = heart_data_clean$num[test]
  train = setdiff(1:335,test)
  
  control = trainControl(method = "repeatedcv", number = 5, repeats=3,  allowParallel = T) #???
  gbm_grid = expand.grid(interaction.depth = 1:4, n.trees = seq(100, 5000,by=500), shrinkage = c(0.01, 0.001), n.minobsinnode = 10) 
  model_gbm = train(num~., data = heart_data_clean[train,], method = "gbm", trControl = control, tuneGrid=gbm_grid, metric='Accuracy', verbose=0)
  
  best = model_gbm$bestTune
  n.trees = best$n.trees
  
  boost_test = predict(model_gbm, newdata=X_test, n.trees=n.trees, type='raw')
  t = table(boost_test, true=y_test)
  
  recall = t[4]/(t[4]+t[3]) 
  recall_b1 = c(recall_b1, recall)
  
  if(is.na(recall_b1[i])) {
    cat('null value at position:',i)
    print(t)
  }
  
}

stopCluster(cl)

recall_b1[9] = 34/(34+5)
mean(recall_b1) #88


### 2) ADABOOST
library(caret)
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
library(parallel)
library(fastAdaboost)
#split in training and validation set
train = sample(1:nrow(heart_data_clean),335*0.7)
X_test = heart_data_clean[-train,]
y_test = heart_data_clean$num[-train]

heart_data_clean$num = as.character(heart_data_clean$num)

#use 5 fold cross validation to find the optimal hyperparameter value
cores=detectCores()
cl=makeCluster(cores-1) 
cl

control = trainControl(method = "cv", number = 5, allowParallel = T)
ada_grid = expand.grid(nIter = seq(100, 1000,by=200), method='Adaboost.M1')
model_ada = train(num~., data = heart_data_clean[train,], method = "adaboost", trControl = control, tuneGrid=ada_grid, metric='Accuracy', verbose=FALSE)

stopCluster(cl)

#look at the optimal value for the hyperparameter
model_ada$bestTune

#predict classes for validation data and calculate recall
ada_test = predict(model_ada, newdata=X_test, type='raw')
t = table(ada_test, true=y_test)
recall = t[4]/(t[4]+t[3]) 
recall #90.32


#use 5-fold cross validation to find the average recall rate
set.seed(1, kind='Mersenne-Twister',
         normal.kind='Inversion', sample.kind='Rounding')
heart_data_clean$num = as.character(heart_data_clean$num)
recall_ada = c()

cores=detectCores()
cl=makeCluster(cores-1) 
cl

for(i in seq(1,(length(v)-1))){
  test = seq(v[i],v[i+1])
  X_test = heart_data_clean[test,]
  y_test = heart_data_clean$num[test]
  train = setdiff(1:335,test)
  
  control = trainControl(method = "cv", number = 5, allowParallel = T)
  ada_grid = expand.grid(nIter = seq(100, 1000,by=200), method='Adaboost.M1')
  model_ada = train(num~., data = heart_data_clean[train,], method = "adaboost", trControl = control, tuneGrid=ada_grid, metric='Accuracy', verbose=FALSE)
  
  ada_test = predict(model_ada, newdata=X_test, type='raw')
  t = table(ada_test, true=y_test)
  
  recall = t[4]/(t[4]+t[3]) 
  recall_ada = c(recall_ada, recall)
  
  if(is.na(recall_ada[i])) {
    cat('null value at position:',i)
    print(t)
  }
  
}

stopCluster(cl)

recall_ada[9] = 32/(32+7)
mean(recall_ada) #80.58
