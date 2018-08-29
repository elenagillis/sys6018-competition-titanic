##################################################################
# gillis-sys6018-competition-titanic2.R
# Titanic Kaggle Competition
# Elena Gillis
# August 30, 2018
##################################################################

install.packages("randomForest")
library(randomForest)
library(caret)

# Importing the data
train <- read.csv('./Data/train.csv')
test <- read.csv('./Data/test.csv')

# Delete non-categoric character rows
train[,c('Name','Ticket','Cabin')] <- list(NULL)
test[,c('Name','Ticket','Cabin')] <- list(NULL)

# Identifying categorical variables
train$Pclass <- as.factor(train$Pclass)
train$Sex <- as.factor(train$Sex)
train$Embarked <- as.factor(train$Embarked)
train$Survived <- as.factor(train$Survived)
train$Fare <- as.numeric(train$Fare)

#insert median age and fare for missing values and convert to numeric
train$Age[is.na(train$Age)] <- 0
train$Age[train$Age==0 & train$SibSp>1] <- median(train$Age[train$SibSp>1])
train$Age[train$Age==0 & train$SibSp<=1] <- median(train$Age[train$SibSp<=1])
train$Age <- as.numeric(train$Age)

test$Pclass <- as.factor(test$Pclass)
test$Sex <- as.factor(test$Sex)
test$Embarked <- factor(test$Embarked)
test$Fare <- as.numeric(test$Fare)

#insert median age and fare for missing values and convert to numeric
test$Age[is.na(test$Age)] <- 0
test$Age[test$Age==0 & test$SibSp>1] <- median(test$Age[test$SibSp>1])
test$Age[test$Age==0 & test$SibSp<=1] <- median(test$Age[test$SibSp<=1])
test$Age <- as.numeric(test$Age)
test[is.na(test)]<-0

# Subsetting to cross-validate the model
sub <- sample(1:891,size=623)
sub.train <- train[sub,]     
sub.valid <- train[-sub,]

# optimizing number of parameters (picking random variables)
bestmtry <- tuneRF(sub.train, sub.train$Survived, stepFactor=1.2,improve=0.01,trace=T,plot=T)

# Random forest
titanic.sub.forest <- randomForest(Survived~.-PassengerId, data=sub.train)

importance(titanic.sub.forest) # shows importance of each model variable
varImpPlot(titanic.sub.forest) # plots importance of each model variable

# test model
predict.sub.titanic <- predict(titanic.sub.forest, newdata=sub.valid, type='response')

# print accuracy of confusion matrix on tested set
confusionMatrix(table(predict.sub.titanic, sub.valid$Survived))

# combine train and test to match types
index_train <- 1:nrow(train)
index_test <- nrow(train)+1:nrow(train)+nrow(test)
test$Survived <- 0
all.titanic <- rbind(train, test)

split_train <- all.titanic[1:nrow(train),]
split_test <- all.titanic[(nrow(train)+1):(nrow(train)+nrow(test)),]

# use model on entire training data
bestmtry.train <- tuneRF(train, train$Survived, stepFactor=1.2,improve=0.01,trace=T,plot=T)
titanic.forest <- randomForest(Survived~.-PassengerId, data=train)
importance(titanic.forest)
predict.titanic <- as.vector(predict(titanic.forest, newdata = split_test, type = 'response'))

# combine with id column and write to csv
titanic.predict.merged <- as.data.frame(cbind(split_test$PassengerId, predict.titanic))
names(titanic.predict.merged) <- c("PassengerId","Survived")
write.table(titanic.predict.merged, file = "emg3sc_titanic_predict_forest2.csv", 
            row.names=F, col.names=T, sep=",")
