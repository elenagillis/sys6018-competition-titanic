##################################################################
# gillis-sys6018-competition-titanic.R
# Titanic Kaggle Competition
# Elena Gillis
# August 30, 2018
##################################################################

# Importing the data
train <- read.csv('./Data/train.csv')
test <- read.csv('./Data/test.csv')

# Looking at the data
head(train)

# Identifying categorical variables
train$Pclass <- factor(train$Pclass)
train$Sex <- factor(train$Sex)
train$Embarked <- factor(train$Embarked)

test$Pclass <- factor(test$Pclass)
test$Sex <- factor(test$Sex)
test$Embarked <- factor(test$Embarked)

# Subsetting to cross-validate the model
sub <- sample(1:891,size=446)
sub.train <- train[sub,]     
sub.valid <- train[-sub,]

# Logistic regression model on class, age, sex, and siblings
sub.train.lm1 <- glm(Survived~Pclass+Sex+Age+SibSp, data=sub.train, family = "binomial")
summary(sub.train.lm1)

# Testing on validation set
probs1 <-as.vector(predict(sub.train.lm1, newdata = sub.valid, type="response"))
preds1 <- rep(0,445)  
preds1[probs1>0.5] <- 1
table(preds1,sub.valid$Survived)

# preds1   0   1
#      0 240  61
#      1  38 106

# Logistic regression model on class, age, sex
sub.train.lm2 <- glm(Survived~Pclass+Sex+Age, data=sub.train, family = "binomial")
summary(sub.train.lm2)

# Testing on validation set
probs2<-as.vector(predict(sub.train.lm2, newdata = sub.valid, type="response"))
preds2 <- rep(0,445)  
preds2[probs2>0.5] <- 1
table(preds2,sub.valid$Survived)

# preds2   0   1
#      0 240  62
#      1  38 105

# Logistic regression model on class and sex
sub.train.lm3 <- glm(Survived~Pclass+Sex, data=sub.train, family = "binomial")
summary(sub.train.lm3)

# Testing on validation set
probs3<-as.vector(predict(sub.train.lm3, newdata = sub.valid, type="response"))
preds3 <- rep(0,445)  
preds3[probs3>0.5] <- 1
table(preds3,sub.valid$Survived)


# preds3   0   1
#      0 245  51
#      1  33 116


# Using model 3 for prediction: Logistic regression model based on class and sex
train.lm <- glm(Survived~Pclass+Sex, data=train, family = "binomial")
summary(train.lm)

# predicting and writing to csv file
probs <- as.vector(predict(train.lm, newdata = test, type="response"))
titanic.predict <- rep(0,418) 
titanic.predict[probs>0.5] <- 1 
titanic.predict.merged <- as.data.frame(cbind(test$PassengerId, titanic.predict))
names(titanic.predict.merged) <- c("PassengerId","Survived")
write.table(titanic.predict.merged, file = "titanic_predict.csv", 
            row.names=F, col.names=T, sep=",")

