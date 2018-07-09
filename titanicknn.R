#KNN

setwd("/home/kenelly/workspaces/titanic/")

titanicdf <- read.csv("train.csv", header = T, na.strings = c(""), stringsAsFactors = T)

#checking missing values
library(Amelia)

sapply(titanicdf,function(x) sum(is.na(x)))
missmap(titanicdf, legend = TRUE, main = "Missing values vs observed", x.cex = 0.8, y.cex = 0.8)

#As cabin has more than 50% missing values, it will be removed. Te id will be also removed from the model.

titanicdf <- titanicdf[, -c(1, 11)]
str(titanicdf)

#Transforming some features to factor

titanicdf$Pclass <- as.factor(titanicdf$Pclass)
titanicdf$Survived <- factor(titanicdf$Survived, levels = c(0, 1), labels = c("No", "Yes") )

#Handling the 2 embarked missing values
titanicdf[is.na(titanicdf$Embarked),]
#checking the fare to infer where this 2 were embarked
mean(titanicdf[titanicdf$Embarked == 'C', "Fare" ], na.rm = TRUE)
#Filling both with C
titanicdf[is.na(titanicdf$Embarked), 'Embarked'] <- 'C'

#Use Randon Forest to impute missing values in Age column
library(mice)
micemodel <- mice(titanicdf[, !names(titanicdf) %in% c('Name', 'Ticket')], 
                  method = 'rf')
miceoutput <- complete(micemodel)
titanicdf$Age <- miceoutput$Age

#Creating a feature called Child

titanicdf$Child[titanicdf$Age<18] <- 'Yes'
titanicdf$Child[titanicdf$Age>18] <- 'No'
titanicdf$Child <- as.factor(titanicdf$Child)

#Creating the feature Title

titanicdf$Title <- gsub('(.*, )|(\\..*)', '', titanicdf$Name)

table(titanicdf$Title)

raretitle <-c('Dona', 'Lady', 'The Countess', 'Capt', 'Col', 'Don', 'Dr',
              'Major', 'Rev', 'Sir', 'Jonkheer')

titanicdf$Title[titanicdf$Title == 'Mlle'|titanicdf$Title == 'Ms'] <- 'Miss'
titanicdf$Title[titanicdf$Title == 'Mme'] <- 'Mrs'
titanicdf$Title[titanicdf$Title %in% raretitle] <- 'Rare Title'

titanicdf$Title <- as.factor(titanicdf$Title)

#Splitting Name and Surname
titanicdf$Name <- as.character(titanicdf$Name)
titanicdf$Surname <- sapply(titanicdf$Name, FUN = function(x) {strsplit(x, split = '[,.]')[[1]][1]})

#Family size
titanicdf$Famsize <- titanicdf$SibSp+titanicdf$Parch +1
titanicdf$FamsizeCateg[titanicdf$Famsize ==1] <- 'Alone'
titanicdf$FamsizeCateg[titanicdf$Famsize<5 && titanicdf$Famsize >1] <- 'small'
titanicdf$FamsizeCateg[titanicdf$Famsize>4] <-'large'
titanicdf$FamsizeCateg <- as.factor(titanicdf$FamsizeCateg)

#Removing some features : Name, Ticket, Surname
titanicdf[3] <- NULL
titanicdf[7] <- NULL
titanicdf[11] <- NULL
str(titanicdf)

#Encoding
contrasts(titanicdf$Sex)
contrasts(titanicdf$Pclass)

#checking the numeric features
numeric <- sapply(titanicdf, function(x) {is.numeric(x)})
numeric

numericfeat <- titanicdf[, numeric]
summary(numericfeat)
str(numericfeat)
#normalization
normalize <- function(x) {return((x - min(x))/(max(x)-min(x)))}
numericnormal <- normalize(numericfeat)
summary(numericnormal)

#checking the histogram before/after the normalization
hist(numericfeat$Age)
hist(numericnormal$Age)

#Encoding
titanicdfknn <- titanicdf[,!numeric]
titanicdfknn <- cbind(titanicdfknn, numericnormal)

library(dummies)
tknn <- dummy.data.frame(titanicdfknn[,-1])
summary(tknn)
str(tknn)
str(titanicdfknn)
#Isolating the dependent variable
Survived <- titanicdfknn$Survived
tknn$survived <-titanicdfknn$Survived

#Creating holdout
set.seed(1337)
index <- sample(1:dim(tknn)[1], dim(tknn)[1]*.80, replace = FALSE)
trainingknn <-tknn[index,]
testingknn <-tknn[-index,]
trainingknn <- trainingknn[,-25]
testingknn <- testingknn[,-25]
str(trainingknn)
survivedtrain <- trainingknn$survived
survivedtest <- testingknn$survived

#Selecting k value
k1 <- round(sqrt(dim(trainingknn)[1]))
k2 <- round(sqrt(dim(trainingknn)[2]))
k3 <- 7

#Running KNN
sapply(survivedtrain,function(x) sum(is.na(x)))
library(class)
knn1 <- knn(train = trainingknn, test = testingknn, cl = survivedtrain, k = k1)
knn2 <- knn(train = trainingknn, test = testingknn, cl = survivedtrain, k = k2)
knn3 <- knn(train = trainingknn, test = testingknn, cl = survivedtrain, k = k3)

#Performance Evaluation
library(caret)
confusionMatrix(knn1, survivedtest,positive = "Yes")
confusionMatrix(knn2, survivedtest,positive = "Yes")
confusionMatrix(knn3, survivedtest,positive = "Yes")

library(MLmetrics)
F1_Score(y_true =  as.numeric(survivedtest), y_pred =  as.numeric(knn1), positive = "1")
F1_Score(y_true =  as.numeric(survivedtest), y_pred =  as.numeric(knn2), positive = "1")
F1_Score(y_true =  as.numeric(survivedtest), y_pred =  as.numeric(knn3), positive = "1")
str(knn1)

