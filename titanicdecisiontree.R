#Decision Tree

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

titanicdf$Child[titanicdf$Age<=18] <- 'Yes'
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

#Creatig Holdout

set.seed(1337)
index <- sample(1:dim(titanicdf)[1], dim(titanicdf)[1]*.75, replace = FALSE)
trainingdtree <- titanicdf[index,]
testingdtree <- titanicdf[-index,]

str(titanicdf)
#CART
library(rpart)
library(rpart.plot)
library(RColorBrewer)
#Regression Tree using all the features
regressiontree1 <- rpart(Survived ~., data = trainingdtree, method = "class")
#Regression Tree using selected features
regressiontree2 <- rpart(Survived ~ Pclass + Title + Fare + Famsize, data = trainingdtree, method = "class")
regressiontree3 <- rpart(Survived ~ Pclass + Sex + Fare + Age, data = trainingdtree, method = "class")

plot(regressiontree1)
text(regressiontree1)

plot(regressiontree2)
text(regressiontree2)
summary(regressiontree1)

#tree visualization
library(rattle)
fancyRpartPlot(regressiontree1)
fancyRpartPlot(regressiontree2)
fancyRpartPlot(regressiontree3)

#Evaluation 
library(caret)
decisiontree_prediction <- predict(regressiontree1, testingdtree, type = "class")
library(MLmetrics)
F1_Score(y_true =  as.numeric(testingdtree$Survived), 
         y_pred =  as.numeric(decisiontree_prediction), positive = "1")
confusionMatrix(decisiontree_prediction, testingdtree$Survived, positive = "Yes")

#including some parameters on rpart function - forcing the overfitting
newrpart <- rpart(Survived ~., data = trainingdtree, method = "class", control = rpart.control(minsplit = 2, cp=0))
fancyRpartPlot(newrpart)

#evaluating againg
library(caret)
newrpart_prediction <- predict(newrpart, testingdtree, type = "class")
library(MLmetrics)
F1_Score(y_true =  as.numeric(testingdtree$Survived), 
         y_pred =  as.numeric(newrpart_prediction), positive = "1")
confusionMatrix(newrpart_prediction, testingdtree$Survived, positive = "Yes")

#pruning the tree
prunedtree <- prp(newrpart, snip = TRUE)$obj
fancyRpartPlot(prunedtree)


#Random Forest
library(randomForest)
rforestmodel <- randomForest(Survived ~., data = trainingdtree, importance = TRUE, ntree = 2000) 

#checing the relevance of features
varImpPlot(rforestmodel)

#Evaluation of Random Forest
rforest_prediction <- predict(rforestmodel, testingdtree, type = "class")
confusionMatrix(rforest_prediction, testingdtree$Survived, positive = "Yes")
F1_Score(y_true =  as.numeric(testingdtree$Survived), 
         y_pred =  as.numeric(rforest_prediction), positive = "1")

#Conditional Inference Trees (CI)
library(partykit)
citree <- ctree(Survived ~., data = trainingdtree)

#Visualisation
print(citree)
plot(citree, type = "simple")

#Evaluation of CI Tree
citree_prediction <- predict(citree, testingdtree)
confusionMatrix(citree_prediction, testingdtree$Survived, positive =  "Yes")
F1_Score(y_true =  as.numeric(testingdtree$Survived), 
          y_pred =  as.numeric(citree_prediction), positive = "1")

#There is a CForest that it is an extension of CI Tree.
library(party)
ciforest <- cforest(Survived ~., data = trainingdtree, controls = cforest_unbiased(ntree=2000, mtry=3))
print(ciforest)

#Rotation Forest - only numeric features
str(titanicdf)

#checking the numeric features - Isolating the numeric features
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
titanicdfrotationf <- titanicdf[,!numeric]
titanicdfrotationf <- cbind(titanicdfrotationf, numericnormal)

library(dummies)
#removing dependent variable

trorationf <- dummy.data.frame(titanicdfrotationf[,-1])
summary(trorationf)
str(trorationf)

#Isolating the dependent variable
Survived <- titanicdfrotationf$Survived
#trorationf$survived <-titanicdfrotationf$Survived

#Holdout
set.seed(1337)
index <- sample(1:dim(titanicdf)[1], dim(titanicdf)[1]*.75, replace = FALSE)
trainingrotationf <- trorationf[index,]
testingrotationf <- trorationf[-index,]
Survivedtrain <- Survived[index]
Survivedtest <- Survived[-index]

#Running the model - The Rotation Forest only accept binary Rotation Forest
Survived <- factor(Survived, levels = c(0,1), labels = c(0,1))
Survived <- as.numeric(Survived)
library(rotationForest)
str(trainingrotationf)
trainingrotationf$
sapply(trainingrotation, function(x) sum(is.na(x)))
rotationf_model <- rotationForest(trainingrotationf,Survivedtrain)
dim(trainingrotationf)[1]
length(Survivedtrain)
str(trainingrotationf)
