#C5.0

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


#creating a holdout
set.seed(1337)
index <- sample(1:dim(titanicdf)[1], dim(titanicdf)[1]*0.75, replace = FALSE)
traininngc50 <- titanicdf[index,]
testingc50 <- titanicdf[-index,]

#Running C5.0 | trials = 10 means the boosting
library(C50)
c50model <- C5.0(Survived ~., data = traininngc50, trials = 10)
c50model_predict <- predict(c50model, testingc50, type = "class")

#other way to run C50 - Winnowing - selection process
c50winnow <- C5.0(Survived ~., data = traininngc50, control = C5.0Control(winnow = TRUE))
c50winnow_predict <- predict(c50winnow, testingc50, type = "class")

#other way to run C5.0

library(caret)
titanicdfcontrol <- titanicdf[!is.na(titanicdf$Child),]
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
train(Survived~., data = titanicdfcontrol, method = "C5.0", metric = "Kappa", trControl = control)


sapply(titanicdf, function(x) sum(is.na(x)))
##Assessing the model
library(caret)
confusionMatrix(c50model_predict, testingc50$Survived, positive = "Yes")
confusionMatrix(c50winnow_predict, testingc50$Survived, positive = "Yes")

library(MLmetrics)
F1_Score(y_true =  as.numeric(testingc50$Survived), 
         y_pred =  as.numeric(c50model_predict), positive = "1")

F1_Score(y_true =  as.numeric(testingc50$Survived), 
         y_pred =  as.numeric(c50winnow_predict), positive = "1")

