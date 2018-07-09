#Naive Bayes


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

#Data Preparation for Naive Bayes - All the features needs to be categorical.
str(titanicdf)

#Transformation of Age, Sibsp, Parch, Fare to categorical. Removing Famsize,
# because it has the same meaning as FamsizeCateg
titanicdf <-titanicdf[, -11]

#Cheking the distributions
table(titanicdf$SibSp)
titanicdf$SibSp <- as.factor(titanicdf$SibSp)

table(titanicdf$Parch)
titanicdf$Parch <- as.factor(titanicdf$Parch)

#There are to many levels to Age and Fare, then we need to divide it in ranges.
table(titanicdf$Agerange)
hist(titanicdf$Age)
hist(titanicdf$Fare)

titanicdf$Farerange <- cut(titanicdf$Fare, 
                           breaks = c(0, 10, 50, max(titanicdf$Fare)), 
                           labels = c("low", "middle", "high"))
table(titanicdf$Farerange)

titanicdf$Agerange <- cut(titanicdf$Age, 
                          breaks = c(0, 10, 20, 30, 40, 50, 60, 70, max(titanicdf$Age)), 
                          labels = c("0-10", "10-20", "20-30", "30-40", "40-50",
                                     "50-60", "60-70", "70+"))

#Removing the numeric features
str(titanicdf)
titanicdf$Age <-NULL
titanicdf$Fare <- NULL

#Creating Houldout 75-25
index <- sample(1:dim(titanicdf)[1], dim(titanicdf)[1]*.80, replace = FALSE)
trainingbayes <- titanicdf[index,]
testingbayes <- titanicdf[-index,]

#Training the model
library(e1071)
naivebayes <- naiveBayes(Survived ~., data = trainingbayes)
naivebayes_predict <- predict(naivebayes, testingbayes)
head(naivebayes)

#Assessing the model
library(caret)
confusionMatrix(naivebayes_predict, testingbayes$Survived, positive = "Yes")
library(MLmetrics)
F1_Score(y_true =  as.numeric(testingbayes$Survived), y_pred =  as.numeric(naivebayes_predict), positive = "1")
