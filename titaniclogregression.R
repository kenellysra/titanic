#Classification with Regression Logistic

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

#Creating Hold out samples train and test
set.seed(1337)
index <- sample(1:dim(titanicdf)[1], dim(titanicdf)[1]*.80, replace = FALSE)
train <- titanicdf[index,]
test <-titanicdf[-index,]

#Logistic Regression Model

logregmodel <- glm(Survived ~., data = train, family = binomial(link = logit) )
summary(logregmodel)

plot(logregmodel)

#Removing some features that does not seems relevant after check the p-value, and run the model again .
titanicdf$SibSp <- NULL
titanicdf$Parch <- NULL
titanicdf$Title <- NULL
titanicdf$Famsize <- NULL

anova(logregmodel, test = "Chisq")

logregpredict <- predict(logregmodel, newdata = test, type = 'response')
test$Survived <- as.factor(test$Survived)
logregresults <- ifelse(logregpredict > 0.5, "Yes", "No")
logregresults <- as.factor(logregresults)
#Performance Evaluation
library(gmodels)
library(caret)
library(e1071)
confusionMatrix(logregresults, test$Survived,positive = "Yes")

CrossTable(test$Survived, logregresults)
as.factor(logregresults)
