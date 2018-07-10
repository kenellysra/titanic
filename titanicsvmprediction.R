#SVM
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

titanicdf$Age <- scale(titanicdf$Age)
titanicdf$SibSp <- scale(titanicdf$SibSp)
titanicdf$Parch <- scale(titanicdf$Parch)
titanicdf$Fare <- scale(titanicdf$Fare)

#HoldOut
set.seed(1337)
index <- sample(1:dim(titanicdf)[1], dim(titanicdf)[1]*.75, replace = FALSE)
trainingsvm <- titanicdf[index,] 
testingsvm <- titanicdf[-index,]
str(trainingsvm)

#Function to assess SVM Performance
svmPerformance <- function(svm.model, training, testing) {
    p3 <- predict(svm.model, training, type = "response")
    p3 <- mean(p3 != training$Survived)
    print(paste('Accuracy on training data ',1-p3))
    
    ksvmPredition <- predict(svm.model,testing, type = "response")
    ksvmMissclassificationRate <- mean(ksvmPredition != testing$Survived)
    print(paste('Accuracy on testing data ',1-ksvmMissclassificationRate))
}
#Running SVM model
library(kernlab)
svm.model.1 <- ksvm(Survived ~ ., data = trainingsvm)
svmPerformance(svm.model.1, trainingsvm, testingsvm)

#some different kernels
svm.model.2 <- ksvm(Survived ~ ., data = trainingsvm, kernel="rbfdot")
svmPerformance(svm.model.2, trainingsvm, testingsvm)

svm.model.3 <- ksvm(Survived ~ ., data = trainingsvm, kernel="vanilladot")
svmPerformance(svm.model.3, trainingsvm, testingsvm)

svm.model.4 <- ksvm(Survived ~ ., data = trainingsvm, kernel="polydot")
svmPerformance(svm.model.4, trainingsvm, testingsvm)

svm.model.5 <- ksvm(Survived ~ ., data = trainingsvm, kernel="laplacedot")
svmPerformance(svm.model.5, trainingsvm, testingsvm)

svm.model.6 <- ksvm(Survived ~ ., data = trainingsvm, kernel="besseldot")
svmPerformance(svm.model.6, trainingsvm, testingsvm)

#swtich SVM type
svm.model.7  <- ksvm(Survived ~ ., data = trainingsvm, kernel="laplacedot", type="C-svc")
svmPerformance(svm.model.7, trainingsvm, testingsvm)

svm.model.8  <- ksvm(Survived ~ ., data = trainingsvm, kernel="laplacedot", type="C-bsvc")
svmPerformance(svm.model.8, trainingsvm, testingsvm)

svm.model.4 <- ksvm(Survived ~ ., data = trainingsvm, kernel="polydot", kpar=list(degree=3))
svmPerformance(svm.model.4, trainingsvm, testingsvm)

svm.model.4 <- ksvm(Survived ~ ., data = trainingsvm, kernel="polydot", kpar=list(degree=4))
svmPerformance(svm.model.4, trainingsvm, testingsvm)

svm.model.4 <- ksvm(Survived ~ ., data = trainingsvm, kernel="polydot", kpar=list(degree=5))
svmPerformance(svm.model.4, trainingsvm, testingsvm)

svm.model.4 <- ksvm(Survived ~ ., data = trainingsvm, kernel="polydot", kpar=list(degree=10))
svmPerformance(svm.model.4, trainingsvm, testingsvm)

svm.model.4 <- ksvm(Survived ~ ., data = trainingsvm, kernel="polydot", kpar=list(degree=3), cross=10)
svmPerformance(svm.model.4, trainingsvm, testingsvm)

#Other SVM implementation
library(e1071)

svm.model.9 <- svm(Survived ~., data=trainingsvm)
svmPerformance(svm.model.9, trainingsvm, testingsvm)

#Tunning
tuned.svm = tune(svm, Survived ~ ., data = train, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary(tuned.svm)

svmPerformance(tuned.svm$best.model, train, test)

#Other Evaluations
library(caret)
svm.model.9 <- svm(Survived ~., data=trainingsvm)
svm_prediction <- predict(svm.model.9, testingsvm, type = "class")
confusionMatrix(svm_prediction, testingsvm$Survived, positive = "Yes")
library(MLmetrics)
F1_Score(y_true =  as.numeric(testingsvm$Survived), 
         y_pred =  as.numeric(svm_prediction), positive = "1")
