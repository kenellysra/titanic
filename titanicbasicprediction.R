
setwd("/home/kenelly/workspaces/titanic/")

titanicdf <- read.csv("train.csv", header = T, na.strings = c(""), stringsAsFactors = T)
str(titanicdf)
#isolating and exploring the dependent variable
survived <- titanicdf$Survived

survived <-factor(survived, levels = c(0,1), labels = c("No", "Yes"))

#% of people who survived or not
prop.table(table(survived))

barplot(table(survived), main = "Distribution of Titanic Survival", ylab = "Frequency")

set.seed(1337)
index <- sample(1:length(survived), length(survived)*.25, replace = FALSE)
testing <-survived[index]

#Perish model - As the majority is "no", the perish model will predict all the values as "no"

perishmodel <- rep("No", length(testing))

#Coin model - predict randomly between Yes or No

coinmodel <- round(runif(length(testing), min = 0, max = 1))
coinmodel <- factor(coinmodel, levels = c(0, 1), labels = c("No", "Yes"))
length(perishmodel)
length(testing)
perishmodel <- factor(perishmodel, levels = c("No", "Yes"), labels = c("No", "Yes"))

table(testing, perishmodel)
table(testing, coinmodel)

#cheking accurancy
coinaccurancy <- 1 - mean(coinmodel!=testing)
coinaccurancy
perishaccurancy <- 1 - mean(perishmodel != testing)
perishaccurancy


prop.table(table(testing))


perish <- c()
coin <- c()

#Evaluating 1000 models from 1000 samples

for (i in 1:1000){
    index <- sample(1:length(survived), length(survived)*0.25, replace = FALSE)
    testing <- survived[index]
    
    coinmodel <- round(runif(length(testing), min = 0, max = 1))
    coinmodel <- factor(coinmodel, levels = c(0,1), labels = c("No", "Yes"))
    
    coin[i] <- 1 - mean(coinmodel!=testing)
    perish[i] <- 1-mean(perishmodel!=testing)
}

results <- data.frame(coin, perish)
names(results) <- c("Coin Accuracy", "Perishes Accurancy")
summary(results)

library(ggplot2)
library(reshape)

ggplot(melt(results), mapping = aes(fill = variable, x = value)) + geom_density(alpha = .5)
boxplot(results)


#Model considering gender as independent variable
gendermodel <- titanicdf[,c("Sex", "Survived")]
gendermodel$Survived <- factor(gendermodel$Survived, levels = c(0, 1), labels = c("No", "Yes"))

index <- sample(1:length(gendermodel$Survived), length(gendermodel$Survived)*.75, replace = FALSE)
trainingdata <-gendermodel[index,]
testingdata <-gendermodel[-index,]
dim(gendermodel)[1]

table(trainingdata$Survived, trainigdata$Sex)

predictsurvival <- function(data){
    model <- rep("No", length(data))
    model[data$Sex == 'female'] <- "Yes"
    model[data$Sex == 'male'] <- "No"
    return(model)
}

women <- c()

for (i in 1:1000) {
    index <- sample(1:length(gendermodel), length(gendermodel)*.75, replace = FALSE)
    testing <- gendermodel[-index,]
    womenmodel <- predictsurvival(testing)
    women[i] <- 1 - mean(womenmodel !=testingdata$Survived)
    
}

women
results$WomenAccurancy <- women
names(results)<- c("Coin", "All Perish", "Women")
boxplot(results)
