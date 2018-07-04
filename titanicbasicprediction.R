
setwd("/home/kenelly/workspaces/titanic/")

titanicdf <- read.csv("train.csv", header = T, na.strings = c(""), stringsAsFactors = T)

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

perishmodel <- rep("No", length(survived))

#Coin model - predict randomly between Yes or No



