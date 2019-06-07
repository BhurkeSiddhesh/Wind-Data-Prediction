rm(list = ls())

library("rWind")
library("shape")
library(caret)
library(randomForest)
library("tidyverse")
library(ggplot2)
library(qdapTools)
library(devtools)
library(C50)
library(corrplot)
library(e1071)
library(ROSE)
library(DMwR)
library(fastAdaboost)
library(adabag)

wind.data<- read.csv("African Data.csv",header = T)
wind.data<- wind.data[,-c(1,2,3,4,5,6,8,9,11,12,13,14,15,19)]
wind.data<- wind.data[complete.cases(wind.data),]

direction <- function(x){ifelse(x >=0  &  x < 11.25, "N", 
                                ifelse(x>=11.25   &  x < 33.75, "NNE",
                                       ifelse(x>=33.75   &  x < 56.25, "NE",
                                              ifelse(x>=56.25   &  x < 78.75, "ENE",
                                                     ifelse(x>=78.75   &  x < 101.25,"E",
                                                            ifelse(x>=101.25  &  x < 123.75,"ESE",
                                                                   ifelse(x>=123.75  &  x < 146.25,"SE",
                                                                          ifelse(x>=146.25  &  x < 168.75,"SSE",
                                                                                 ifelse(x>=168.75  &  x < 191.25,"S",
                                                                                        ifelse(x>=191.25  &  x < 213.75,"SSW",
                                                                                               ifelse(x>=213.75  &  x < 236.25,"SW",
                                                                                                      ifelse(x>=236.25  &  x < 258.75,"WSW",
                                                                                                             ifelse(x>=258.75  &  x < 281.25,"W",
                                                                                                                    ifelse(x>=281.25  &  x < 303.75,"WNW",
                                                                                                                           ifelse(x>=303.75  &  x < 326.25,"NW",
                                                                                                                                  ifelse(x>=326.25  &  x < 348.75,"NNW",
                                                                                                                                         ifelse(x>=348.75  &  x <361,"N",NA)))))))))))))))))}




wind.data.direction<- as.data.frame(direction(wind.data$wdct))
colnames(wind.data.direction)<- "Wind.Direction"
finaldata<- cbind(wind.data,wind.data.direction)
table(finaldata$Wind.Direction)
finaldata<- finaldata[,-c(2,3,4,5,8,9,10,11,13,14)] #,15,17
cor<-cor(finaldata[,2:7])
corrplot(cor)
summary(lm(data = finaldata[,2:7]))
finaldata1<- finaldata[,-c(5,7)]

RMSE <- function(m,o){
  sqrt(mean((m - o)^2))
}

MSE <- function(m,o){
  (mean((m - o)^2))
}

#sample size
smp_size <- floor(0.80 * nrow(finaldata1))

## set the seed to make your partition reproducible
set.seed(54)
train_smp <- sample(seq_len(nrow(finaldata1)), size = smp_size)

train <- finaldata1[train_smp, ]
test <- finaldata1[-train_smp, ]

#C50
model1 <- C5.0(Wind.Direction~.,train[,-5])
summary(model1)
preds1 <- predict(model1,test[,-c(5,6)])
accu1 <- table(preds1,test$Wind.Direction)
accu1 <- as.data.frame(accu1)
Acc_C50 <- (accu1[1,3]+accu1[4,3])/sum(accu1[,3]) *100

confusionMatrix(preds1,test$Wind.Direction)


predict1 <- as.data.frame(preds1)
test_new1 <- data.frame(predict1,test$Wind.Direction)
colnames(test_new1) <- c("Predicted","Observed")
test_new1 <- as.data.frame((sapply(test_new1[,1:2], as.numeric)))
r1 <- RMSE(test_new1$Predicted,test_new1$Observed)
m1 <- MSE(test_new1$Predicted,test_new1$Observed)

#Naive Bayes
model2 <- naiveBayes(train$Wind.Direction~.,train)
summary(model2)
preds2 <- predict(model2,test[,-6])
accu2 <- table(preds2,test$Wind.Direction)
accu2 <- as.data.frame(accu2)
Acc_NB <- (accu2[1,3]+accu2[4,3])/sum(accu2[,3]) *100

preds2_1 <- predict(model2,train[,-6])
accu2_1 <- table(preds2_1,train$Wind.Direction)
accu2_1 <- as.data.frame(accu2_1)
Acc_NB1 <- (accu2_1[1,3]+accu2_1[4,3])/sum(accu2_1[,3]) *100

confusionMatrix(preds2,test$Wind.Direction)
confusionMatrix(preds2_1,train$Wind.Direction)

predict2 <- as.data.frame(preds2)
test_new2 <- data.frame(predict2,test$Wind.Direction)
colnames(test_new2) <- c("Predicted","Observed")
test_new2 <- as.data.frame(sapply(test_new2[,1:2], as.numeric))
r2 <- RMSE(test_new2$Predicted,test_new2$Observed)
m2 <- MSE(test_new2$Predicted,test_new2$Observed)

#logistic regression model
model3 <- glm (formula = Wind.Direction~.,data = train, family = "binomial")
summary(model3)
preds3 <- as.numeric(predict(model3,test[,-6]))
accu3 <- table(preds3,test$Wind.Direction)
accu3 <- as.data.frame(accu3)
Acc_LRM <- (accu3[1,3]+accu3[4,3])/sum(accu3[,3]) *100

confusionMatrix(preds3,as.numeric(test$Wind.Direction))

predict3 <- as.data.frame(preds3)
test_new3 <- data.frame(predict3,test$Wind.Direction)
colnames(test_new3) <- c("Predicted","Observed")
test_new3 <- as.data.frame(sapply(test_new3[,1:2], as.numeric))
r3 <- RMSE(test_new3$Predicted,test_new3$Observed)
m3 <- MSE(test_new3$Predicted,test_new3$Observed)

#J48
model4 <- J48(train$Wind.Direction~.,data = train[,-5])
summary(model4)
preds4 <- predict(model4,test[,-c(5,6)],type = "class")
accu4 <- table(preds4,test$Wind.Direction)
accu4 <- as.data.frame(accu4)
Acc_J48 <- (accu4[1,3]+accu4[4,3])/sum(accu4[,3]) *100
confusionMatrix(preds4,test$Wind.Direction)

predict4 <- as.data.frame(preds4)
test_new4 <- data.frame(predict4,test$Wind.Direction)
colnames(test_new4) <- c("Predicted","Observed")
test_new4 <- as.data.frame(sapply(test_new4[,1:2], as.numeric))
r4 <- RMSE(test_new4$Predicted,test_new4$Observed)
m4 <- MSE(test_new4$Predicted,test_new4$Observed)

#RPART
model5 <- rpart(train$Wind.Direction~.,data = train)
summary(model5)
preds5 <- predict(model5,test[,-6],type = "class")
accu5 <- table(preds5,test$Wind.Direction)
accu5 <- as.data.frame(accu5)
Acc_R <- (accu5[1,3]+accu5[4,3])/sum(accu5[,3]) *100

confusionMatrix(preds5,test$Wind.Direction)

predict5 <- as.data.frame(preds5)
test_new5 <- data.frame(predict5,test$Wind.Direction)
colnames(test_new5) <- c("Predicted","Observed")
test_new5 <- as.data.frame(sapply(test_new5[,1:2], as.numeric))
r5 <- RMSE(test_new5$Predicted,test_new5$Observed)
m5 <- MSE(test_new5$Predicted,test_new5$Observed)

##Application of openair package for wind.data

finaldata2<- wind.data[,-c(1,3,4,5,6,7,8,9,10,11,12,13,14,17,19)]
finaldata2<- cbind(finaldata2,wind.data.direction)
mydata <- finaldata2

Dates_10m <- as.character(mydata$date)

Wind <- mydata[,2] 

AD <- mydata$wdct
Dir <- ifelse(AD<0,360+AD,ifelse(AD>360,AD-360,AD))

data <- cbind(Dates_10m,Wind,Dir)
data <- as.data.frame(data)


mydata <- data
colnames(mydata) <- c("date","ws","wd")

windRose(mydata, ws = "ws", wd = "wd", ws2 = NA, wd2 = NA, 
         ws.int = 2, angle = 22.5, type = "year", bias.corr = TRUE, cols
         = "default", grid.line = NULL, width = 1, seg = NULL, auto.text 
         = TRUE, breaks = 4, offset = 10, normalise = FALSE, max.freq = 
           NULL, paddle = TRUE, key.header = NULL, key.footer = "(m/s)", 
         key.position = "bottom", key = TRUE, dig.lab = 5, statistic = 
           "prop.count", pollutant = NULL, annotate = TRUE, angle.scale = 
           315, border = NA)


##Afonso City Data Analysis

Afonso<- subset(wind.data, wind.data$city == "Afonso")
Afonso1<- Afonso[complete.cases(Afonso),]
direction <- function(x){ifelse(x >=0  &  x < 11.25, "N", 
                                ifelse(x>=11.25   &  x < 33.75, "NNE",
                                       ifelse(x>=33.75   &  x < 56.25, "NE",
                                              ifelse(x>=56.25   &  x < 78.75, "ENE",
                                                     ifelse(x>=78.75   &  x < 101.25,"E",
                                                            ifelse(x>=101.25  &  x < 123.75,"ESE",
                                                                   ifelse(x>=123.75  &  x < 146.25,"SE",
                                                                          ifelse(x>=146.25  &  x < 168.75,"SSE",
                                                                                 ifelse(x>=168.75  &  x < 191.25,"S",
                                                                                        ifelse(x>=191.25  &  x < 213.75,"SSW",
                                                                                               ifelse(x>=213.75  &  x < 236.25,"SW",
                                                                                                      ifelse(x>=236.25  &  x < 258.75,"WSW",
                                                                                                             ifelse(x>=258.75  &  x < 281.25,"W",
                                                                                                                    ifelse(x>=281.25  &  x < 303.75,"WNW",
                                                                                                                           ifelse(x>=303.75  &  x < 326.25,"NW",
                                                                                                                                  ifelse(x>=326.25  &  x < 348.75,"NNW",
                                                                                                                                         ifelse(x>=348.75  &  x <361,"N",NA)))))))))))))))))}




RMSE <- function(m,o){
  sqrt(mean((m - o)^2))
}

MSE <- function(m,o){
  (mean((m - o)^2))
}
Afonso.direction<- as.data.frame(direction(Afonso$wdct))
colnames(Afonso.direction)<- "Wind.Direction"
Afonsodata<- cbind(Afonso,Afonso.direction)
Afonsotable<-as.data.frame(table(Afonsodata$Wind.Direction))
colnames(Afonsotable)<- c("Direction","Count")
plot(Afonsotable$Count,Afonsotable$Direction)
#normalized = (Afonsotable$Count-min(Afonsotable$Count))/(max(Afonsotable$Count)-min(Afonsotable$Count))

#NumericDir<-as.numeric(Afonsotable$Direction)
#Afonsotable1<-cbind(NumericDir,normalized)
#Afonsotable1<- as.data.frame(Afonsotable1)
#Afonsotable<- Afonsotable[1:5,]


Afonsodata<- Afonsodata[,-c(2,3,4,5,8,9,10,11,13,14)]
Afonsodata[Afonsodata==0] <- NA
Afonsodata
Afonsodata<- Afonsodata[complete.cases(Afonsodata),]
Afonsodata<- Afonsodata[,-c(5,7)]
#plot(Afonsotable$Direction,Afonsotable$Count)

#sample size
Afonso_size <- floor(0.81 * nrow(Afonsodata))

## set the seed to make your partition reproducible
set.seed(72)
train_Afonso <- sample(seq_len(nrow(Afonsodata)), size = Afonso_size)

Aftrain <- Afonsodata[train_Afonso, ]
Aftest <- Afonsodata[-train_Afonso, ]


#Random Forest
modelaf <- randomForest(Aftrain$Wind.Direction~.,data = Aftrain, ntree = 100)
summary(modelaf)
predsaf <- predict(modelaf,Aftest[,-6])
accuAf <- table(predsaf,Aftest$Wind.Direction)
accuAf<- as.data.frame(accuAf)
Acc_Af <- (accuAf[1,3]+accuAf[4,3])/sum(accuAf[,3]) *100

confusionMatrix(predsaf,Aftest$Wind.Direction)

predsaf1 <- predict(modelaf,Aftrain[,-6])
accuAf1 <- table(predsaf1,Aftrain$Wind.Direction)
accuAf1<- as.data.frame(accuAf1)
Acc_Af1 <- (accuAf1[1,3]+accuAf1[4,3])/sum(accuAf1[,3]) *100

confusionMatrix(predsaf1,Aftrain$Wind.Direction)


predict2 <- as.data.frame(predsaf)
test_new_af <- data.frame(predict2,Aftest$Wind.Direction)
colnames(test_new_af) <- c("Predicted","Observed")
test_new1_af <- as.data.frame((sapply(test_new_af[,1:2], as.numeric)))
x<-cbind(test_new_af,test_new1_af)
r <- RMSE(test_new1_af$Predicted,test_new1_af$Observed)
m <- MSE(test_new1_af$Predicted,test_new1_af$Observed)

#Naive Bayes
modelaf1 <- naiveBayes(Aftrain$Wind.Direction~.,Aftrain)
summary(modelaf1)
predsaf2 <- predict(modelaf1,Aftest[,-6])
accuaf2 <- table(predsaf2,Aftest$Wind.Direction)
accuaf2 <- as.data.frame(accuaf2)
Acc_NBaf <- (accuaf2[1,3]+accuaf2[4,3])/sum(accuaf2[,3]) *100
confusionMatrix(predsaf2,Aftest$Wind.Direction)

preds2_af1 <- predict(modelaf1,Aftrain[,-6])
accu2_af1 <- table(preds2_af1,Aftrain$Wind.Direction)
accu2_af1 <- as.data.frame(accu2_af1)
Acc_NB1af <- (accu2_af1[1,3]+accu2_af1[4,3])/sum(accu2_af1[,3]) *100
confusionMatrix(preds2_af1,Aftrain$Wind.Direction)

predict2af <- as.data.frame(predsaf2)
test_new2af <- data.frame(predict2af,Aftest$Wind.Direction)
colnames(test_new2af) <- c("Predicted","Observed")
test_new2af1 <- as.data.frame(sapply(test_new2af[,1:2], as.numeric))
r2af <- RMSE(test_new2af1$Predicted,test_new2af1$Observed)
m2af <- MSE(test_new2af1$Predicted,test_new2af1$Observed)


Afonso2<- Afonso[,-c(1,3,4,5,6,7,8,9,10,11,12,13,14,17,19)]
Afonso2<- cbind(Afonso2,Afonso.direction)
#Afonso2[Afonso2==0] <- NA
Afonso2
#Afonso2<- Afonso2[complete.cases(Afonso2),]
mydata1 <- Afonso2

Dates_10m1 <- as.character(mydata1$date)

Wind1 <- mydata1[,2] 

AD1 <- mydata1$wdct
Dir1 <- ifelse(AD1<0,360+AD1,ifelse(AD1>360,AD1-360,AD1))

data1 <- cbind(Dates_10m1,Wind1,Dir1)
data1 <- as.data.frame(data1)


mydata1 <- data1
colnames(mydata1) <- c("date","ws","wd")

windRose(mydata1, ws = "ws", wd = "wd", ws2 = NA, wd2 = NA, 
         ws.int = 2, angle = 22.5, type = "year", bias.corr = TRUE, cols
         = "default", grid.line = NULL, width = 1, seg = NULL, auto.text 
         = TRUE, breaks = 4, offset = 10, normalise = FALSE, max.freq = 
           NULL, paddle = TRUE, key.header = NULL, key.footer = "(m/s)", 
         key.position = "bottom", key = TRUE, dig.lab = 5, statistic = 
           "prop.count", pollutant = NULL, annotate = TRUE, angle.scale = 
           315, border = NA)