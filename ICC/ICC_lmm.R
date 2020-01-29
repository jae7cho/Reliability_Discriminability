##-------------------------------------------------- ICC and variation within and between ratio with LMM
## Two-level 
## Created by Annachiara Korchmaros 10/24/2018, CMI
## Updated by Annachiara Korchmaros 03/15/2019, CMI

## JWC update xx/xx/2020 add arg for covs

# LMM function parcelwise 
LMM.model.parcel=function(conn_all){ 
  nconns<-dim(conn_all)[1]
  #(1)1 table with all the statistics
  stats <- data.frame(matrix(0, nconns, 5))
  colnames(stats) <- c('Value', 'Std.Error', 'DF', 't-value', 'p-value')
  stats_inter <- stats
  stats_age <- stats
  stats_gender <- stats
  stats_meanFD <- stats
  stats_gconn <- stats
  nxvar <- 5 # the num of x variables
  tTable0 <- data.frame(matrix(0, nxvar, 5), row.names = c('(Intercept)','age', 'genderM', 'meanFD', 'gconn'))
  colnames(tTable0) <- c('Value', 'Std.Error', 'DF', 't-value', 'p-value')
  tTable0[,'p-value'] <- 1
  out_icc <- data.frame(matrix(0, nconns, 5))
  colnames(out_icc) <- c('ICC', 'VarWithinSubj', 'VarBetweenSubj', 'varY', 'varPredictY')  
  #(2) run the model  
  for (n in 1:nconns){
    if(n%%500 == 0){print(sprintf("LMM %0.2f%% done",  n/nconns*100))}
    y <- unlist(conn_all[n,])
    #(2)2 run 1level LMM with lmer
    df <- data.frame(y, subIDs, age, gender, gconn, meanFD)
    #ctrl <- lmerControl(check.nobs.vs.nlev="ignore", check.nobs.vs.nRE="ignore")
    #browser()
    tryCatch({fm<- lmer(y ~ age + gender + gconn + meanFD + (1 | subIDs), data = df)#,control = ctrl)
    out_fm<-summary(fm)
    statistics.variances<-as.data.frame(out_fm$varcor) 
    sigma_b<-statistics.variances$vcov[1] #variance of fixed effect
    sigma_r<-statistics.variances$vcov[2] #variance of random effect
    icc <- sigma_b/(sigma_r+sigma_b)
    var_predict <- var(predict(fm))
    tTable <- out_fm$coefficients
    out_icc[n,'ICC'] <- icc
    out_icc[n,'VarWithinSubj'] <- sigma_r
    out_icc[n,'VarBetweenSubj'] <- sigma_b
    out_icc[n, 'varY'] <- var(y)
    out_icc[n, 'varPredictY'] <- var_predict
    stats_inter[n,] <- tTable['(Intercept)',]
    stats_age[n,] <- tTable['age',]
    stats_gender[n,] <- tTable['genderM',]
    stats_meanFD[n,] <- tTable['meanFD',]
    stats_gconn[n,] <- tTable['gconn', ]
    }, error = function(e) {
      print(paste0("MY_ERROR: ",e,"at connection: ",n))
    },warning = function(w) {
      print(paste0("WARNING: ",w,"at connection: ",n))
    },message = function(m) {
      print(paste0("MESSAGE: ",m,"at connection: ",n))
    })
  }
  return(list("out_icc"=out_icc,"stats_inter"= stats_inter,"stats_age"=stats_age,'stats_gender'=stats_gender,"stats_meanFD"=stats_meanFD,"stats_gconn"=stats_gconn))
} 

