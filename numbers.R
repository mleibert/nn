library(readr)
mnist_raw <- read_csv("https://pjreddie.com/media/files/mnist_train.csv",
	col_names = FALSE)

mnist<-as.data.frame(mnist_raw )
rm(mnist_raw)



#eta will be the alphas, the number of x's * m (number of nodes in layer 1) 
#plus m (the intercepts), and the betas, the number of classifications of y 
# (table(dat$y)) * m plus the number of classifications


#alpha = 784*4+4

NN<-function(x,eta,m){
  	
  X<-as.matrix(x)
  
  Alpha<-matrix(eta[1:prod( c(dim(X)[2] + 1, m ) )], dim(X)[2] + 1 , m)
  Z<-apply(Alpha, 2, function(w) 1/(1+exp(- w[1] -   X %*% w[-1]  )) ) 
  
  Beta<-matrix( eta[(prod(dim(Alpha))+1):length(eta) ], dim(Z)[2]+1, 10)
  
  TT<-apply(Beta, 2, function(w) 1/(1+exp(- w[1] - Z %*% w[-1]  )) ) 
  Y<- apply(TT,2, function(x) exp(x))
  Y<-Y/rowSums(Y)
  return(Y)
}

#this loss function is meaningless in this context, a multinomial loglik 
#could work
logL<-function(x,y,eta,m){
	Y<-NN(x,eta,m)
	return(- sum( (1-y)*log(Y[,1]) + ( y)*log(Y[,2])  ) )	}

system.time(
for(U in 1:3200){ logL(sdf[,-1],sdf[,1],eta,4) }
)

gradL<-function(DF=mnist,eta,m ){
  
  sdf<-mnist[sample(nrow(mnist), 1000), ]
  sx<-sdf[,-1];sy<-sdf[,1]
  dimeta<-m*(dim(sdf)[2]+1)+2*(1+m)
  gradf<-rep(NA,dimeta)
  
  Y<-NN(sdf[,-1],eta,m)
  ETA<-eta
  
  logl<-logL(sx,sy,eta,m)
  for( i in 1:dimeta){
    ETA <-eta
    ETA[i]<-eta[i]+(10^-6)
    
    gradf[i]<- ( logL(sx,sy,ETA,m) - logl  )	}
	gradf[i]<-gradf[i] / (10^-6)
  return(gradf)
}

gradL(mnist,eta,4)




