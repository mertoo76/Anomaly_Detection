library(EMD)
library(data.table)
library(dplyr)
library(RcppCNPy)
# Clear the workspace
rm( list = ls() )

#data<-read.csv("../New/KDD10Percentage/Data/kddcup.data_10_percent_corrected", stringsAsFactors = FALSE)
data<-read.csv("../New/KDD10Percentage/Data/corrected", stringsAsFactors = FALSE)

# Process the data
colnames <- read.table("../New/KDD10Percentage/Data/kddcup.names", skip = 1, sep = ":")
names(data) <- colnames$V1
d <- dim(data)
names(data)[d[2]] <- "label"

#get 4 time-series data
x_prep=select(data, count, srv_count, dst_host_count, dst_host_srv_count)
x_prep=as.matrix(x_prep)

window = as.integer(nrow(x_prep)/10)
window=window - 1
x=matrix(0L, nrow=4*2, nrow(x_prep))

for (j in 1:ncol(x_prep)){
  i=0
  print(j)
  while ( i<window){
    
    s=x_prep[((i*10)+1):(10*(i+1)),j]
    #print("x1")
    t=seq(10*i*2,(10*(i+1)*2)-2,2)
    tryCatch( { res <- emd(xt=s, 
                           tt=t, max.imf = 1)
    
    if (length(res$imf)==0){
      res$imf=rep(0, 10)
    }
    
    #print("x3")
    mat <- cbind(res$imf,res$residue)
    x[((j*2) -1):((j*2)),((i*10)+1):((i*10)+10)] = t(mat)
    
    }
    , error = function(e,i) {
    mat <- cbind(rep(0, 10),s)
    x[((j*2) -1):((j*2)),((i*10)+1):((i*10)+10)] = t(mat)

    }  )
    
    i=i+1
 
  }
}
X1 <-x
x <-t(x)
newLen=i*10
x=x[1:newLen,]

npySave("xDataTest.npy", x)

