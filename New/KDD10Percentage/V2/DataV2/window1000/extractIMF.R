library(EMD)
library(data.table)
library(dplyr)
library(RcppCNPy)
# Clear the workspace
rm( list = ls() )

data<-read.csv("../../../Data/kddcup.data_10_percent_corrected", stringsAsFactors = FALSE)

# Process the data
colnames <- read.table("../../../Data/kddcup.names", skip = 1, sep = ":")
names(data) <- colnames$V1
d <- dim(data)
names(data)[d[2]] <- "label"

#get 4 time-series data
x_prep=select(data, count, srv_count, dst_host_count, dst_host_srv_count)
x_prep=as.matrix(x_prep)

window = as.integer(nrow(x_prep)/1000)
window=window - 1
x=matrix(0L, nrow=4*2, nrow(x_prep))

for (j in 1:ncol(x_prep)){
  i=0
  print(j)
  while ( i<window){
    
    s=x_prep[((i*1000)+1):(1000*(i+1)),j]
    #print("x1")
    t=seq(1000*i*2,(1000*(i+1)*2)-2,2)
    #print("x2")
    res <- emd(xt=s, 
               tt=t, max.imf = 1)
    #print("x3")
    mat <- cbind(res$imf,res$residue)
    #x[((j*2) -1):((j*2)),((i*100)+1):((i*100)+100)] = t(res$imf)
    x[((j*2) -1):((j*2)),((i*1000)+1):((i*1000)+1000)] = t(mat)
    #print("x4")
    i=i+1
    #print("x5")
  }
}

X1 <-x
x <-t(x)
newLen=i*1000
x=x[1:newLen,]

npySave("xData.npy", x)

