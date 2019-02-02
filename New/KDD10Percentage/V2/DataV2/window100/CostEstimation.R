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

window = as.integer(nrow(x_prep)/100)
window=window - 1
x=matrix(0L, nrow=4*2, nrow(x_prep))

cpuTimeList <- matrix(0L, nrow=5, 4)
t=seq(0,(100*2)-2,2)
for(j in 1:ncol(x_prep)){
  ptm <- proc.time()
  res <- emd(xt=x_prep[201:300,j], 
             tt=t, max.imf = 1)
  xx=proc.time() - ptm
  cpuTimeList[1:5,j] = xx
}

cpuTimeList = t(cpuTimeList)

colnames(cpuTimeList) <- c("User CPU Time","System CPU Time", "Elapsed","v1","v2")


