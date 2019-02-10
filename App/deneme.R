library(EMD)
library(data.table)
library(dplyr)
library(RcppCNPy)
# Clear the workspace
rm( list = ls() )

#data<-read.csv("../New/KDD10Percentage/Data/corrected", stringsAsFactors = FALSE)
data<-read.csv("../New/KDD10Percentage/Data/corrected", stringsAsFactors = FALSE)
# Process the data
colnames <- read.table("../New/KDD10Percentage/Data/kddcup.names", skip = 1, sep = ":")
names(data) <- colnames$V1
d <- dim(data)
names(data)[d[2]] <- "label"

#get 4 time-series data
x_prep=select(data, count, srv_count, dst_host_count, dst_host_srv_count)
x_prep=as.matrix(x_prep)

x_prep[1:10,1:4]

t=seq(0,(100*2)-2,2)

i=3
res <- emd(xt=x_prep[((i-1)*100+1):(i*100),1:4], tt=t, max.imf = 1) 

tryCatch( { res <- emd(xt=x_prep[1:10,1:4], tt=t, max.imf = 1) }
          , error = function(e) {an.error.occured <<- TRUE})

i=0
j=1
s=x_prep[((i*10)+1):(10*(i+1)),j]
#x=x_prep[((i*10)+1):(10*(i+1)),1:4]
#print("x1")
t=seq(10*i*2,(10*(i+1)*2)-2,2)
#print("x2")
res <- emd(xt=s, 
           tt=t, max.imf = 1)

if (length(res$imf)==0){
  print("s")
}
x=matrix(0L, nrow=10, 4)
x=res$imf
##############################


library(EMD)
rm( list = ls() )



x=matrix(0L, nrow=10, 4)
myArgs <- commandArgs(trailingOnly = TRUE)

# Convert to numerics
nums = as.numeric(myArgs)



for( j in 1:ncol(x)){
  x[1:nrow(x),j]=nums[(10*(j-1)+1):(10*j)]
}

t=seq(0,(10*2)-2,2)
an.error.occured <- FALSE
#res <- emd(xt=x, 
#          tt=t, max.imf = 1)
res = matrix(0L, nrow=10, 8)
res[1:10,5:8]=x
tryCatch( { res <- emd(xt=x, tt=t, max.imf = 1) }
          , error = function(e) {an.error.occured <<- TRUE})

#if (length(res$imf) == 0){
#  mat1=matrix(0L, nrow=10, 4)
#}
#mat <- cbind(mat1,res$residue)

write.csv(res,file="emd.csv",row.names=FALSE) # drops the rownames

# cat will write the result to the stdout stream
#cat(x,sep="\n")


