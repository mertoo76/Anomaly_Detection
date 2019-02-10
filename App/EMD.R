# Clear the workspace
library(EMD)
rm( list = ls() )



x=matrix(0L, nrow=10, 4)
mat<<-matrix(0L, nrow=10, 8)
myArgs <- commandArgs(trailingOnly = TRUE)

# Convert to numerics
nums = as.numeric(myArgs)



for( j in 1:ncol(x)){
  x[1:nrow(x),j]=nums[(10*(j-1)+1):(10*j)]
}
print("R:")
print(x)
t=seq(0,(10*2)-2,2)

an.error.occured <<- FALSE

for(i in  1:ncol(x)){
  tryCatch( { res <- emd(xt=x[1:10,i], tt=t, max.imf = 1)
  
          if (length(res$imf)==0){
              res$imf=rep(0, 10)
            }
              mat[1:10,(i*2)]=res$residue
              mat[1:10,((i*2)-1)]=res$imf
  }
            , error = function(e,i) {mat[1:10,(i*2)]=x[1:10,i]
            mat[1:10,((i*2)-1)]=rep(0, 10)
            })
}

#res <- emd(xt=x, 
#          tt=t, max.imf = 1)




#if (length(res$imf) == 0){
#  mat1=matrix(0L, nrow=10, 4)
#}
#mat <- cbind(mat1,res$residue)

write.csv(mat,file="emd.csv",row.names=FALSE) # drops the rownames

# cat will write the result to the stdout stream
#cat(x,sep="\n")