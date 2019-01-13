library(EMD)
library(data.table)
library(dplyr)
library(RcppCNPy)
# Clear the workspace
rm( list = ls() )

data <- npyLoad("xData.npy")
t=seq(1,(2*nrow(data))-1,2)

t=seq(1,8,2)
IMF=data[1:nrow(data),t]


test1 <- hilbertspec(IMF)
#spectrogram(test1$amplitude[1:100,1], test1$instantfreq[1:100,1])

npySave("hilbertSpec.npy", test1$amplitude)

