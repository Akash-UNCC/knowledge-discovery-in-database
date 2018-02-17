slam_url <- "https://cran.r-project.org/src/contrib/Archive/slam/slam_0.1-37.tar.gz"
#install_url(slam_url)

#install.packages("tm")
library(tm)

getwd()
#set working directory – fix path as needed!
setwd("C:/Users/hi5an/Downloads/KDD/project")


#Create Corpus
docs1 <- Corpus(DirSource("diabetes"))

#Transform to lower case
docs1 <- tm_map(docs1,content_transformer(tolower))
#remove potentiallyy problematic symbols
toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, "" , x))})
docs1 <- tm_map(docs1, toSpace, "-")
docs1 <- tm_map(docs1, toSpace, ":")
docs1 <- tm_map(docs1, toSpace, "'")
docs1 <- tm_map(docs1, toSpace, "•")
docs1 <- tm_map(docs1, toSpace, "•    ")
docs1 <- tm_map(docs1, toSpace, " -")
docs1 <- tm_map(docs1, toSpace, "\"")
docs1 <- tm_map(docs1, toSpace, "")
#remove punctuation
docs1 <- tm_map(docs1, removePunctuation)
#Strip digits
docs1 <- tm_map(docs1, removeNumbers)

docs1 <- tm_map(docs1, function(x) iconv(enc2utf8(x), sub = "byte"))
#remove stopwords
docs1 <- tm_map(docs1, removeWords, stopwords("english"))
#remove whitespace
docs1 <- tm_map(docs1, stripWhitespace)



install.packages("SnowballC")
library(SnowballC)
#truncate words to their base form.
#For example, “education”, “educate” and “educative” are stemmed to “educat.”:
docs1 <- tm_map(docs,stemDocument)

dtm <- DocumentTermMatrix(docs1)

inspect(dtm)

#We will downweight the terms that occur frequently accross the documents. 
#This is done by computing the tf-idf Statistics.
dtm <- weightTfIdf(dtm, normalize = TRUE)

m = as.matrix(dtm)


#Create Corpus
corpus <- Corpus(DirSource("diabetes"))
summary(corpus)

ndocs <- length(corpus)
# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- ndocs * 0.01
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * .5
dtm = DocumentTermMatrix(corpus,
                         control = list(
                           stopwords = TRUE, 
                           wordLengths=c(4, 15),
                           removePunctuation = T,
                           removeNumbers = T,
                           stemming = T,
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))

write.csv((as.matrix(dtm)), "test.csv")
#head(sort(as.matrix(dtm)[18,], decreasing = TRUE), n=15)
dtm.matrix = as.matrix(dtm)
wordcloud(colnames(dtm.matrix), dtm.matrix[21, ], max.words = 20)
inspect(dtm)

#Term frequency-invese document frequecy (tf-idf)
dtm <- weightTfIdf(dtm, normalize = TRUE)
dtm.matrix = as.matrix(dtm)
wordcloud(colnames(dtm.matrix), dtm.matrix[27, ], max.words = 50)
inspect(dtm)
write.csv((as.matrix(dtm)), "test.csv")



#calculating distance
m  <- as.matrix(dtm)
# # # m <- m[1:2, 1:3]



distMatrix <- dist(m, method="euclidean")
print(distMatrix)
distMatrix <- dist(m, method="cosine")
print(distMatrix)



groups <- hclust(distMatrix,method="ward.D")
plot(groups, cex=0.9, hang=-1)
rect.hclust(groups, k=8)




head(sort(as.matrix(dtm)[1,], decreasing = TRUE), n=15)
wordcloud(colnames(dtm.matrix), dtm.matrix[3, ], max.words = 50)

  
library(wordcloud)
wordcloud(colnames(m), m[28, ], max.words = 20, maxTermFreq=1)


f = findFreqTerms(dtm, lowfreq = 1)
tail(f)

freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)

wf <- data.frame(word=names(freq), freq=freq) 

p <- ggplot(subset(wf, freq<10), aes(x = reorder(word, -freq), y = freq)) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x=element_text(angle=45, hjust=1))


barplot(f[1:10,]$freq, las = 2, names.arg = f[1:10,]$word,
        col ="lightblue", main ="Most frequent words",
        ylab = "Word frequencies")

findAssocs(dtm, terms = "zinc", corlimit = 0.3)