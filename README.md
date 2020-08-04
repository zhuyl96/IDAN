## Introuction:
**The source code and data for the paper entitled "Interactive Dual Attention Network for Text Sentiment Classification".**


## Requirements:
keras 2.3.1，tensorflow 1.14.0，pandas 0.24.2，numpy 1.16.2，sklearn 0.20.3, nltk 3.4


## Directory structure:
+ config 	|	 *Experimental parameter configuration*
+ data	|	 *Experimental data*
	+ npy data 	|	 *Word embedding data of each dataset after BERT processing*
	+ raw data 	|	 *Raw dataset*
+ model 	|	 *Code of the experimental model*
+ preprocessing 	|	 *Preprocessing code*
+ resources 	|	 *External resources for the experiment*
	+ lexicon 	|	 *Lexicon resources*
	+ pre-trained model 	|	 *Pre-trained word embedding model (BERT and Word2Vec)*
+ visualization 	|	 *Trained model and code to visualize the attention weight*


## Resources:

### Datasets
**Chinese Datasets:** [ChnSentiCorp](https://www.aitechclub.com/data-detail?data_id=29), [NLPCC-CN](http://tcci.ccf.org.cn/conference/2014/pages/page04_sam.html)

**English Datasets:** [NLPCC-EN](http://tcci.ccf.org.cn/conference/2014/pages/page04_sam.html), [MR](https://www.cs.cornell.edu/people/pabo/movie-review-data/)

*ChnSentiCorp* is a collection of hotel review data collected and compiled by Professor Songbo Tan.

*NLPCC-CN* and *NLPCC-EN* are the Chinese and English datasets of the sentiment analysis task of the 2014 NLPCC International Conference.

*MR* is a movie review dataset from the IMDB website.

### Lexicon resources:
Sentiment, intensity, and negative lexicon resources include [English lexicon resources](http://www.keenage.com/html/c_index.html) published by HowNet, and [Chinese lexicon resources](https://kexue.fm/archives/3360) organized by Jianlin Su.

### Pre-trained model:
**BERT**: [BERT-base](https://github.com/google-research/bert) for English, [BERT-wwm]( https://github.com/ymcui/Chinese-BERT-wwm) for Chinese.

**Word2Vec**: [English](https://code.google.com/archive/p/word2vec/), [Chinese](https://github.com/Embedding/Chinese-Word-Vectors)

## Data preprocess:
Execute *chnsenticorp.py*, *nlpcc.py*, *nlpcc_en.py*, and *mr.py*, respectively.

*Note: each preprocess program corresponds to a dataset, which only needs to be executed once.*

## Usage:
Execute different program according to the model name in the "**model**" directory.

*Note: modify the "parser" field in the "main" function can specify the corresponding dataset.*