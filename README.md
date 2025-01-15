# Fake News Detection

**Alvin Tan**

<ins>tanalvin@usc.edu</ins>

### Introduction

In this project, we constructed a BERT (Bidirectional Encoder Representations from Transformers) model that distinguishes fake news articles from real ones. 

### Dataset

The dataset used in this project is the ISOT Fake News Dataset from kaggle. It contains around 20000 truthful articles and 20000 fake articles. The truthful articles were obtained mainly from Reuters.com. As for the fake articles, they were collected from different sources. The dataset is in the form of csv files. For each data point entry, the title and the text of the article is recorded. We noticed that in the dataset, every truthful article contains a “(Reuters)” tag in it. For data preprocessing, we removed the tag from every true article entry as it might skew the model. After that, we concatenated the title and the text of each article and tokenized each data point with a pre-trained BERT tokenizer. We also truncated the max length of the tokens to 100 characters due to limitations in computing power. 

### Model Development and Training

For this binary classification, we decided to apply a pre-trained BERT classification model. As a Transformers model, BERT is typically well-performing in text sequence interpretation and classification. Fine-tuning a BERT model for the task is the most straight-forward approach. In the training process, we used a 8 to 2 train and validation split on the dataset and a batch size of 16. The batch size is slightly smaller than typical due to memory limitations. The model is trained with 2 epochs as it reaches almost 100% accuracy on validation after the first epoch. We used an adam optimizer with 0.00002 learning rate. The learning rate should be low since we are conducting fine-tuning on a pre-trained model. 

### Model

The model reached about 100% accuracy on both the training and validation set with about 100% recall and precision as well. The results demonstrate that the model is extremely capable of the task.

### Discussion

The model architecture and training procedures fit the task very well with almost perfect performance in classifying fake news against true ones. The model can be wrapped into an application or web browser extension that can identify fake news articles. In this way, less people would be tricked by fake news. There are not notable limitations as the dataset used is public and the pre-trained BERT model is open source. For the continuation of the project, we may enlarge the dataset by crawling more news articles and test the model on a wider variety of news article types. The current dataset majorly consists of international news and political news.  

### References

[1] https://www.kaggle.com/code/evilspirit05/bert-based-fake-news-detector

[2] https://colab.research.google.com/drive/1NrVDXktmixZuHIILBUujVp9nCwwGsPu8?usp=sharing

The code was run in this Colab notebook:

https://colab.research.google.com/drive/17C_uzgd75BFgJ09atetJ90Wcp-YkNj0w?usp=sharing
