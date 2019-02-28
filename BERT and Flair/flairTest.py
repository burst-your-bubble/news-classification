import pandas as pd

data = pd.read_csv("./data/allNews.csv", encoding='utf-8').sample(frac=1).drop_duplicates()
data = data[['V1', 'V2']].rename(columns={"V1":"label", "V2":"text"})
 
data['label'] = '__label__' + data['label'].astype(str)
data.iloc[0:int(len(data)*0.8)].to_csv('train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('dev.csv', sep='\t', index = False, header = False);

from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
from torch.optim.adam import Adam
from flair.embeddings import ELMoEmbeddings

corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./'), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')
word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]
#word_embeddings = [BertEmbeddings('bert-base-uncased')]
document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512,bidirectional=True,reproject_words=True, reproject_words_dimension=256)
classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
trainer = ModelTrainer(classifier, corpus,optimizer=Adam)
trainer.train('./', learning_rate=0.001,
              mini_batch_size=6,
             embeddings_in_memory = False,
              max_epochs=150)