This folder contains Bert classification and Flair classification on our allsides dataset. 
To run Bert classification, you need to first download the model (recommend BERT-Base, Uncased) from here https://github.com/google-research/bert#pre-trained-models. Then, here are all the commends (Please note that BERT needs at least 11G graphic card ram to run) :
export BERT_BASE_DIR=./BERT-Classification-Tutorial/model
python run_classifier.py \
  --task_name=mnli \
  --do_train=true \
  --do_eval=true \
  --data_dir=./BERT-Classification-Tutorial/data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=6 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./output

Flair is another NLP library and to run flair, first download the train,test,eval dataset from here:https://drive.google.com/open?id=1m_MuXgPPc2JRTMLPQl-ISSJL-vtHjCAl
To run flair classification, please run flairTest.py (Using Bert word embedding may cause out of memory issue).

