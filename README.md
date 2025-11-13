# Chinese-text-classifier
This work helps to realize quick Chinese text classification through the pretrained model—google-bert/bert-base-chinese.

pretrained-model：bert-base-chinese

architechture：BERT，language：Chinese，vocabulary：21128，parameters：110M，model_size：440MB

download pretrained model：

python download_pretrained_model.py

virtual environment：

conda create -n bert python=3.9

conda activate bert

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install fast-fit

dataset path：/work

-train_new.csv(text-text_label)

-test_new.csv(text-text_label)

model_train：

python train_bert.py

model inference：

single text/text sequence inference：

python test_bert.py

batch inference：

python predict_bert_dataset.py

batch test：

python test_bert_dataset.py
