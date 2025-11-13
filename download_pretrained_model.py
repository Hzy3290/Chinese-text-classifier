import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModel, AutoTokenizer
#下载模型和分词器
#model_name ="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_name = "google-bert/bert-base-chinese"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#保存到本地
model.save_pretrained("./bert-base-chinese" )
tokenizer.save_pretrained("./bert-base-chinese")