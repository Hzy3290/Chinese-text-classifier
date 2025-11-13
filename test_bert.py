import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline
import time

# 1. 定义本地模型路径
# 这必须是 train.py 脚本保存模型的目录 (例如 "./final_model")
LOCAL_MODEL_PATH = "./bert_final_model_context"

print(f"从本地路径加载模型: {LOCAL_MODEL_PATH}")
# 2. 实例化 pipeline
# 指定任务为 "text-classification"
# 将模型路径传递给 model= 参数 [4, 8]
# pipeline 会自动加载该目录下的所有配置、权重和分词器
try:
    classifier = pipeline("text-classification", model=LOCAL_MODEL_PATH)
    print("模型加载成功。")
except Exception as e:
    print(f"加载模型失败: {e}")
    print("请确保路径正确，并且目录中包含 config.json, pytorch_model.bin 和 tokenizer.json 等文件。")
    exit(1)

#test 8,19
# 3. 准备要推理的文本
texts_to_classify = [
    f'数据源名：产业金融域，数据库名：cyjry_database，数据表名：SYS_SMSCUSTOMIZATION，数据表备注：，"key": "发送模板"，"value": "['']"，"comment": "发送模板"，"label": ""，context:[{"key": "收信人姓名", "value": "['']", "comment": "收信人姓名", "label": ""}]',
    f'数据源名：产业金融域，数据库名：cyjry_database，数据表名：PRPDPRINTRECORD，数据表备注：，"key": "备用字段 被保险人证件号"，"value": "['']"，"comment": "备用字段 被保险人证件号"，"label": ""，context:[]',
]

# 4. 执行推理
print("\n开始推理...")
start_time = time.time()
results = classifier(texts_to_classify)
end_time = time.time()

# 5. 打印结果
print(f"推理完成，耗时: {end_time - start_time:.4f} 秒")
print("\n推理结果:")
for i, (text, result) in enumerate(zip(texts_to_classify, results)):
    print(f"--- 文本 {i+1} ---")
    print(f"内容: {text}")
    # pipeline 会自动使用 id2label 映射返回 'label'
    print(f"预测: label='{result['label']}', score={result['score']:.4f}")

# 单条文本推理
print("\n--- 单条推理示例 ---")
single_text = f'数据源名：产业金融域，数据库名：cyjry_database，数据表名：PRPDPRINTRECORD，数据表备注：，"key": "备用字段 被保险人证件号"，"value": "['']"，"comment": "备用字段 被保险人证件号"，"label": ""，context:[]'
single_result = classifier(single_text)
print(f"内容: {single_text}")
print(f"预测: {single_result}")