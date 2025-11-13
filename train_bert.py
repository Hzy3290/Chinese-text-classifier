import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# train.py
import numpy as np
import evaluate
import json
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


def main():
    # --- 1. 数据加载与映射 ---
    print("加载数据...")
    data_files = {"train": "work/train_new.csv", "test": "work/test_new.csv"}
    raw_datasets = load_dataset("csv", data_files=data_files)

    # 创建标签映射
    label_names = raw_datasets["train"].unique("text_label")
    label_names.sort()
    num_labels = len(label_names)
    label2id = {label_name: i for i, label_name in enumerate(label_names)}
    id2label = {i: label_name for i, label_name in enumerate(label_names)}

    print(f"标签数量: {num_labels}")
    print(f"Label to ID 映射: {label2id}")

    # 将字符串标签转换为整数 ID
    def map_labels(example):
        example['label'] = label2id[example['text_label']]
        return example

    raw_datasets = raw_datasets.map(map_labels, batched=False)

    # --- 2. 初始化 (Tokenizer 和 Model) ---
    print("加载模型和分词器...")
    model_checkpoint = "google-bert/bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    # 期望看到一条关于 'classifier.weight' 未初始化的警告，这是正常的。

    # --- 3. 预处理 ---
    print("预处理数据...")

    def preprocess_function(examples):
        # max_length=256 是一个基于中文 WordPiece 特性的稳健选择
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "text_label"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")


    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 4. 评估函数 ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        # 计算top-k准确率
        top_k_accuracies = {}
        k_values = [10, 20, 30, 40, 50]  # 可调整的k值

        # 确保k值不超过类别数量
        max_possible_k = min(50, num_labels)
        k_values = [k for k in k_values if k <= max_possible_k]

        for k in k_values:
            # 获取每个样本的前k个预测结果
            top_k_predictions = np.argsort(logits, axis=1)[:, -k:]

            # 检查真实标签是否在前k个预测中
            correct = 0
            for i, label in enumerate(labels):
                if label in top_k_predictions[i]:
                    correct += 1

            top_k_accuracy = correct / len(labels)
            top_k_accuracies[f"top_{k}_accuracy"] = top_k_accuracy

        # 合并所有指标
        metrics = {"accuracy": accuracy, "f1": f1}
        metrics.update(top_k_accuracies)

        return metrics

    # --- 5. 训练参数 ---
    training_args = TrainingArguments(
        output_dir="./bert_results_work",  # 检查点输出目录
        num_train_epochs=40,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,

        # (修改点) 评估、日志和保存策略
        eval_strategy="epoch",  # [12, 24]
        logging_strategy="epoch",  # (新增) 匹配评估策略
        save_strategy="epoch",  #

        load_best_model_at_end=True,
        # (修改点) 明确指定使用 'eval' 数据集上的 'f1' 分数
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,

        logging_dir="./bert_logs_work",  # (新增) TensorBoard 日志目录
        report_to="tensorboard",  # (新增) 启用 TensorBoard 报告

        save_total_limit=2,
        push_to_hub=False,
    )

    # --- 6. Trainer 实例化与执行 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    print("开始训练...")
    train_result = trainer.train()

    # 保存最终的最佳模型
    final_model_path = "./bert_final_model_work"
    print(f"训练完成，将最佳模型保存到 {final_model_path}")
    trainer.save_model(final_model_path)
    # 确保分词器也保存在同一目录中，以便 pipeline 加载
    tokenizer.save_pretrained(final_model_path)

    # --- 7. (新增) 保存训练日志 ---
    print("保存训练日志...")

    # 保存指标摘要 (例如: 'train_runtime', 'train_samples_per_second'等)
    trainer.save_metrics("all", train_result.metrics)
    print(f"指标摘要已保存到: {final_model_path}/all_metrics.json")

    # 保存完整的日志历史 (包含每一步的 train_loss, eval_loss, train_accuracy, eval_accuracy)
    # trainer.state.log_history 包含了所有日志
    log_history_path = f"{final_model_path}/log_history.json"

    # --- 8. 最终评估（输出详细的top-k结果） ---
    print("\n=== 最终评估结果 ===")
    eval_results = trainer.evaluate()

    print(f"准确率 (Top-1): {eval_results['eval_accuracy']:.4f}")
    print(f"加权F1分数: {eval_results['eval_f1']:.4f}")

    # 打印所有top-k准确率
    for metric_name, metric_value in eval_results.items():
        if metric_name.startswith('eval_top_') and metric_name.endswith('_accuracy'):
            k_value = metric_name.replace('eval_top_', '').replace('_accuracy', '')
            print(f"Top-{k_value} 准确率: {metric_value:.4f}")

    try:
        with open(log_history_path, "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, indent=4)
        print(f"详细日志历史已保存到: {log_history_path}")
    except Exception as e:
        print(f"保存 log_history.json 失败: {e}")


if __name__ == "__main__":
    main()