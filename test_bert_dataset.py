import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoConfig
import time


def truncate_text(text, tokenizer, max_length=510):
    """
    截断文本到最大长度，保留[CLS]和[SEP]位置[3](@ref)
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        # 使用head+tail策略：前后截断组合[3](@ref)
        head_len = max_length // 4  # 前1/4
        tail_len = max_length - head_len  # 后3/4
        tokens = tokens[:head_len] + tokens[-tail_len:]
    return tokenizer.convert_tokens_to_string(tokens)

#直接截断
def truncate_text_direct(text, tokenizer, max_length=510):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.convert_tokens_to_string(tokens)


def main():
    # 1. 定义路径
    LOCAL_MODEL_PATH = "./bert_final_model_work"
    TEST_CSV_PATH = "work/processed_data_work.csv"
    OUTPUT_CSV_PATH = "test_all.csv"

    print(f"从本地路径加载模型: {LOCAL_MODEL_PATH}")

    # 2. 加载模型配置和分词器
    config = AutoConfig.from_pretrained(LOCAL_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    # 设置分词器最大长度并启用截断[4](@ref)
    tokenizer.model_max_length = 512
    tokenizer.truncation = True

    id2label = config.id2label
    label2id = config.label2id
    num_labels = len(id2label)

    print(f"标签数量: {num_labels}")
    print(f"最大序列长度: {tokenizer.model_max_length}")

    # 3. 实例化 pipeline
    try:
        classifier = pipeline(
            "text-classification",
            model=LOCAL_MODEL_PATH,
            tokenizer=tokenizer,
            return_all_scores=True,
            truncation=True,  # 启用截断[4](@ref)
            max_length=512  # 设置最大长度[6](@ref)
        )
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 4. 加载测试数据
    print("加载测试数据...")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"测试集大小: {len(test_df)}")
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return

    # 5. 预处理文本：截断超长文本[3](@ref)
    print("预处理文本（截断超长部分）...")
    texts = test_df['text'].tolist()
    true_labels = test_df['text_label'].tolist()

    truncated_texts = []
    for i, text in enumerate(texts):
        # 检查文本长度并截断
        tokens = tokenizer.tokenize(str(text))
        if len(tokens) > 510:  # 预留[CLS]和[SEP]位置
            truncated_text = truncate_text_direct(str(text), tokenizer)
            truncated_texts.append(truncated_text)
            if i < 5:  # 打印前5个截断示例
                print(f"样本 {i} 被截断: {len(tokens)} -> 510 tokens")
        else:
            truncated_texts.append(str(text))

    texts = truncated_texts

    # 6. 批量推理
    print("开始批量推理...")
    start_time = time.time()

    batch_size = 32  # 减小批大小以避免内存问题
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            batch_results = classifier(
                batch_texts,
                truncation=True,  # 确保截断[4](@ref)
                padding=True,  # 启用填充
                max_length=512  # 设置最大长度
            )
            all_predictions.extend(batch_results)
        except Exception as e:
            print(f"推理批次 {i // batch_size} 失败: {e}")
            # 如果批量失败，尝试单条处理
            for j, text in enumerate(batch_texts):
                try:
                    result = classifier([text], truncation=True, max_length=512)
                    all_predictions.extend(result)
                except Exception as single_e:
                    print(f"单条推理失败: {single_e}")
                    # 返回空预测
                    all_predictions.append([{'label': 'ERROR', 'score': 0.0}])

        if i % 100 == 0:
            print(f"已处理 {i}/{len(texts)} 条数据")

    end_time = time.time()
    print(f"推理完成，耗时: {end_time - start_time:.4f} 秒")

    # 7. 处理预测结果
    print("处理预测结果...")

    # 定义需要的top-k值（不超过标签数量）
    k_values = [1, 5, 10, 20]
    k_values = [k for k in k_values if k <= num_labels]

    results = []
    valid_predictions = 0

    for i, (true_label, prediction_scores) in enumerate(zip(true_labels, all_predictions)):
        # 跳过错误预测
        if not prediction_scores or 'ERROR' in str(prediction_scores):
            continue

        valid_predictions += 1

        # 将预测结果按概率排序
        sorted_predictions = sorted(prediction_scores, key=lambda x: x['score'], reverse=True)

        # 提取各top-k的预测结果
        row_result = {
            'text': texts[i],
            'text_label': true_label
        }

        for k in k_values:
            top_k_labels = [pred['label'] for pred in sorted_predictions[:k]]
            top_k_scores = [pred['score'] for pred in sorted_predictions[:k]]

            # 将标签和概率组合成字符串
            label_score_pairs = [f"{label}:{score:.4f}" for label, score in zip(top_k_labels, top_k_scores)]
            row_result[f'top{k}_res'] = ';'.join(label_score_pairs)

        results.append(row_result)

    print(f"有效预测数量: {valid_predictions}/{len(true_labels)}")

    # 8. 保存结果到CSV
    print("保存预测结果...")
    if results:
        results_df = pd.DataFrame(results)

        # 确保列的顺序正确
        columns_order = ['text', 'text_label'] + [f'top{k}_res' for k in k_values]
        results_df = results_df[columns_order]

        results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"预测结果已保存到: {OUTPUT_CSV_PATH}")
    else:
        print("无有效预测结果，跳过保存。")
        return

    # 9. 计算top-k准确率
    print("\n=== Top-K 准确率 ===")

    # 过滤无效预测
    valid_indices = []
    valid_true_labels = []
    valid_predictions_list = []

    for i, (true_label, prediction_scores) in enumerate(zip(true_labels, all_predictions)):
        if prediction_scores and 'ERROR' not in str(prediction_scores):
            valid_indices.append(i)
            valid_true_labels.append(true_label)
            valid_predictions_list.append(prediction_scores)

    if not valid_true_labels:
        print("无有效预测用于准确率计算。")
        return

    # 将真实标签转换为ID
    true_label_ids = []
    for label in valid_true_labels:
        if label in label2id:
            true_label_ids.append(label2id[label])
        else:
            # 如果标签不在映射中，使用默认值
            true_label_ids.append(0)

    top_k_accuracies = {}

    for k in k_values:
        correct_count = 0

        for i, (true_label_id, prediction_scores) in enumerate(zip(true_label_ids, valid_predictions_list)):
            # 获取前k个预测的标签ID
            sorted_predictions = sorted(prediction_scores, key=lambda x: x['score'], reverse=True)
            sorted_label_ids = []
            for pred in sorted_predictions[:k]:
                if pred['label'] in label2id:
                    sorted_label_ids.append(label2id[pred['label']])

            # 检查真实标签是否在前k个预测中
            if true_label_id in sorted_label_ids:
                correct_count += 1

        accuracy = correct_count / len(true_label_ids) if true_label_ids else 0
        top_k_accuracies[f'top{k}_acc'] = accuracy
        print(f"Top-{k} 准确率: {accuracy:.4f}")

    # 10. 打印统计信息
    print(f"\n=== 统计信息 ===")
    print(f"测试集大小: {len(test_df)}")
    print(f"有效预测数: {len(valid_true_labels)}")
    print(f"标签数量: {num_labels}")
    #print(f"截断策略: head+tail (前128+后382 tokens)")
    print(f"截断策略: 直接截断 (前512 tokens)")
    print(f"处理的Top-K值: {k_values}")


if __name__ == "__main__":
    main()