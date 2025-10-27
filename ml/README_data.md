# 数据版本说明（DVC跟踪）
## 1. 数据来源
- 原始数据（v1）来自Kaggle「XXX数据集」（链接：https://www.kaggle.com/xxx/xxx），包含用户行为特征（feature1：用户活跃度，feature2：消费频次），共1000条样本。

## 2. 数据版本变更（v1 → v2）
| 版本 | 对应DVC元文件 | 变更内容 | 适用场景 |
|------|---------------|----------|----------|
| v1   | data/raw.dvc  | 原始数据，含5% outliers（feature1>1000，feature2>10000）、3%缺失值 | 基线模型训练 |
| v2   | data/processed.dvc | 1. 删除outliers（基于IQR法则）；2. 用中位数填充缺失值；3. 特征标准化（mean=0，std=1） | 改进模型训练 |

## 3. 代码与数据版本关联（Git Commit → DVC数据）
- v1数据对应Git Commit：a1b2c3d（备注："DVC数据v1：原始数据集"）
- v2数据对应Git Commit：e4f5g6h（备注："DVC数据v2：清洗后数据集"）

## 4. 如何拉取指定版本数据
```bash
# 拉取v1原始数据（先切换到对应Git Commit）
git checkout a1b2c3d data/raw.dvc
dvc pull data/raw.dvc

# 拉取v2清洗后数据
git checkout e4f5g6h data/processed.dvc
dvc pull data/processed.dvc


## 二、MLflow实验跟踪（满足2个实验要求）
### 1. 编写`ml/train.py`（带MLflow日志功能）
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import dvc.api  # 用于加载DVC管理的数据
import os
from git import Repo  # 用于获取Git Commit SHA

# --------------------------
# 1. 初始化MLflow实验
# --------------------------
mlflow.set_experiment("用户行为预测实验")  # 实验名称（自定义）

# --------------------------
# 2. 获取Git Commit和DVC数据版本（关联代码+数据）
# --------------------------
# 获取当前Git Commit SHA（代码版本）
repo = Repo(os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True)
git_commit = repo.head.commit.hexsha[:8]  # 取前8位便于查看

# 加载DVC数据（v1：原始数据；v2：清洗后数据，按需切换）
# 方式1：加载v1原始数据
# raw_data_path = dvc.api.open("data/raw/train_raw.csv").name
# 方式2：加载v2清洗后数据（用于改进模型）
processed_data_path = dvc.api.open("data/processed/train_clean.csv").name
test_data_path = dvc.api.open("data/processed/test_clean.csv").name

# 获取DVC数据版本（哈希值，从.dvc文件中读取）
with open("data/processed.dvc", "r") as f:
    dvc_data_hash = f.readlines()[1].split(":")[1].strip()  # 提取data.dvc中的哈希

# --------------------------
# 3. 数据加载与预处理
# --------------------------
train_df = pd.read_csv(processed_data_path)
test_df = pd.read_csv(test_data_path)
X_train = train_df[["feature1", "feature2"]]
y_train = train_df["label"]  # 假设标签列为"label"（0/1）
X_test = test_df[["feature1", "feature2"]]
y_test = test_df["label"]

# --------------------------
# 4. 实验1：基线模型（LogisticRegression默认参数）
# --------------------------
with mlflow.start_run(run_name="基线模型-LogisticRegression默认参数"):
    # 日志代码版本、数据版本
    mlflow.log_param("git_commit", git_commit)
    mlflow.log_param("dvc_data_version", dvc_data_hash)
    mlflow.log_param("data_version", "v2（清洗后数据）")
    
    # 日志超参数
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", 1.0)  # 默认正则化参数
    mlflow.log_param("solver", "lbfgs")  # 默认求解器
    
    # 训练模型
    baseline_model = LogisticRegression(random_state=42)
    baseline_model.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = baseline_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 日志指标（优化指标选F1，因数据可能存在类别不平衡）
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # 日志 artifacts（模型文件、测试集预测结果）
    mlflow.sklearn.log_model(baseline_model, "baseline_model")  # 保存模型
    test_df["y_pred"] = y_pred
    test_df.to_csv("ml/results/baseline_pred.csv", index=False)
    mlflow.log_artifact("ml/results/baseline_pred.csv", "prediction_results")

# --------------------------
# 5. 实验2：改进模型（调优超参数）
# --------------------------
with mlflow.start_run(run_name="改进模型-LogisticRegression调优"):
    # 日志代码版本、数据版本
    mlflow.log_param("git_commit", git_commit)
    mlflow.log_param("dvc_data_version", dvc_data_hash)
    mlflow.log_param("data_version", "v2（清洗后数据）")
    
    # 日志调优后的超参数
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", 0.1)  # 调小正则化强度
    mlflow.log_param("solver", "liblinear")  # 适合小数据集的求解器
    mlflow.log_param("class_weight", "balanced")  # 处理类别不平衡
    
    # 训练模型
    improved_model = LogisticRegression(
        C=0.1, solver="liblinear", class_weight="balanced", random_state=42
    )
    improved_model.fit(X_train, y_train)
    
    # 预测与评估
    y_pred_improved = improved_model.predict(X_test)
    accuracy_improved = accuracy_score(y_test, y_pred_improved)
    f1_improved = f1_score(y_test, y_pred_improved)
    
    # 日志指标
    mlflow.log_metric("accuracy", accuracy_improved)
    mlflow.log_metric("f1_score", f1_improved)
    
    # 日志 artifacts（模型文件、预测结果）
    mlflow.sklearn.log_model(improved_model, "improved_model")  # 保存模型（生产用）
    test_df["y_pred_improved"] = y_pred_improved
    test_df.to_csv("ml/results/improved_pred.csv", index=False)
    mlflow.log_artifact("ml/results/improved_pred.csv", "prediction_results")

print("实验跟踪完成！可通过`mlflow ui`查看详情")