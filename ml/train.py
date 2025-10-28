# 依赖安装提示：
# pip install mlflow scikit-learn pandas gitpython

import mlflow
import mlflow.sklearn
import pandas as pd
import pickle  # 用于保存模型
import os  # 用于创建目录
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import git
from git.exc import InvalidGitRepositoryError


# 1. 配置MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("user_classification_model_experiments")


# 2. 获取代码版本（带异常处理）
def get_git_commit():
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except InvalidGitRepositoryError:
        print("警告：当前目录不是Git仓库，无法获取commit版本")
        return "unknown_commit"


# 3. 加载并预处理数据
def load_data():
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [10, 20, 30, 40, 50, 60],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    X = data[["feature1", "feature2"]]
    y = data["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)


# 4. 训练模型并保存到指定路径
def train_model(penalty="l2", C=1.0):
    X_train, X_test, y_train, y_test = load_data()
    git_commit = get_git_commit()

    # 定义模型保存路径（按需求指定）
    model_save_path = r"C:\Users\32583\Desktop\APP\data\model\model.pkl"
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    with mlflow.start_run() as run:
        # 记录标签、超参数
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("experiment_type", "baseline" if C == 1.0 else "improved")
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("C", C)

        # 训练模型
        model = LogisticRegression(penalty=penalty, C=C, random_state=42)
        model.fit(X_train, y_train)

        # 评估指标
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)

        # 保存模型到MLflow（Artifact）
        mlflow.sklearn.log_model(model, "model")

        # 保存模型到指定本地路径（核心修改）
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"模型已保存到本地路径：{model_save_path}")

        # 打印运行信息
        print(
            f"Run完成：Run ID = {run.info.run_id}，Accuracy = {accuracy:.2f}，F1 = {f1:.2f}"
        )
        return model


# 5. 运行实验
if __name__ == "__main__":
    print("实验1：基线模型（默认超参数 C=1.0）")
    train_model(penalty="l2", C=1.0)
    print("\n实验2：改进模型（调优超参数 C=0.5）")
    train_model(penalty="l2", C=0.5)

    print("\n=== 实验结束 ===")
    print("可通过以下命令启动MLflow UI查看详细结果：")
    print("mlflow ui --backend-store-uri ./mlruns")
    print("访问地址：http://localhost:5000")
