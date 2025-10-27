import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression  # 示例模型（需训练）
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import git

# 1. 配置MLflow（符合1-57要求）
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("user_classification_model_experiments")  # 如“用户分类模型实验”

# 2. 获取代码版本（Git commit SHA，符合1-60要求）
def get_git_commit():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha

# 3. 加载并预处理数据（支撑模型训练，符合1-5）
def load_data():
    # 示例：用本地数据（跳过DVC，暂用简单数据集）
    data = pd.DataFrame({
        "feature1": [1,2,3,4,5,6],
        "feature2": [10,20,30,40,50,60],
        "target": [0,0,0,1,1,1]
    })
    X = data[["feature1", "feature2"]]
    y = data["target"]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 训练模型（支持2个实验：基线+改进，符合1-71要求）
def train_model(penalty="l2", C=1.0):
    X_train, X_test, y_train, y_test = load_data()
    git_commit = get_git_commit()

    # 启动MLflow Run（符合1-59至1-64要求）
    with mlflow.start_run():
        # 记录代码版本（符合1-60）
        mlflow.set_tag("git_commit", git_commit)
        # 记录超参数（符合1-62）
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("C", C)
        # 训练模型（符合1-5“训练/微调”要求）
        model = LogisticRegression(penalty=penalty, C=C)
        model.fit(X_train, y_train)
        # 记录指标（符合1-63）
        y_pred = model.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        # 记录模型Artifact（符合1-64）
        mlflow.sklearn.log_model(model, "model")
        print(f"Run完成：Accuracy={accuracy_score(y_test, y_pred):.2f}")
        return model

# 5. 运行2个实验（基线+改进，符合1-71至1-73要求）
if __name__ == "__main__":
    print("实验1：基线模型（默认超参数）")
    train_model(penalty="l2", C=1.0)  # 基线模型
    print("实验2：改进模型（调优超参数）")
    train_model(penalty="l2", C=0.5)  # 改进模型（调整正则化强度）