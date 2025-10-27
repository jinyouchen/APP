import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import git
import time
import os  # 新增：用于文件路径处理

# 1. 配置MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("user_classification_model_experiments")

# 新增：定义Run ID保存路径（项目根目录下）
RUN_ID_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "run_id.txt")

# 2. 获取代码版本（保持不变）
def get_git_commit():
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        return "non_git_environment"

# 3. 加载并预处理数据（保持不变）
def load_data():
    np.random.seed(42)
    class0_feature1 = np.random.normal(1, 0.6, 50)
    class0_feature2 = np.random.normal(1, 0.6, 50)
    class1_feature1 = np.random.normal(3, 0.6, 50)
    class1_feature2 = np.random.normal(3, 0.6, 50)
    
    data = pd.DataFrame({
        "feature1": np.concatenate([class0_feature1, class1_feature1]),
        "feature2": np.concatenate([class0_feature2, class1_feature2]),
        "target": [0]*50 + [1]*50
    })
    
    X_train, X_test, y_train, y_test = train_test_split(
        data[["feature1", "feature2"]], 
        data["target"], 
        test_size=0.3, 
        random_state=42,
        stratify=data["target"]
    )
    return X_train, X_test, y_train, y_test

# 4. 训练模型（修改：保存Run ID到文件）
def train_model(penalty="l2", C=1.0, experiment_name=""):
    X_train, X_test, y_train, y_test = load_data()
    git_commit = get_git_commit()
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with mlflow.start_run(run_name=f"{experiment_name}_{current_time}"):
        # 获取当前Run ID
        run_id = mlflow.active_run().info.run_id  # 新增：获取Run ID
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("run_time", current_time)
        
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("C", C)
        
        model = LogisticRegression(penalty=penalty, C=C, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=1)
        
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)
        mlflow.sklearn.log_model(model, "model")
        
        # 新增：保存Run ID到文件
        with open(RUN_ID_FILE, "w") as f:
            f.write(run_id)
        print(f"[{current_time}] {experiment_name} 完成：Accuracy={accuracy:.2f}, F1={f1:.2f}")
        print(f"✅ Run ID已保存到 {RUN_ID_FILE}：{run_id}")  # 新增：打印保存信息
        return model

# 5. 运行实验（保持不变）
if __name__ == "__main__":
    print("=== 开始模型训练实验 ===")
    train_model(penalty="l2", C=1.0, experiment_name="实验1：基线模型（默认超参数）")
    train_model(penalty="l2", C=0.5, experiment_name="实验2：改进模型（调优超参数）")
    print("=== 所有实验完成 ===")