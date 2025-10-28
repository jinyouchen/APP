# 导入库
import dagshub
import mlflow

# 1. 初始化Dagshub连接（自动配置MLflow跟踪地址到Dagshub）
# 注意：repo_owner和repo_name需与你的Dagshub仓库一致
dagshub.init(
    repo_owner="jinyouchen",  # 你的Dagshub用户名
    repo_name="APP",          # 你的仓库名
    mlflow=True               # 自动配置MLflow连接到Dagshub
)

# 2. 使用MLflow记录实验（示例：记录指标和参数）
with mlflow.start_run(run_name="first_experiment"):  # 定义实验名称
    # 记录参数（如超参数）
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("model_type", "LogisticRegression")
    
    # 记录指标（如模型性能）
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("f1_score", 0.82)
    
    # 可选：自动日志（支持主流框架，如scikit-learn、TensorFlow等）
    # mlflow.autolog()  # 打开后，训练模型时会自动记录参数、指标、模型文件

print("实验记录完成！可在Dagshub仓库的MLflow面板查看结果。")