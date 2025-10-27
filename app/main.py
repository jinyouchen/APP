import pandas as pd
from flask import Flask, request, jsonify
import mlflow.pyfunc
import os  # 用于文件路径处理

# 初始化Flask应用
app = Flask(__name__)

# --------------------------
# MLflow 配置（修改：从文件读取Run ID）
# --------------------------
# 定义Run ID文件路径（与train.py保存路径一致）
MLFLOW_TRACKING_URI = "file:///app/mlruns"  
MODEL_ARTIFACT_PATH = "model"
# Run ID 文件路径（train.py 保存在 /app/run_id.txt，main.py 在 /app/app/ 下，需向上跳一级）
RUN_ID_FILE = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "run_id.txt")


# 从文件读取Run ID
try:
    with open(RUN_ID_FILE, "r") as f:
        RUN_ID = f.read().strip()
    print(f"✅ 从文件加载Run ID：{RUN_ID}")
except FileNotFoundError:
    raise Exception(f"❌ 未找到Run ID文件，请先训练模型生成 {RUN_ID_FILE}")
except Exception as e:
    raise Exception(f"❌ 读取Run ID失败：{str(e)}")

# 配置MLflow跟踪地址
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --------------------------
# 模型加载（保持逻辑不变）
# --------------------------
try:
    model_uri = f"runs:/{RUN_ID}/{MODEL_ARTIFACT_PATH}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"✅ 模型加载成功！MLflow URI：\n{model_uri}")
except Exception as e:
    print(f"❌ 启动失败：模型加载错误 - {str(e)}")
    raise

# --------------------------
# 预测接口（保持不变）
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "msg": "请求格式错误！请发送JSON格式数据（示例：{\"feature1\": 3, \"feature2\": 30}）"
            }), 400
        
        required_features = ["feature1", "feature2"]
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({
                "status": "error",
                "msg": f"缺少必要特征：{', '.join(missing_features)}"
            }), 400
        
        try:
            feature1 = float(data["feature1"])
            feature2 = float(data["feature2"])
        except ValueError:
            return jsonify({
                "status": "error",
                "msg": "特征值格式错误！feature1和feature2必须是整数或小数（示例：3 或 3.5）"
            }), 400
        
        input_data = pd.DataFrame(
            data=[[feature1, feature2]],
            columns=["feature1", "feature2"]
        )
        
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "input_features": {"feature1": feature1, "feature2": feature2},
            "msg": "预测完成"
        }), 200
    
    except Exception as e:
        error_msg = f"预测处理失败：{str(e)}"
        print(f"❌ {error_msg}")
        return jsonify({
            "status": "error",
            "msg": error_msg
        }), 500

# --------------------------
# 启动应用（保持不变）
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)