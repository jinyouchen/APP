from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
# 加载MLflow保存的模型（示例路径，需替换为实际MLflow模型路径）
MODEL_PATH = "mlruns/<实验ID>/<Run ID>/artifacts/model"
model = joblib.load(os.path.join(MODEL_PATH, "model.pkl"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # 提取特征（需与模型输入匹配）
        features = [data["feature1"], data["feature2"]]
        # 预测
        pred = model.predict([features])[0]
        return jsonify({"status": "success", "prediction": int(pred)})
    except KeyError:
        return jsonify({"status": "error", "msg": "缺少feature1/feature2"}), 400
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
