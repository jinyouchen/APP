from flask import Flask, request, jsonify
import joblib
import os

# 创建Flask应用实例
app = Flask(__name__)  # 新增这一行

# 明确指定模型所在目录（必须与保存路径的目录一致）
MODEL_PATH = r"C:\Users\32583\Desktop\APP\data\model"  # 注意是目录，不是文件

# 拼接完整文件路径
model_file = os.path.join(MODEL_PATH, "model.pkl")  # 这里修正了多余的小数点，原代码有个错误的"."

# 加载前先检查文件是否存在（避免报错，方便调试）
if not os.path.exists(model_file):
    raise FileNotFoundError(f"模型文件不存在！请检查路径：{model_file}")

# 加载模型
model = joblib.load(model_file)

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