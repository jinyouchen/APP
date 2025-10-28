import pytest
import json
from app.main import app

client = app.test_client()


# 测试1：正常输入（预期成功预测）
def test_normal_prediction():
    response = client.post("/predict", json={"feature1": 3, "feature2": 30})
    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert response.json["prediction"] in [0, 1]


# 测试2：边缘案例 - 缺少特征（预期返回400错误）
def test_missing_feature():
    response = client.post("/predict", json={"feature1": 3})  # 缺少feature2
    assert response.status_code == 400
    assert response.json["status"] == "error"


# 测试3：边缘案例 - 异常值输入（预期返回200，模型正常处理）
def test_outlier_input():
    response = client.post(
        "/predict", json={"feature1": 1000, "feature2": 10000}
    )  # 异常大值
    assert response.status_code == 200
    assert response.json["status"] == "success"


# 运行测试：CMD执行 `pytest app/tests/test_app.py -v` 需全部通过
