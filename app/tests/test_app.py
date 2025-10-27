# test_app.py
import sys
import os
from pathlib import Path

# 1. 获取当前测试文件（test_app.py）的绝对路径
current_file_path = Path(__file__).resolve()
# 2. 从 test_app.py 向上跳转至项目根目录（假设项目结构为：项目根目录/tests/test_app.py）
project_root = current_file_path.parent.parent  # 上两级目录（tests/的父目录即项目根目录）
# 3. 将项目根目录加入 Python 搜索路径
sys.path.insert(0, str(project_root))

# 现在可以正常导入 app 模块了
from main import app  # 注意：根据实际目录结构调整导入路径
import pytest
import json

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
    response = client.post("/predict", json={"feature1": 1000, "feature2": 10000})  # 异常大值
    assert response.status_code == 200
    assert response.json["status"] == "success"

# 运行测试：CMD执行 `pytest tests/test_app.py -v` 需全部通过