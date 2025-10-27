#!/bin/bash
set -e  # 命令执行失败时立即退出（便于定位错误）

# ==============================================
# 步骤1：运行训练脚本，生成Run ID文件（核心：关联模型版本）
# ==============================================
echo -e "\n=== [1/4] 开始训练模型，生成Run ID ==="
python train.py
if [ ! -f "run_id.txt" ]; then
    echo "❌ 训练脚本未生成run_id.txt，请检查train.py逻辑！"
    exit 1
fi
echo "✅ 训练完成，Run ID文件已生成：$(cat run_id.txt)"


# ==============================================
# 步骤2：用uvicorn启动Flask服务（后台运行，适配新增依赖）
# ==============================================
echo -e "\n=== [2/4] 用uvicorn启动Flask服务 ==="
# uvicorn启动命令：绑定0.0.0.0:5000（容器内必须暴露0.0.0.0），后台运行并记录进程ID
uvicorn app.main:app --host 0.0.0.0 --port 5000 --workers 1 &
SERVER_PID=$!  # 保存服务进程ID，后续用于关闭服务
echo "✅ 服务已启动（进程ID：$SERVER_PID），等待就绪..."


# ==============================================
# 步骤3：等待服务就绪（避免测试时服务未启动）
# ==============================================
echo -e "\n=== [3/4] 等待服务就绪 ==="
WAIT_TIMEOUT=30  # 最大等待时间（秒）
WAIT_INTERVAL=2  # 每隔2秒检查一次
elapsed=0

while ! curl -s -o /dev/null "http://localhost:5000"; do
    if [ $elapsed -ge $WAIT_TIMEOUT ]; then
        echo "❌ 服务启动超时（超过$WAIT_TIMEOUT秒），终止测试！"
        kill $SERVER_PID  # 关闭未就绪的服务
        exit 1
    fi
    echo "服务未就绪，已等待$elapsed秒，继续等待..."
    sleep $WAIT_INTERVAL
    elapsed=$((elapsed + WAIT_INTERVAL))
done
echo "✅ 服务已就绪（http://localhost:5000）"


# ==============================================
# 步骤4：运行测试脚本（验证接口功能）
# ==============================================
echo -e "\n=== [4/4] 运行测试用例 ==="
pytest app/tests/test_app.py -v
TEST_EXIT_CODE=$?  # 记录测试结果（0=成功，非0=失败）


# ==============================================
# 步骤5：清理资源（关闭服务，输出最终结果）
# ==============================================
echo -e "\n=== 测试完成，清理资源 ==="
kill $SERVER_PID  # 关闭Flask服务
echo "✅ 服务已关闭（进程ID：$SERVER_PID）"

# 根据测试结果退出容器（0=成功，非0=失败）
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "\n🎉 所有测试用例执行成功！"
    exit 0
else
    echo -e "\n❌ 部分测试用例执行失败，请检查代码！"
    exit $TEST_EXIT_CODE
fi