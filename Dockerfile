# 基础镜像：Python 3.10（适配所有依赖版本）
FROM python:3.10-slim

# 设置工作目录（容器内项目根目录）
WORKDIR /app

# 安装系统依赖：git（train.py用gitpython）、curl（检查服务状态）、gcc（部分Python库编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*  # 清理缓存，减小镜像体积

# 复制依赖清单并安装Python库（--no-cache-dir避免缓存，减小体积）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件到容器内（确保.dockerignore排除冗余文件，如venv、__pycache__）
COPY . .

# 给启动脚本执行权限
RUN chmod +x /app/entrypoint.sh

# 容器启动入口（执行全流程脚本）
ENTRYPOINT ["/app/entrypoint.sh"]

