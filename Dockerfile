# 基础镜像：Python Slim（减少镜像体积，符合1-12“slim Dockerfile”要求）
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件（利用Docker缓存，减少构建时间）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（仅复制必要文件，依赖.dockerignore排除冗余）
COPY . .

# 暴露App端口（与app/main.py一致）
EXPOSE 5000

# 启动命令（生产环境可替换为Gunicorn，此处示例用Flask）
CMD ["python", "app/main.py"]