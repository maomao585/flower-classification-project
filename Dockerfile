FROM efreidevopschina.azurecr.io/cache/library/python:3.9-slim

WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
# 首先安装 PyTorch（使用CPU版本避免CUDA依赖）
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
# 复制依赖文件 - 利用Docker缓存层
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data mlruns models

# 设置入口点
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]