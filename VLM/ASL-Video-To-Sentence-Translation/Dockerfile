FROM python:3.10

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/hf_cache
ENV TRANSFORMERS_CACHE=/data/hf_cache
ENV HF_DATASETS_CACHE=/data/hf_cache/datasets

RUN apt-get update && apt-get install -y \
    git git-lfs wget curl unzip tar ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyterlab

COPY . .

CMD ["bash", "-lc", "mkdir -p /root/.jupyter && printf \"%s\n\" \"c.ServerApp.ip = '0.0.0.0'\" \"c.ServerApp.port = 7860\" \"c.ServerApp.open_browser = False\" \"c.ServerApp.allow_root = True\" \"c.ServerApp.allow_remote_access = True\" \"c.IdentityProvider.token = ''\" \"c.PasswordIdentityProvider.password_required = False\" \"c.ServerApp.password = ''\" \"c.ServerApp.disable_check_xsrf = True\" \"c.ServerApp.trust_xheaders = True\" \"c.ServerApp.tornado_settings = {'headers': {'Content-Security-Policy': \\\"frame-ancestors * 'self' https://huggingface.co https://*.hf.space;\\\"}}\" > /root/.jupyter/jupyter_server_config.py && jupyter lab --config=/root/.jupyter/jupyter_server_config.py"]