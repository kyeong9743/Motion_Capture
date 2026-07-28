# MediaMotion 백엔드: FastAPI + docker CLI (워커 컨테이너를 구동한다).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# docker CLI만 설치한다 (마운트된 소켓을 통해 호스트 데몬과 통신한다)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
      > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

RUN pip install "fastapi==0.115.*" "uvicorn[standard]==0.30.*" "python-multipart==0.0.*"

WORKDIR /srv
COPY backend/ /srv/
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
