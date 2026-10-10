FROM python:3.12-slim
RUN --mount=type=secret,id=cloud_ca \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && if [ -f /run/secrets/cloud_ca ]; then \
         printf 'Acquire::https::CaInfo "/run/secrets/cloud_ca";\n' > /etc/apt/apt.conf.d/99-cloud-ca; \
       fi \
    && apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -f /etc/apt/apt.conf.d/99-cloud-ca && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/cpu.lock.txt requirements/cpu.lock.txt
RUN --mount=type=secret,id=cloud_ca \
    if [ -f /run/secrets/cloud_ca ]; then export PIP_CERT=/run/secrets/cloud_ca; fi; \
    pip install --no-cache-dir -r requirements/cpu.lock.txt
COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-build-isolation --no-deps .
ENV SPATIALMEMORY_RUNS=/runs
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "spatialmemory.app:app", "--host", "0.0.0.0", "--port", "8000"]
