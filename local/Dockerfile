FROM python:3.12.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.5.2 pytest==9.1.1 \
 && pip install --no-cache-dir torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN python -c "import sys, numpy, torch; assert sys.version_info[:3] == (3,12,14); assert (numpy.__version__, torch.__version__) == ('2.5.2','2.6.0+cpu')"
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/scratch/81995298881b/SANW_practical_work
ENTRYPOINT ["python", "local/run.py"]
