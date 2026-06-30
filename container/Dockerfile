# tiny image for the synthetic memory probe, sizes are passed in at run time
FROM python:3.11-slim

RUN pip install --no-cache-dir "numpy>=1.24"

WORKDIR /app
COPY mem_wall.py /app/mem_wall.py

ENTRYPOINT ["python", "/app/mem_wall.py"]
