FROM python:3.9-slim

RUN pip install --no-cache-dir mlflow==2.14.1

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
CMD ["--backend-store-uri", "/mlruns", "--default-artifact-root", "/mlruns"]
