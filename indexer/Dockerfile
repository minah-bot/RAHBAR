# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt

FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp TMPDIR=/tmp
# OCR needs the tesseract binary and poppler (pdf2image). Nothing else is installed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng poppler-utils \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin rahbar
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY pipeline ./pipeline
USER 10001
# One-shot: runs the pipeline and exits. Exit code 0 means the index was written.
ENTRYPOINT ["python", "-m", "pipeline.run"]
CMD []
