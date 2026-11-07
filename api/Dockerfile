# syntax=docker/dockerfile:1
# --- build stage: resolve and install pinned dependencies into a venv --------------------
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt

# --- runtime stage: no compiler, no pip cache, non-root, read-only-friendly --------------
FROM python:3.12-slim
# PYTHONDONTWRITEBYTECODE: the root filesystem is read-only in the cluster (section 4.7).
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin rahbar
COPY --from=build /opt/venv /opt/venv
WORKDIR /srv
COPY app ./app
USER 10001
EXPOSE 8000
# Exec form: SIGTERM reaches uvicorn (PID 1), so a rolling update drains instead of killing.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
