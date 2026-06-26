# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /workspace/app/backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/app/backend

# Runtime library used by some numerical/Torch binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

# Install Python deps into an isolated venv (copied into runtime stage)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY app/backend/requirements.txt app/backend/requirements.docker.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.docker.txt


FROM base AS runtime

ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

# Create a non-root user before COPY --chown uses it.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

# Preserve the repo layout expected by model artifact resolvers:
# /workspace/app/backend and /workspace/artifacts/...
COPY --chown=appuser:appuser app/backend /workspace/app/backend
COPY --chown=appuser:appuser artifacts/DMF/credit_rating_artifacts/dmf_gat_tlstm/dmf_dcs_final_model.pkl \
    /workspace/artifacts/DMF/credit_rating_artifacts/dmf_gat_tlstm/dmf_dcs_final_model.pkl
COPY --chown=appuser:appuser artifacts/DMF/credit_rating_artifacts/dmf_gat_tlstm/dmf_dcs_test_predictions.csv \
    /workspace/artifacts/DMF/credit_rating_artifacts/dmf_gat_tlstm/dmf_dcs_test_predictions.csv
COPY --chown=appuser:appuser artifacts/DMF/credit_rating_artifacts/dmf_dcs_financial_shap_importance_by_class.csv \
    /workspace/artifacts/DMF/credit_rating_artifacts/dmf_dcs_financial_shap_importance_by_class.csv
COPY --chown=appuser:appuser artifacts/DMF/credit_rating_artifacts/dmf_dcs_financial_shap_local_decisions.csv \
    /workspace/artifacts/DMF/credit_rating_artifacts/dmf_dcs_financial_shap_local_decisions.csv
COPY --chown=appuser:appuser artifacts/DMF/credit_rating_artifacts/dmf_dcs_financial_lime_local_decisions.csv \
    /workspace/artifacts/DMF/credit_rating_artifacts/dmf_dcs_financial_lime_local_decisions.csv

RUN mkdir -p /workspace/app/backend/data \
    && chown -R appuser:appuser /workspace
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM runtime AS dev

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
