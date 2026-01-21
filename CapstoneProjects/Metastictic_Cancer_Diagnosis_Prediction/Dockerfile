
FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked


COPY predict.py model.pkl  ./ 

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]