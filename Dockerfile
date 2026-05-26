# Base image: a slim Linux with Python 3.14 pre-installed.
FROM python:3.14-slim

# Install uv, the package manager used by the project.
RUN pip install --no-cache-dir uv

# Set the working directory inside the image.
WORKDIR /app

# Copy dependency files first, so this layer is cached when only code changes.
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into the image.
RUN uv sync --frozen --no-dev

# Copy the rest of the project.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY chroma_db/ ./chroma_db/
COPY eval/ ./eval/

# Gradio serves on port 7860 by default.
EXPOSE 7860

# Launch the demo when the container starts.
CMD ["uv", "run", "python", "scripts/app.py"]
