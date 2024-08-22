# Stage 1: Build
FROM --platform=linux/amd64 python:3.11.8-slim as build
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY . .

RUN chmod u+x setup.sh
CMD ["./setup.sh"]

# Install CPU-optimized PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime
FROM --platform=linux/amd64 python:3.11.8-slim
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /app /app


# Set entrypoint
ENTRYPOINT ["uvicorn", "main:app", "--reload"]