# Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Évite les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    cmake \
    build-essential \
    ninja-build \
    curl \
    jq \
    # Cross-compilation pour Jetson
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    # Podman & Buildah pour container builds
    podman \
    buildah \
    slirp4netns \
    fuse-overlayfs \
    && rm -rf /var/lib/apt/lists/*

# Configure Podman/Buildah en mode rootless
RUN mkdir -p /etc/containers && \
    echo '[storage]' > /etc/containers/storage.conf && \
    echo 'driver = "overlay"' >> /etc/containers/storage.conf && \
    echo 'runroot = "/run/containers/storage"' >> /etc/containers/storage.conf && \
    echo 'graphroot = "/var/lib/containers/storage"' >> /etc/containers/storage.conf

# Install Python packages
RUN pip3 install --no-cache-dir \
    redis \
    requests \
    mcp \
    fastmcp

# Create workspace
WORKDIR /workspace

# Copy worker script
COPY mcp_hpc_worker.py /app/mcp_hpc_worker.py

# Healthcheck (optionnel)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import redis, os; redis.Redis.from_url(os.getenv('REDIS_URL')).ping()" || exit 1

# Default command
CMD ["python3", "/app/mcp_hpc_worker.py", "worker-1"]