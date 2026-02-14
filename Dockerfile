FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Installation de Podman, Buildah et Python pour le serveur MCP
RUN apt-get update && apt-get install -y \
    podman \
    buildah \
    python3-pip \
    python3-dev \
    git \
    qemu-user-static \
    && rm -rf /var/lib/apt/lists/*

# Configuration Podman Rootless (essentiel pour HPC)
RUN echo "user.max_user_namespaces=28633" > /etc/sysctl.d/userns.conf
# Setup du stockage local pour les images dans le workspace du job
RUN mkdir -p /workspace/containers/storage
ENV CONTAINER_STORAGE_CONF=/workspace/containers/storage

RUN pip install "mcp[cli]" uv

COPY mcp_orchestrator.py /workspace/mcp_orchestrator.py
WORKDIR /workspace

CMD ["python3", "mcp_orchestrator.py"]