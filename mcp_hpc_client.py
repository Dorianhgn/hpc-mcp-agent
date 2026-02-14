# mcp_hpc_client.py
import os
import json
import uuid
import time
import asyncio
from upstash_redis import Redis
from mcp.server.fastmcp import FastMCP

# Init FastMCP
mcp = FastMCP("HPC-Orchestrator")

# Redis client avec upstash_redis (REST API)
redis_client = Redis.from_env()

QUEUE_NAME = "hpc:jobs"
RESULTS_PREFIX = "hpc:result:"


def submit_job(job_type: str, **params) -> str:
    """Soumet un job dans la queue Redis et attend le résultat"""
    job_id = str(uuid.uuid4())
    
    job = {
        "id": job_id,
        "type": job_type,
        "timestamp": time.time(),
        **params
    }
    
    # Envoie dans la queue
    redis_client.lpush(QUEUE_NAME, json.dumps(job))
    print(f"📤 Job {job_id[:8]} submitted (type: {job_type})")
    
    # Attend le résultat (avec timeout adaptatif)
    timeout = {
        "podman_build": 600,      # 10 min
        "podman_run": 3600,       # 1h
        "huggingface_check": 30,  # 30s
        "gpu_info": 60,           # 1 min
        "slurm_queue": 30,        # 30s
    }.get(job_type, 300)
    
    result = wait_for_result(job_id, timeout)
    return result


def wait_for_result(job_id: str, timeout: int) -> str:
    """Poll Redis pour récupérer le résultat"""
    result_key = f"{RESULTS_PREFIX}{job_id}"
    
    for i in range(timeout):
        result_json = redis_client.get(result_key)
        
        if result_json:
            result = json.loads(result_json)
            
            if result.get("status") == "success":
                return result.get("output", "")
            else:
                error = result.get("error", "Unknown error")
                stderr = result.get("stderr", "")
                return f"❌ Job failed:\n{error}\n\nStderr:\n{stderr}"
        
        # Feedback toutes les 10s
        if i % 10 == 0 and i > 0:
            print(f"⏳ Still waiting for job {job_id[:8]}... ({i}s elapsed)")
        
        time.sleep(1)
    
    return f"⏱️ Timeout: Job {job_id} took longer than {timeout}s"


# ==================== TOOLS ====================

@mcp.tool()
def build_and_test_image(repo_url: str, dockerfile_content: str, tag: str) -> str:
    """
    Clone un repo, écrit un Dockerfile, build l'image avec Buildah (rootless)
    et tente un dry-run pour vérifier les dépendances.
    
    Args:
        repo_url: URL du repo Git à cloner
        dockerfile_content: Contenu du Dockerfile à créer
        tag: Tag de l'image (ex: 'mamba-jetson:v1')
    
    Returns:
        Build logs et résultat du dry-run
    """
    return submit_job(
        "podman_build",
        repo_url=repo_url,
        dockerfile_content=dockerfile_content,
        tag=tag
    )


@mcp.tool()
def run_benchmark_in_container(image_tag: str, command: str, gpus: int = 1) -> str:
    """
    Lance une commande dans un container avec accès GPU.
    
    Args:
        image_tag: Tag de l'image à utiliser
        command: Commande à exécuter (ex: 'python train.py --epochs 1')
        gpus: Nombre de GPUs à allouer (default: 1)
    
    Returns:
        Stdout de la commande
    """
    return submit_job(
        "podman_run",
        image_tag=image_tag,
        command=command,
        gpus=gpus
    )


@mcp.tool()
def run_script_on_hpc(script: str, partition: str = "dev", cpus: int = 8, 
                      mem: str = "64G", gpus: int = 1) -> str:
    """
    Exécute un script bash arbitraire sur le HPC via srun.
    
    Args:
        script: Script bash à exécuter
        partition: Partition SLURM (default: 'dev')
        cpus: Nombre de CPUs (default: 8)
        mem: Mémoire (default: '64G')
        gpus: Nombre de GPUs (default: 1)
    
    Returns:
        Output du script
    """
    return submit_job(
        "srun_script",
        script=script,
        partition=partition,
        cpus=cpus,
        mem=mem,
        gpus=gpus
    )


@mcp.tool()
def check_huggingface_model(model_id: str) -> str:
    """
    Interroge l'API HuggingFace pour récupérer les infos d'un modèle.
    
    Args:
        model_id: ID du modèle (ex: 'meta-llama/Llama-3.2-1B')
    
    Returns:
        Infos JSON du modèle (taille, safetensors, etc.)
    """
    return submit_job(
        "huggingface_check",
        model_id=model_id
    )


@mcp.tool()
def check_slurm_queue() -> str:
    """
    Affiche l'état de la queue SLURM (squeue).
    
    Returns:
        Output de squeue formaté
    """
    return submit_job("slurm_queue")


@mcp.tool()
def get_gpu_info() -> str:
    """
    Récupère les infos des GPUs disponibles (nvidia-smi).
    
    Returns:
        Output de nvidia-smi
    """
    return submit_job("gpu_info")


if __name__ == "__main__":
    mcp.run()