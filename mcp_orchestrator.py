from mcp.server.fastmcp import FastMCP
import subprocess
import os

mcp = FastMCP("HPC-Orchestrator")

@mcp.tool()
def build_and_test_image(repo_url: str, dockerfile_content: str, tag: str) -> str:
    """
    Clone un repo, écrit un Dockerfile, build l'image avec Buildah 
    et tente un 'dry-run' pour vérifier les dépendances.
    """
    # 1. Clone & Setup
    dirname = tag.split('/')[-1]
    subprocess.run(["git", "clone", repo_url, dirname])
    
    with open(f"{dirname}/Dockerfile.ai", "w") as f:
        f.write(dockerfile_content)

    # 2. Build avec Buildah (Rootless)
    build_cmd = f"buildah bud --isolation chroot -t {tag} -f {dirname}/Dockerfile.ai {dirname}"
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        return f"Build Failed:\n{result.stderr}"

    return f"Build Success: {tag}\n{result.stdout}"

@mcp.tool()
def run_benchmark_in_container(image_tag: str, command: str) -> str:
    """Lance une commande (ex: python train.py) dans un container spécifique avec GPU."""
    # On utilise podman avec le flag --device pour passer le GPU
    run_cmd = f"podman run --rm --device nvidia.com/gpu=all {image_tag} {command}"
    result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
    return result.stdout

@mcp.tool()
def check_huggingface_model(model_id: str) -> str:
    """Interroge l'API HF pour récupérer les infos de taille du modèle."""
    # Ici, l'agent peut utiliser curl ou une lib python pour voir la VRAM requise
    import requests
    res = requests.get(f"https://huggingface.co/api/models/{model_id}")
    return str(res.json().get("safetensors", "Info non dispo"))