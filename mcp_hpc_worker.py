# mcp_hpc_worker.py
import os
import json
import time
import subprocess
import asyncio
from upstash_redis import Redis
from datetime import datetime

class HPCWorker:
    def __init__(self, worker_id="worker-1"):
        self.worker_id = worker_id
        self.redis = Redis.from_env()
        self.queue_name = "hpc:jobs"
        self.results_prefix = "hpc:result:"
        
        print(f"🚀 Worker {worker_id} started")
        print(f"📡 Connected to Upstash Redis")
    
    async def run(self):
        """Boucle principale"""
        while True:
            try:
                # Poll avec timeout de 5s (upstash_redis ne supporte pas BRPOP avec timeout)
                # Donc on fait du polling classique
                job_json = self.redis.rpop(self.queue_name)
                
                if job_json:
                    job = json.loads(job_json)
                    
                    print(f"\n📥 Job {job['id'][:8]}: {job['type']}")
                    
                    # Exécute
                    result = await self.execute_job(job)
                    
                    # Stocke résultat (expire après 1h = 3600s)
                    result_key = f"{self.results_prefix}{job['id']}"
                    self.redis.set(result_key, json.dumps(result), ex=3600)
                    
                    status = "✅" if result['status'] == 'success' else "❌"
                    print(f"{status} Job {job['id'][:8]} done ({result.get('duration', 0)}s)")
                
                else:
                    # Aucun job, on attend un peu
                    print("💤 Idle...", end='\r')
                    await asyncio.sleep(5)
            
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(5)
    
    async def execute_job(self, job):
        """Route vers le bon handler"""
        start = time.time()
        
        handlers = {
            "podman_build": self.handle_podman_build,
            "podman_run": self.handle_podman_run,
            "srun_script": self.handle_srun_script,
            "huggingface_check": self.handle_huggingface_check,
            "slurm_queue": self.handle_slurm_queue,
            "gpu_info": self.handle_gpu_info,
        }
        
        handler = handlers.get(job['type'])
        if not handler:
            return {
                "status": "failed",
                "error": f"Unknown job type: {job['type']}"
            }
        
        try:
            result = await handler(job)
            result['duration'] = round(time.time() - start, 2)
            result['worker_id'] = self.worker_id
            return result
        
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "duration": round(time.time() - start, 2)
            }
    
    # ==================== HANDLERS ====================
    
    async def handle_podman_build(self, job):
        """Build une image avec Buildah"""
        repo_url = job['repo_url']
        dockerfile_content = job['dockerfile_content']
        tag = job['tag']
        
        dirname = tag.split('/')[-1].replace(':', '_')
        work_dir = f"/tmp/build_{job['id'][:8]}"
        
        script = f"""#!/bin/bash
set -e

# Clone
git clone {repo_url} {work_dir}
cd {work_dir}

# Dockerfile
cat > Dockerfile.ai << 'DOCKERFILE_EOF'
{dockerfile_content}
DOCKERFILE_EOF

# Build avec Buildah (rootless)
buildah bud --isolation chroot -t {tag} -f Dockerfile.ai .

# Test dry-run
echo "Testing image..."
podman run --rm {tag} python --version || echo "No python in image"

echo "✅ Build complete: {tag}"
"""
        
        return await self._run_bash(script)
    
    async def handle_podman_run(self, job):
        """Run une commande dans un container"""
        image_tag = job['image_tag']
        command = job['command']
        gpus = job.get('gpus', 1)
        
        gpu_flags = f"--device nvidia.com/gpu=all" if gpus > 0 else ""
        
        script = f"""#!/bin/bash
set -e

echo "Running in container: {image_tag}"
echo "Command: {command}"

podman run --rm {gpu_flags} {image_tag} {command}
"""
        
        return await self._run_bash(script)
    
    async def handle_srun_script(self, job):
        """Exécute un script via srun"""
        script = job['script']
        partition = job.get('partition', 'dev')
        cpus = job.get('cpus', 8)
        mem = job.get('mem', '64G')
        gpus = job.get('gpus', 1)
        
        wrapped = f"""#!/bin/bash
srun -p {partition} -c {cpus} --mem={mem} --gres=gpu:{gpus} \\
     --time=0-4:00:00 \\
     bash -c {repr(script)}
"""
        
        return await self._run_bash(wrapped)
    
    async def handle_huggingface_check(self, job):
        """Check un modèle HF"""
        model_id = job['model_id']
        
        script = f"""#!/bin/bash
curl -s "https://huggingface.co/api/models/{model_id}" | jq .
"""
        
        return await self._run_bash(script)
    
    async def handle_slurm_queue(self, job):
        """Affiche la queue SLURM"""
        return await self._run_bash("squeue -u $USER")
    
    async def handle_gpu_info(self, job):
        """Affiche nvidia-smi"""
        return await self._run_bash("nvidia-smi")
    
    # ==================== UTILS ====================
    
    async def _run_bash(self, script: str):
        """Exécute un script bash"""
        proc = await asyncio.create_subprocess_shell(
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        return {
            "status": "success" if proc.returncode == 0 else "failed",
            "output": stdout.decode('utf-8', errors='replace'),
            "stderr": stderr.decode('utf-8', errors='replace'),
            "returncode": proc.returncode
        }


if __name__ == "__main__":
    import sys
    worker_id = sys.argv[1] if len(sys.argv) > 1 else "worker-1"
    worker = HPCWorker(worker_id)
    asyncio.run(worker.run())