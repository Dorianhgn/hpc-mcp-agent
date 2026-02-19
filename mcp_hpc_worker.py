# mcp_hpc_worker.py
import os
import sys
import json
import time
import subprocess
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
        sys.stdout.flush()

    def run(self):
        """Boucle principale — 100% synchrone, pas d'asyncio"""
        while True:
            try:
                job_json = self.redis.rpop(self.queue_name)

                if job_json:
                    job = json.loads(job_json)
                    print(f"\n📥 Job {job['id'][:8]}: {job['type']}")
                    sys.stdout.flush()

                    result = self.execute_job(job)

                    result_key = f"{self.results_prefix}{job['id']}"
                    self.redis.set(result_key, json.dumps(result), ex=3600)

                    status = "✅" if result['status'] == 'success' else "❌"
                    print(f"{status} Job {job['id'][:8]} done ({result.get('duration', 0)}s)")
                    sys.stdout.flush()

                else:
                    print("💤 Idle...", end='\r')
                    sys.stdout.flush()
                    time.sleep(5)

            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                sys.stdout.flush()
                time.sleep(5)

    def execute_job(self, job):
        """Route vers le bon handler"""
        start = time.time()

        handlers = {
            "podman_build":       self.handle_podman_build,
            "podman_run":         self.handle_podman_run,
            "srun_script":        self.handle_srun_script,
            "huggingface_check":  self.handle_huggingface_check,
            "slurm_queue":        self.handle_slurm_queue,
            "gpu_info":           self.handle_gpu_info,
        }

        handler = handlers.get(job['type'])
        if not handler:
            return {
                "status": "failed",
                "error": f"Unknown job type: {job['type']}"
            }

        try:
            result = handler(job)
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

    def handle_podman_build(self, job):
        repo_url    = job['repo_url']
        dockerfile  = job['dockerfile_content']
        tag         = job['tag']
        work_dir    = f"/tmp/build_{job['id'][:8]}"

        script = f"""#!/bin/bash
set -e
git clone {repo_url} {work_dir}
cd {work_dir}

cat > Dockerfile.ai << 'DOCKERFILE_EOF'
{dockerfile}
DOCKERFILE_EOF

buildah bud --isolation chroot -t {tag} -f Dockerfile.ai .

echo "Testing image..."
buildah from --name test-{job['id'][:8]} {tag}
buildah run --isolation chroot test-{job['id'][:8]} python3 --version 2>&1 || echo "No python in image"
buildah rm test-{job['id'][:8]} 2>/dev/null || true
echo "✅ Build complete: {tag}"
"""
        return self._run_bash(script)

    def handle_podman_run(self, job):
        image_tag = job['image_tag']
        command   = job['command']
        gpus      = job.get('gpus', 1)

        gpu_mounts = self._detect_gpu_mounts() if gpus > 0 else ""

        script = f"""#!/bin/bash
set -e
echo "Running in container: {image_tag}"
echo "Command: {command}"

ctr=$(buildah from {image_tag})

# Execute /bin/bash inside the container and feed it the command via standard input.
# This bypasses the $PATH issue and safely handles any messy quotes in the command string.
buildah run --isolation chroot {gpu_mounts} $ctr -- /bin/bash << 'EOF_CMD'
{command}
EOF_CMD

buildah rm $ctr
"""
        return self._run_bash(script)

    def handle_srun_script(self, job):
        # The worker already runs inside a SLURM-allocated container (via Pyxis/Enroot).
        # srun is not available inside the container — execute the script directly.
        script = job['script']

        wrapped = f"""#!/bin/bash
{script}
"""
        return self._run_bash(wrapped)

    def handle_huggingface_check(self, job):
        model_id = job['model_id']
        script = f'curl -s "https://huggingface.co/api/models/{model_id}" | python3 -m json.tool'
        return self._run_bash(script)

    def handle_slurm_queue(self, job):
        return self._run_bash("squeue -u $USER")

    def handle_gpu_info(self, job):
        return self._run_bash("nvidia-smi")

    # ==================== UTILS ====================

    def _detect_gpu_mounts(self) -> str:
        """
        Auto-detect GPU device nodes and host driver libs by parsing Pyxis mounts.
        """
        import glob
        import os
        
        mounts = []

        # 1. Mount the entire /dev so all nvidia device nodes are accessible
        mounts.append("-v /dev:/dev")

        # 2. Parse /proc/mounts to find exactly what Slurm/Pyxis injected
        try:
            with open("/proc/mounts", "r") as f:
                for line in f:
                    # Look for nvidia binaries or libraries injected by the HPC
                    if "nvidia" in line.lower() and ("/usr/bin/" in line or "/usr/lib/" in line):
                        parts = line.split()
                        if len(parts) >= 2:
                            mount_point = parts[1]
                            if os.path.exists(mount_point) and not os.path.isdir(mount_point):
                                mounts.append(f"-v {mount_point}:{mount_point}")
        except Exception as e:
            print(f"Warning: Could not parse /proc/mounts: {e}")

        # 3. Explicitly grab essential symlinks (Pyxis sometimes mounts the raw versioned file, 
        # but PyTorch/nvidia-smi look for the .so.1 or .so symlinks)
        extra_libs = (
            glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + 
            glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so*")
        )
        for lib in extra_libs:
            if os.path.exists(lib):
                mounts.append(f"-v {lib}:{lib}")

        # 4. Ensure the nvidia-smi binary is included
        smi_path = "/usr/bin/nvidia-smi"
        if os.path.exists(smi_path):
            mounts.append(f"-v {smi_path}:{smi_path}")

        # Remove duplicates and return as a single string
        unique_mounts = list(dict.fromkeys(mounts))
        return " ".join(unique_mounts)

    def _run_bash(self, script: str):
        """Exécute un script bash de façon synchrone"""
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True
        )

        return {
            "status":     "success" if result.returncode == 0 else "failed",
            "output":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode
        }


if __name__ == "__main__":
    worker_id = sys.argv[1] if len(sys.argv) > 1 else "worker-1"
    worker = HPCWorker(worker_id)
    worker.run()