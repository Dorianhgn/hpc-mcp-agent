# test_simple.py
import os
from dotenv import load_dotenv
load_dotenv()

from upstash_redis import Redis

redis = Redis.from_env()

# Test 1: Ping
print("🧪 Test 1: Ping Redis")
redis.set('test', 'hello')
value = redis.get('test')
print(f"✅ Result: {value}")

# Test 2: Submit un fake job
import json
import uuid

job = {
    "id": str(uuid.uuid4()),
    "type": "gpu_info",
    "timestamp": 1234567890
}

print("\n🧪 Test 2: Submit job")
redis.lpush("hpc:jobs", json.dumps(job))
print(f"✅ Job submitted: {job['id'][:8]}")

# Test 3: Check la queue
print("\n🧪 Test 3: Check queue")
length = redis.llen("hpc:jobs")
print(f"✅ Queue length: {length}")