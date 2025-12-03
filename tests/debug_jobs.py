#!/usr/bin/env python3
"""
Debug stuck jobs by checking their status directly
"""

import requests
import json

def debug_stuck_jobs():
    """Debug jobs that are stuck in processing"""
    base_url = "http://localhost:8080"
    
    print("🔍 Debugging stuck jobs...\n")
    
    # Get health status first
    try:
        health = requests.get(f"{base_url}/health").json()
        print(f"📊 Server health: {health}")
        print(f"📊 Total jobs in memory: {health.get('total_jobs', 0)}\n")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Try to get job status for known stuck job IDs from logs
    stuck_job_ids = [
        "10671fbc-4b28-4d21-a25c-cb307e2c6c52",
        "96a79212-523d-4a5b-95ef-e45674dadd45"
    ]
    
    for job_id in stuck_job_ids:
        print(f"🔍 Checking job {job_id}:")
        try:
            response = requests.get(f"{base_url}/v1/jobs/{job_id}")
            if response.status_code == 200:
                job_data = response.json()
                print(f"   Status: {job_data.get('status')}")
                print(f"   Progress: {job_data.get('progress')}")
                print(f"   Error: {job_data.get('error')}")
            else:
                print(f"   ❌ HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()

if __name__ == "__main__":
    debug_stuck_jobs()