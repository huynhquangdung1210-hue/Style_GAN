#!/usr/bin/env python3
"""
Load testing script for the style transfer API.
Tests performance under concurrent load.
"""

import asyncio
import aiohttp
import base64
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import argparse

@dataclass
class TestResult:
    """Test result data."""
    duration: float
    status_code: int
    success: bool
    error: str = ""

class LoadTester:
    def __init__(self, base_url: str, concurrent_users: int, test_duration: int):
        self.base_url = base_url
        self.concurrent_users = concurrent_users
        self.test_duration = test_duration
        self.results: List[TestResult] = []
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    async def single_request(self, session: aiohttp.ClientSession, subject_b64: str, style_b64: str) -> TestResult:
        """Perform a single style transfer request."""
        start_time = time.time()
        
        try:
            payload = {
                "subject_image": subject_b64,
                "style_image": style_b64,
                "quality": "medium",  # Use medium for faster processing
                "style_strength": 0.7,
                "num_inference_steps": 15
            }
            
            async with session.post(
                f"{self.base_url}/v1/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                duration = time.time() - start_time
                
                if response.status == 202:
                    data = await response.json()
                    return TestResult(
                        duration=duration,
                        status_code=response.status,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        duration=duration,
                        status_code=response.status,
                        success=False,
                        error=error_text[:100]  # Truncate error
                    )
                    
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return TestResult(
                duration=duration,
                status_code=0,
                success=False,
                error="Timeout"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                duration=duration,
                status_code=0,
                success=False,
                error=str(e)[:100]
            )
    
    async def user_simulation(self, user_id: int, subject_b64: str, style_b64: str):
        """Simulate a single user making requests."""
        print(f"🤖 Starting user {user_id}")
        
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            end_time = time.time() + self.test_duration
            request_count = 0
            
            while time.time() < end_time:
                result = await self.single_request(session, subject_b64, style_b64)
                self.results.append(result)
                request_count += 1
                
                # Wait a bit before next request
                await asyncio.sleep(1)
        
        print(f"👤 User {user_id} completed {request_count} requests")
    
    async def run_load_test(self, subject_path: str, style_path: str):
        """Run the load test."""
        print(f"🔥 Starting load test:")
        print(f"   Concurrent Users: {self.concurrent_users}")
        print(f"   Duration: {self.test_duration} seconds")
        print(f"   Target: {self.base_url}")
        print()
        
        # Encode images once
        subject_b64 = self.encode_image_to_base64(subject_path)
        style_b64 = self.encode_image_to_base64(style_path)
        
        # Start all user simulations
        tasks = []
        start_time = time.time()
        
        for user_id in range(self.concurrent_users):
            task = asyncio.create_task(
                self.user_simulation(user_id, subject_b64, style_b64)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total test time: {total_time:.2f} seconds")
        
        self.print_results()
    
    def print_results(self):
        """Print test results and statistics."""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100
        
        durations = [r.duration for r in self.results]
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        p95_duration = statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max_duration
        
        # Group errors
        error_counts = {}
        for result in self.results:
            if not result.success:
                error = result.error or f"HTTP {result.status_code}"
                error_counts[error] = error_counts.get(error, 0) + 1
        
        print("📊 Load Test Results:")
        print("=" * 50)
        print(f"Total Requests:      {total_requests}")
        print(f"Successful:          {successful_requests}")
        print(f"Failed:              {failed_requests}")
        print(f"Success Rate:        {success_rate:.1f}%")
        print()
        print("⏱️  Response Times:")
        print(f"Average:             {avg_duration:.3f}s")
        print(f"Minimum:             {min_duration:.3f}s")
        print(f"Maximum:             {max_duration:.3f}s")
        print(f"95th Percentile:     {p95_duration:.3f}s")
        print()
        
        if error_counts:
            print("❌ Error Summary:")
            for error, count in error_counts.items():
                print(f"  {error}: {count}")
        
        print()
        
        # Performance assessment
        if success_rate >= 95:
            print("🎉 Excellent performance!")
        elif success_rate >= 85:
            print("✅ Good performance")
        elif success_rate >= 70:
            print("⚠️  Acceptable performance")
        else:
            print("🔴 Poor performance - investigate issues")

async def main():
    parser = argparse.ArgumentParser(description="Load test the Style Transfer API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8080",
        help="Base URL of the API (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--concurrent-users", 
        type=int, 
        default=5,
        help="Number of concurrent users (default: 5)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="Test duration in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Check for test images
    test_dir = Path(__file__).parent
    subject_image = test_dir / "test_images" / "subject.jpg"
    style_image = test_dir / "test_images" / "style.jpg"
    
    if not subject_image.exists() or not style_image.exists():
        print("❌ Test images not found. Please add test images to tests/test_images/")
        print("   Required: subject.jpg, style.jpg")
        return False
    
    # Run load test
    tester = LoadTester(
        base_url=args.url,
        concurrent_users=args.concurrent_users,
        test_duration=args.duration
    )
    
    await tester.run_load_test(str(subject_image), str(style_image))

if __name__ == "__main__":
    asyncio.run(main())