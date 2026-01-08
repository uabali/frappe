import requests
import threading
import time
import json
import statistics
import subprocess
import argparse
import os
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_URL = "http://localhost:8088/query"
VLLM_URL = "http://localhost:8282/health"

QUESTIONS = [
    "Agile nedir?",
    "Scrum nedir?",
    "Yazilim mimarisi neden onemlidir?",
    "Mikroservis avantajlari nelerdir?",
    "Monolitik ve mikroservis mimarisi arasindaki farklar nelerdir?",
    "Guvenli yazilim gelistirme sureci nasil olmalidir?",
]

LOAD_LEVELS = {
    "smoke": [1, 2],
    "load": [5, 10, 20, 30],
    "stress": [50, 75, 100],
    "spike": [150, 200],
}

WARMUP_REQUESTS = 7
COOLDOWN_SECONDS = 3
REPEAT_COUNT = 3

@dataclass
class GPUInfo:
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
    temperature_c: int
    power_watts: float

def get_gpu_info() -> Optional[GPUInfo]:
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        
        parts = result.stdout.strip().split(", ")
        return GPUInfo(
            name=parts[0].strip(),
            memory_total_mb=int(float(parts[1])),
            memory_used_mb=int(float(parts[2])),
            memory_free_mb=int(float(parts[3])),
            utilization_percent=int(float(parts[4])),
            temperature_c=int(float(parts[5])),
            power_watts=float(parts[6])
        )
    except Exception:
        return None

def percentile(data: List[float], p: float) -> float:
    """Calculate percentile"""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)

def calculate_apdex(times: List[float], threshold: float = 5.0) -> float:
    """Calculate Apdex score (0.0-1.0)"""
    if not times:
        return 0
    satisfied = sum(1 for t in times if t <= threshold)
    tolerating = sum(1 for t in times if threshold < t <= 4 * threshold)
    return round((satisfied + tolerating * 0.5) / len(times), 3)

def calculate_metrics(times: List[float], total_time: float, num_users: int, successful: int) -> dict:
    """Calculate comprehensive metrics"""
    if not times:
        return {}
    
    return {
        "avg_latency_s": round(statistics.mean(times), 3),
        "min_latency_s": round(min(times), 3),
        "max_latency_s": round(max(times), 3),
        "std_dev_s": round(statistics.stdev(times), 3) if len(times) > 1 else 0,
        "p50_s": round(percentile(times, 50), 3),
        "p75_s": round(percentile(times, 75), 3),
        "p90_s": round(percentile(times, 90), 3),
        "p95_s": round(percentile(times, 95), 3),
        "p99_s": round(percentile(times, 99), 3),
        "throughput_rps": round(successful / total_time, 2) if total_time > 0 else 0,
        "total_time_s": round(total_time, 2),
        "success_rate_pct": round(successful / num_users * 100, 2),
        "error_rate_pct": round((num_users - successful) / num_users * 100, 2),
        "concurrency_factor": round(sum(times) / total_time, 2) if total_time > 0 else 0,
        "apdex_score": calculate_apdex(times)
    }

results = []

def send_request(user_id: int, question: str):
    """Send single request and record result"""
    start = time.time()
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=120)
        elapsed = time.time() - start
        data = response.json()
        api_metrics = data.get("metrics", {})
        
        results.append({
            "user": user_id,
            "question": question,
            "success": True,
            "status": response.status_code,
            "total_time": round(elapsed, 3),
            "retrieval_time": api_metrics.get("retrieval_time", 0),
            "llm_time": api_metrics.get("llm_time", 0),
            "answer_length": len(data.get("answer", ""))
        })
    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "user": user_id,
            "question": question,
            "success": False,
            "status": "error",
            "total_time": round(elapsed, 3),
            "error": str(e)
        })

def run_test(num_users: int, questions: List[str]) -> dict:
    """Run single concurrent test"""
    results.clear()
    threads = []

    shuffled_questions = questions[:]
    random.shuffle(shuffled_questions)
    
    start_time = time.time()
    
    for i in range(num_users):
        question = shuffled_questions[i % len(shuffled_questions)]
        t = threading.Thread(target=send_request, args=(i+1, question))
        threads.append(t)
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    times = [r["total_time"] for r in results if r["success"]]
    
    metrics = calculate_metrics(times, total_time, num_users, successful)
    
    retrieval_times = [r["retrieval_time"] for r in results if r["success"] and r["retrieval_time"] > 0]
    llm_times = [r["llm_time"] for r in results if r["success"] and r["llm_time"] > 0]
    
    if retrieval_times:
        metrics["avg_retrieval_time_s"] = round(statistics.mean(retrieval_times), 3)
    if llm_times:
        metrics["avg_llm_time_s"] = round(statistics.mean(llm_times), 3)
    
    return {
        "user_count": num_users,
        "successful": successful,
        "failed": num_users - successful,
        "metrics": metrics,
        "raw_times": times
    }

def warmup():
    """Warmup the system before benchmarking"""
    print("Warming up system...")
    for i in range(WARMUP_REQUESTS):
        try:
            requests.post(API_URL, json={"question": QUESTIONS[0]}, timeout=60)
            print(f"  Warmup {i+1}/{WARMUP_REQUESTS} complete")
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")
    time.sleep(2)
    print("Warmup complete\n")

def generate_summary(results: List[dict]) -> dict:
    """Generate summary statistics"""
    if not results:
        return {}
    
    best_throughput = max(results, key=lambda x: x["metrics"]["throughput_rps"])
    best_latency = min(results, key=lambda x: x["metrics"]["avg_latency_s"])
    
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        latency_increase = (last["metrics"]["avg_latency_s"] - first["metrics"]["avg_latency_s"]) / first["metrics"]["avg_latency_s"] * 100
    else:
        latency_increase = 0
    
    return {
        "peak_throughput_rps": best_throughput["metrics"]["throughput_rps"],
        "peak_throughput_at_users": best_throughput["user_count"],
        "best_latency_s": best_latency["metrics"]["avg_latency_s"],
        "best_latency_at_users": best_latency["user_count"],
        "latency_degradation_pct": round(latency_increase, 1),
        "max_tested_users": results[-1]["user_count"],
        "all_tests_passed": all(r["total_successful"] == r["total_requests"] for r in results)
    }

def print_summary(report: dict):
    """Print formatted summary"""
    summary = report["summary"]
    
    print(f"\n{'-'*70}")
    print("SUMMARY")
    print(f"{'-'*70}")
    print(f"  Peak Throughput:     {summary['peak_throughput_rps']:.2f} req/s @ {summary['peak_throughput_at_users']} users")
    print(f"  Best Latency:        {summary['best_latency_s']:.2f}s @ {summary['best_latency_at_users']} users")
    print(f"  Max Tested Users:    {summary['max_tested_users']}")
    print(f"  Latency Degradation: {summary['latency_degradation_pct']:.1f}%")
    print(f"  All Tests Passed:    {'Yes' if summary['all_tests_passed'] else 'No'}")
    
    print(f"\n{'-'*70}")
    print("COMPARISON TABLE")
    print(f"{'-'*70}")
    print(f"{'Users':>6} | {'Throughput':>12} | {'Avg':>8} | {'P50':>8} | {'P95':>8} | {'P99':>8} | {'Apdex':>6}")
    print(f"{'-'*70}")
    
    for r in report["results"]:
        m = r["metrics"]
        print(f"{r['user_count']:>6} | {m['throughput_rps']:>10.2f}/s | {m['avg_latency_s']:>7.2f}s | "
              f"{m['p50_s']:>7.2f}s | {m['p95_s']:>7.2f}s | {m['p99_s']:>7.2f}s | {m['apdex_score']:>6.3f}")

def run_benchmark(gpu_name: str = None, test_type: str = "load"):
    """Run complete benchmark suite"""
    gpu_info = get_gpu_info()
    if gpu_info:
        gpu_name = gpu_name or gpu_info.name
        print(f"GPU: {gpu_info.name}")
        print(f"VRAM: {gpu_info.memory_total_mb} MB")
    else:
        gpu_name = gpu_name or "Unknown_GPU"
        print("GPU: Could not detect (nvidia-smi not available)")
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {test_type.upper()} TEST")
    print(f"{'='*70}")
    print(f"GPU: {gpu_name}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        requests.get(f"{API_URL.replace('/query', '/health')}", timeout=5)
        print("RAG API: OK")
    except:
        print("ERROR: RAG API not running!")
        return None
    
    try:
        requests.get(VLLM_URL, timeout=5)
        print("vLLM: OK")
    except:
        print("WARNING: vLLM health check failed")
    
    print()
    warmup()
    
    load_levels = LOAD_LEVELS.get(test_type, LOAD_LEVELS["load"])
    all_results = []
    
    for num_users in load_levels:
        print(f"\n{'-'*70}")
        print(f"Testing: {num_users} concurrent users")
        print(f"{'-'*70}")
        
        run_results = []
        
        for run in range(REPEAT_COUNT):
            print(f"  Run {run+1}/{REPEAT_COUNT}...", end=" ", flush=True)
            result = run_test(num_users, QUESTIONS)
            run_results.append(result)
            print(f"Throughput: {result['metrics']['throughput_rps']:.2f} req/s, "
                  f"Avg: {result['metrics']['avg_latency_s']:.2f}s")
            time.sleep(COOLDOWN_SECONDS)
        
        all_times = []
        for r in run_results:
            all_times.extend(r["raw_times"])
        
        total_successful = sum(r["successful"] for r in run_results)
        total_requests = num_users * REPEAT_COUNT
        total_time = sum(r["metrics"]["total_time_s"] for r in run_results)
        
        aggregated_metrics = calculate_metrics(all_times, total_time, total_requests, total_successful)
        
        all_results.append({
            "user_count": num_users,
            "total_requests": total_requests,
            "total_successful": total_successful,
            "metrics": aggregated_metrics
        })
        
        print(f"\n  Aggregated: Throughput={aggregated_metrics['throughput_rps']:.2f} req/s, "
              f"Avg={aggregated_metrics['avg_latency_s']:.2f}s, "
              f"P95={aggregated_metrics['p95_s']:.2f}s")
    
    gpu_info_final = get_gpu_info()
    
    report = {
        "benchmark_info": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": test_type,
            "gpu_name": gpu_name,
            "repeat_count": REPEAT_COUNT,
            "warmup_requests": WARMUP_REQUESTS,
            "question_count": len(QUESTIONS)
        },
        "gpu_info": asdict(gpu_info) if gpu_info else None,
        "gpu_after_test": asdict(gpu_info_final) if gpu_info_final else None,
        "results": all_results,
        "summary": generate_summary(all_results)
    }
    
    safe_gpu_name = gpu_name.replace(' ', '_').replace('/', '_')
    filename = RESULTS_DIR / f"benchmark_{safe_gpu_name}_{test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved: {filename}")
    
    print_summary(report)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Benchmark for RAG Performance")
    parser.add_argument("--gpu-name", type=str, help="Override GPU name")
    parser.add_argument("--test-type", type=str, default="load",
                        choices=["smoke", "load", "stress", "spike"],
                        help="Test type")
    args = parser.parse_args()
    
    run_benchmark(gpu_name=args.gpu_name, test_type=args.test_type)
