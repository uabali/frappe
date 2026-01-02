import requests
import threading
import time
import json
import statistics
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_URL = "http://localhost:8000/query"

QUESTIONS = [
    "Agile nedir?",
    "Scrum nedir?",
    "Yazilim mimarisi neden onemlidir?",
    "Mikroservis avantajlari nelerdir?",
    "Monolitik ve mikroservis mimarisi arasindaki farklar nelerdir?",
    "Guvenli yazilim gelistirme sureci nasil olmalidir?",
]

results = []
all_test_results = []

def percentile(data, p):
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

def send_request(user_id, question):
    start = time.time()
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=120)
        elapsed = time.time() - start
        data = response.json()
        results.append({
            "user": user_id,
            "question": question,
            "answer": data.get("answer", ""),
            "status": response.status_code,
            "time": round(elapsed, 2),
            "success": True
        })
        print(f"User {user_id}: {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        results.append({
            "user": user_id,
            "question": question,
            "status": "error",
            "time": round(elapsed, 2),
            "success": False,
            "error": str(e)
        })
        print(f"User {user_id}: ERROR - {e}")

def test_concurrent_users(num_users):
    print(f"\n{'='*60}")
    print(f"TEST: {num_users} concurrent users")
    print(f"{'='*60}")
    
    results.clear()
    threads = []
    start_time = time.time()
    
    for i in range(num_users):
        question = QUESTIONS[i % len(QUESTIONS)]
        t = threading.Thread(target=send_request, args=(i+1, question))
        threads.append(t)
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    failed = num_users - successful
    times = [r["time"] for r in results if r["success"]]
    
    avg_time = sum(times) / len(times) if times else 0
    throughput = successful / total_time if total_time > 0 else 0
    
    if times:
        p50 = percentile(times, 50)
        p95 = percentile(times, 95)
        p99 = percentile(times, 99)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
    else:
        p50 = p95 = p99 = std_dev = min_time = max_time = 0
    
    print(f"\n{'-'*60}")
    print(f"RESULTS")
    print(f"{'-'*60}")
    print(f"  Total time:          {total_time:.2f}s")
    print(f"  Successful:          {successful}/{num_users} ({successful/num_users*100:.1f}%)")
    print(f"  Throughput:          {throughput:.2f} req/s")
    print(f"  Average:             {avg_time:.2f}s")
    print(f"  P50/P95/P99:         {p50:.2f}s / {p95:.2f}s / {p99:.2f}s")
    
    test_result = {
        "user_count": num_users,
        "total_time": round(total_time, 2),
        "successful": successful,
        "failed": failed,
        "success_rate": round(successful/num_users*100, 1),
        "avg_response_time": round(avg_time, 2),
        "throughput": round(throughput, 2),
        "min_time": round(min_time, 2),
        "max_time": round(max_time, 2),
        "std_dev": round(std_dev, 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "p99": round(p99, 2)
    }
    all_test_results.append(test_result)
    
    return successful == num_users

def print_summary():
    """Print summary table"""
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Users':>8} | {'Success':>8} | {'Throughput':>12} | {'Avg':>8} | {'P50':>8} | {'P95':>8} | {'P99':>8}")
    print(f"{'-'*80}")
    
    for t in all_test_results:
        print(f"{t['user_count']:>8} | {t['success_rate']:>7.1f}% | {t['throughput']:>10.2f}/s | "
              f"{t['avg_response_time']:>7.2f}s | {t['p50']:>7.2f}s | {t['p95']:>7.2f}s | {t['p99']:>7.2f}s")

def save_results():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"test_results_{timestamp}.json"
    
    total_requests = sum(t['user_count'] for t in all_test_results)
    total_successful = sum(t['successful'] for t in all_test_results)
    avg_throughput = sum(t['throughput'] for t in all_test_results) / len(all_test_results) if all_test_results else 0
    
    report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_tests": len(all_test_results),
            "total_requests": total_requests,
            "total_successful": total_successful,
            "overall_success_rate": round(total_successful/total_requests*100, 1) if total_requests > 0 else 0,
            "avg_throughput": round(avg_throughput, 2)
        },
        "tests": all_test_results
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved: {filename}")
    return filename

if __name__ == "__main__":
    print("="*60)
    print("CONCURRENT USER PERFORMANCE TEST")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    
    try:
        requests.get("http://localhost:8000/health", timeout=5)
        print("Server is running\n")
    except:
        print("Server is not running! First run 'python main.py'")
        exit(1)
    
    test_scenarios = [3, 9, 27, 82, 100]
    
    for num_users in test_scenarios:
        test_concurrent_users(num_users)
        time.sleep(2)
    
    print_summary()
    save_results()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
