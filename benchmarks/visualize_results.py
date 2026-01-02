import json
import sys
import os
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def find_latest_results():
    """Find the most recent test results file"""
    patterns = [
        str(RESULTS_DIR / "test_results_*.json"),
        str(RESULTS_DIR / "benchmark_*.json"),
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_results(filename):
    """Load test results from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_metric(t, key, alt_key=None):
    """Get metric from test dict, checking both formats"""
    if 'metrics' in t:
        key_map = {
            'throughput': 'throughput_rps',
            'avg_response_time': 'avg_latency_s',
            'p50': 'p50_s', 'p95': 'p95_s', 'p99': 'p99_s',
            'max_time': 'max_latency_s', 'min_time': 'min_latency_s',
            'success_rate': 'success_rate_pct'
        }
        return t['metrics'].get(key_map.get(key, key), 0)
    return t.get(key, t.get(alt_key, 0) if alt_key else 0)

def print_text_report(data, filename):
    """Generate a text-based report"""
    print("\n" + "="*70)
    print("PERFORMANCE TEST REPORT")
    print("="*70)
    
    is_benchmark = 'benchmark_info' in data
    
    if is_benchmark:
        info = data['benchmark_info']
        print(f"Test Date: {info['date']}")
        print(f"GPU: {info['gpu_name']}")
        print(f"Test Type: {info['test_type']}")
    else:
        print(f"Test Date: {data['date']}")
    print(f"File: {filename}")
    
    if 'summary' in data:
        summary = data['summary']
        print(f"\n{'-'*70}")
        print("OVERALL SUMMARY")
        print(f"{'-'*70}")
        if is_benchmark:
            print(f"  Peak Throughput:     {summary['peak_throughput_rps']} req/s @ {summary['peak_throughput_at_users']} users")
            print(f"  Best Latency:        {summary['best_latency_s']}s @ {summary['best_latency_at_users']} users")
        else:
            print(f"  Total Tests:         {summary['total_tests']}")
            print(f"  Success Rate:        {summary['overall_success_rate']}%")
            print(f"  Avg Throughput:      {summary['avg_throughput']} req/s")
    
    tests = data.get('results', data.get('tests', []))
    
    print(f"\n{'-'*70}")
    print("THROUGHPUT vs CONCURRENT USERS")
    print(f"{'-'*70}")
    max_throughput = max(get_metric(t, 'throughput') for t in tests)
    for t in tests:
        throughput = get_metric(t, 'throughput')
        bar_length = int(throughput / max_throughput * 40) if max_throughput > 0 else 0
        bar = "#" * bar_length
        print(f"  {t['user_count']:>3} users | {bar:<40} {throughput:.2f} req/s")
    
    print(f"\n{'-'*70}")
    print("RESPONSE TIME vs CONCURRENT USERS")
    print(f"{'-'*70}")
    max_time = max(get_metric(t, 'avg_response_time') for t in tests)
    for t in tests:
        avg_time = get_metric(t, 'avg_response_time')
        bar_length = int(avg_time / max_time * 40) if max_time > 0 else 0
        bar = "=" * bar_length
        print(f"  {t['user_count']:>3} users | {bar:<40} {avg_time:.2f}s")
    
    print(f"\n{'-'*70}")
    print("LATENCY PERCENTILES")
    print(f"{'-'*70}")
    print(f"  {'Users':>6} | {'P50':>8} | {'P95':>8} | {'P99':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for t in tests:
        p50 = get_metric(t, 'p50')
        p95 = get_metric(t, 'p95')
        p99 = get_metric(t, 'p99')
        print(f"  {t['user_count']:>6} | {p50:>7.2f}s | {p95:>7.2f}s | {p99:>7.2f}s")
    
    print(f"\n{'='*70}\n")

def generate_matplotlib_charts(data, filename):
    """Generate charts using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[INFO] matplotlib not installed. Showing text report only.\n")
        return False
    
    is_benchmark = 'benchmark_info' in data
    tests = data.get('results', data.get('tests', []))
    users = [t['user_count'] for t in tests]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    title = f"Performance Test Results"
    if is_benchmark:
        title += f" - {data['benchmark_info']['gpu_name']}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Throughput
    throughput = [get_metric(t, 'throughput') for t in tests]
    axes[0,0].bar(users, throughput, color='steelblue', edgecolor='navy')
    axes[0,0].set_xlabel('Concurrent Users')
    axes[0,0].set_ylabel('Throughput (req/s)')
    axes[0,0].set_title('Throughput vs Concurrent Users')
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # 2. Response Time
    avg_times = [get_metric(t, 'avg_response_time') for t in tests]
    p50 = [get_metric(t, 'p50') for t in tests]
    p95 = [get_metric(t, 'p95') for t in tests]
    p99 = [get_metric(t, 'p99') for t in tests]
    
    axes[0,1].plot(users, avg_times, 'o-', label='Avg', linewidth=2, markersize=8)
    axes[0,1].plot(users, p50, 's--', label='P50', linewidth=1.5)
    axes[0,1].plot(users, p95, '^--', label='P95', linewidth=1.5)
    axes[0,1].plot(users, p99, 'd--', label='P99', linewidth=1.5)
    axes[0,1].set_xlabel('Concurrent Users')
    axes[0,1].set_ylabel('Response Time (s)')
    axes[0,1].set_title('Response Time vs Concurrent Users')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # 3. Success Rate
    success_rates = [get_metric(t, 'success_rate') or t.get('successful', 0)/t['user_count']*100 for t in tests]
    colors = ['green' if r == 100 else 'orange' if r >= 95 else 'red' for r in success_rates]
    axes[1,0].bar(users, success_rates, color=colors, edgecolor='darkgray')
    axes[1,0].set_xlabel('Concurrent Users')
    axes[1,0].set_ylabel('Success Rate (%)')
    axes[1,0].set_title('Success Rate vs Concurrent Users')
    axes[1,0].set_ylim(0, 105)
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # 4. Response Time Range
    min_times = [get_metric(t, 'min_time') for t in tests]
    max_times = [get_metric(t, 'max_time') for t in tests]
    
    for i, u in enumerate(users):
        axes[1,1].plot([u, u], [min_times[i], max_times[i]], 'k-', linewidth=2)
        axes[1,1].plot(u, min_times[i], 'g^', markersize=8)
        axes[1,1].plot(u, max_times[i], 'rv', markersize=8)
        axes[1,1].plot(u, avg_times[i], 'bo', markersize=10)
    
    axes[1,1].set_xlabel('Concurrent Users')
    axes[1,1].set_ylabel('Response Time (s)')
    axes[1,1].set_title('Response Time Range (Min/Avg/Max)')
    axes[1,1].legend(['Range', 'Min', 'Max', 'Avg'], loc='upper left')
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    base_name = Path(filename).stem
    output_file = REPORTS_DIR / f"{base_name}_report.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {output_file}")
    
    return True

def compare_two_gpus_visual(json_file1: str, json_file2: str):
    """Compare two GPU results side by side with charts"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("[ERROR] matplotlib required for GPU comparison charts")
        return False
    
    with open(json_file1) as f:
        data1 = json.load(f)
    with open(json_file2) as f:
        data2 = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    users1 = [r['user_count'] for r in data1['results']]
    latency1 = [r['metrics']['avg_latency_s'] for r in data1['results']]
    throughput1 = [r['metrics']['throughput_rps'] for r in data1['results']]
    p95_1 = [r['metrics']['p95_s'] for r in data1['results']]
    apdex1 = [r['metrics']['apdex_score'] for r in data1['results']]
    
    users2 = [r['user_count'] for r in data2['results']]
    latency2 = [r['metrics']['avg_latency_s'] for r in data2['results']]
    throughput2 = [r['metrics']['throughput_rps'] for r in data2['results']]
    p95_2 = [r['metrics']['p95_s'] for r in data2['results']]
    apdex2 = [r['metrics']['apdex_score'] for r in data2['results']]
    
    gpu1_name = data1['benchmark_info']['gpu_name'].split('NVIDIA GeForce ')[-1]
    gpu2_name = data2['benchmark_info']['gpu_name'].split('NVIDIA GeForce ')[-1]
    
    # Plot 1: Latency
    axes[0, 0].plot(users1, latency1, 'o-', label=gpu1_name, linewidth=2.5, markersize=8, color='#2E86AB')
    axes[0, 0].plot(users2, latency2, 's-', label=gpu2_name, linewidth=2.5, markersize=8, color='#A23B72')
    axes[0, 0].set_xlabel('Concurrent Users')
    axes[0, 0].set_ylabel('Avg Latency (s)')
    axes[0, 0].set_title('Average Latency Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Throughput
    x = range(len(users1))
    width = 0.35
    axes[0, 1].bar([i - width/2 for i in x], throughput1, width, label=gpu1_name, color='#2E86AB', edgecolor='navy')
    axes[0, 1].bar([i + width/2 for i in x], throughput2, width, label=gpu2_name, color='#A23B72', edgecolor='darkred')
    axes[0, 1].set_xlabel('Concurrent Users')
    axes[0, 1].set_ylabel('Throughput (req/s)')
    axes[0, 1].set_title('Throughput Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(users1)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: P95 Latency
    axes[1, 0].plot(users1, p95_1, '^-', label=f'{gpu1_name} (P95)', linewidth=2.5, markersize=8, color='#2E86AB')
    axes[1, 0].plot(users2, p95_2, 'v-', label=f'{gpu2_name} (P95)', linewidth=2.5, markersize=8, color='#A23B72')
    axes[1, 0].set_xlabel('Concurrent Users')
    axes[1, 0].set_ylabel('P95 Latency (s)')
    axes[1, 0].set_title('P95 Latency Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Apdex Score
    axes[1, 1].plot(users1, apdex1, 'D-', label=gpu1_name, linewidth=2.5, markersize=8, color='#2E86AB')
    axes[1, 1].plot(users2, apdex2, 'D-', label=gpu2_name, linewidth=2.5, markersize=8, color='#A23B72')
    axes[1, 1].axhline(y=0.94, color='green', linestyle='--', alpha=0.5, label='Excellent')
    axes[1, 1].axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Good')
    axes[1, 1].set_xlabel('Concurrent Users')
    axes[1, 1].set_ylabel('Apdex Score')
    axes[1, 1].set_title('User Satisfaction (Apdex Score)')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f'GPU Performance Comparison: {gpu1_name} vs {gpu2_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = REPORTS_DIR / f"gpu_comparison_{Path(json_file1).stem}_vs_{Path(json_file2).stem}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ GPU comparison chart saved: {output_file}\n")
    
    return True

def compare_all_gpus_visual():
    """Compare all available GPU benchmark results"""
    json_files = glob.glob(str(RESULTS_DIR / "benchmark_*.json"))
    
    if len(json_files) < 2:
        print("[INFO] Need at least 2 benchmark results for comparison")
        return False
    
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"\n{'='*70}")
    print("GPU VISUAL COMPARISON")
    print(f"{'='*70}")
    print(f"Found {len(json_files)} benchmark results")
    print("Comparing latest 2 results...\n")
    
    return compare_two_gpus_visual(json_files[0], json_files[1])

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if not os.path.isabs(filename):
            filename = SCRIPT_DIR / filename
    else:
        filename = find_latest_results()
        if not filename:
            print("[ERROR] No test results found in 'results/' directory.")
            print("        Run benchmark.py or concurrent_test.py first.")
            sys.exit(1)
        print(f"Using latest results: {filename}")
    
    if not os.path.exists(filename):
        print(f"[ERROR] File not found: {filename}")
        sys.exit(1)
    
    data = load_results(filename)
    generate_matplotlib_charts(data, str(filename))
    print_text_report(data, str(filename))

if __name__ == "__main__":
    main()
