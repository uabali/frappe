"""GPU Benchmark Comparison Tool"""

import json
import glob
import sys
import csv
from pathlib import Path
from typing import List, Dict

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def load_benchmark_results() -> List[Dict]:
    """Load all benchmark JSON files"""
    json_files = glob.glob(str(RESULTS_DIR / "benchmark_*.json"))
    results = []
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.append({
                'file': Path(file).name,
                'path': file,
                'data': data
            })
    
    results.sort(key=lambda x: x['data']['benchmark_info']['date'], reverse=True)
    return results

def load_specific_files(file_paths: List[str]) -> List[Dict]:
    """Load specific benchmark files"""
    results = []
    
    for file_path in file_paths:
        if not Path(file_path).exists():
            alt_path = RESULTS_DIR / file_path
            if alt_path.exists():
                file_path = str(alt_path)
            else:
                print(f"[WARN] File not found: {file_path}")
                continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            results.append({
                'file': Path(file_path).name,
                'path': file_path,
                'data': data
            })
    
    return results

def compare_two_gpus(result1: Dict, result2: Dict):
    """Compare two GPU benchmark results"""
    data1 = result1['data']
    data2 = result2['data']
    
    gpu1_name = data1['benchmark_info']['gpu_name'].replace('NVIDIA GeForce ', '')
    gpu2_name = data2['benchmark_info']['gpu_name'].replace('NVIDIA GeForce ', '')
    
    print("\n" + "="*80)
    print(f"GPU COMPARISON: {gpu1_name} vs {gpu2_name}")
    print("="*80)
    
    gpu1_info = data1['gpu_info']
    gpu2_info = data2['gpu_info']
    
    print(f"\n{'─'*80}")
    print("HARDWARE SPECIFICATIONS")
    print(f"{'─'*80}")
    print(f"{'Specification':<30} | {gpu1_name:>20} | {gpu2_name:>20}")
    print(f"{'-'*30}-+-{'-'*20}-+-{'-'*20}")
    print(f"{'Memory (GB)':<30} | {gpu1_info['memory_total_mb']/1024:>19.1f} | {gpu2_info['memory_total_mb']/1024:>19.1f}")
    print(f"{'Temperature (°C)':<30} | {gpu1_info['temperature_c']:>19d} | {gpu2_info['temperature_c']:>19d}")
    
    users1 = {r['user_count']: r for r in data1['results']}
    users2 = {r['user_count']: r for r in data2['results']}
    common_users = sorted(set(users1.keys()) & set(users2.keys()))
    
    print(f"\n{'─'*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'─'*80}")
    
    for user_count in common_users:
        r1 = users1[user_count]['metrics']
        r2 = users2[user_count]['metrics']
        
        print(f"\n📊 {user_count} CONCURRENT USERS")
        print(f"{'─'*80}")
        
        tp1 = r1['throughput_rps']
        tp2 = r2['throughput_rps']
        winner = "🥇" if tp1 > tp2 else "🥈"
        diff = abs(tp1 - tp2) / max(tp1, tp2) * 100
        print(f"{'Throughput (req/s)':<30} | {tp1:>14.2f} {winner if tp1 > tp2 else '🥈':>4} | {tp2:>14.2f} {'🥇' if tp2 > tp1 else '🥈':>4} | Diff: {diff:>5.1f}%")
        
        lat1 = r1['avg_latency_s']
        lat2 = r2['avg_latency_s']
        winner = "🥇" if lat1 < lat2 else "🥈"
        diff = abs(lat1 - lat2) / max(lat1, lat2) * 100
        print(f"{'Avg Latency (s)':<30} | {lat1:>14.2f} {winner if lat1 < lat2 else '🥈':>4} | {lat2:>14.2f} {'🥇' if lat2 < lat1 else '🥈':>4} | Diff: {diff:>5.1f}%")
        
        print(f"{'P95 Latency (s)':<30} | {r1['p95_s']:>18.2f} | {r2['p95_s']:>18.2f}")
        print(f"{'Apdex Score':<30} | {r1['apdex_score']:>18.3f} | {r2['apdex_score']:>18.3f}")
    
    sum1 = data1['summary']
    sum2 = data2['summary']
    
    print(f"\n{'─'*80}")
    print("OVERALL SUMMARY")
    print(f"{'─'*80}")
    print(f"{'Metric':<30} | {gpu1_name:>20} | {gpu2_name:>20}")
    print(f"{'-'*30}-+-{'-'*20}-+-{'-'*20}")
    print(f"{'Peak Throughput (req/s)':<30} | {sum1['peak_throughput_rps']:>19.2f} | {sum2['peak_throughput_rps']:>19.2f}")
    print(f"{'Best Latency (s)':<30} | {sum1['best_latency_s']:>19.2f} | {sum2['best_latency_s']:>19.2f}")
    print(f"{'Latency Degradation (%)':<30} | {sum1['latency_degradation_pct']:>19.1f} | {sum2['latency_degradation_pct']:>19.1f}")
    
    scores = {'gpu1': 0, 'gpu2': 0}
    
    print(f"\n{'─'*80}")
    print("🏆 WINNER ANALYSIS")
    print(f"{'─'*80}")
    
    if sum1['peak_throughput_rps'] > sum2['peak_throughput_rps']:
        print(f"✓ Throughput:      {gpu1_name}")
        scores['gpu1'] += 1
    else:
        print(f"✓ Throughput:      {gpu2_name}")
        scores['gpu2'] += 1
    
    if sum1['best_latency_s'] < sum2['best_latency_s']:
        print(f"✓ Best Latency:    {gpu1_name}")
        scores['gpu1'] += 1
    else:
        print(f"✓ Best Latency:    {gpu2_name}")
        scores['gpu2'] += 1
    
    if sum1['latency_degradation_pct'] < sum2['latency_degradation_pct']:
        print(f"✓ Stability:       {gpu1_name}")
        scores['gpu1'] += 1
    else:
        print(f"✓ Stability:       {gpu2_name}")
        scores['gpu2'] += 1
    
    print(f"\n{'─'*80}")
    if scores['gpu1'] > scores['gpu2']:
        print(f"🏆 OVERALL WINNER: {gpu1_name}")
    elif scores['gpu2'] > scores['gpu1']:
        print(f"🏆 OVERALL WINNER: {gpu2_name}")
    else:
        print("🤝 TIE: Both GPUs performed equally well")
    print(f"{'─'*80}\n")

def compare_all_results(results: List[Dict]):
    """Compare all available results"""
    if len(results) < 2:
        print("[ERROR] Need at least 2 benchmark results to compare")
        return
    
    print(f"\nFound {len(results)} benchmark results:")
    for i, r in enumerate(results, 1):
        info = r['data']['benchmark_info']
        gpu_name = info['gpu_name'].replace('NVIDIA GeForce ', '')
        print(f"  {i}. {gpu_name} - {info['date']}")
    
    for i in range(len(results) - 1):
        compare_two_gpus(results[i], results[i + 1])

def export_to_csv(results: List[Dict]):
    """Export comparison to CSV file"""
    csv_file = REPORTS_DIR / "gpu_comparison.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            'GPU', 'Date', 'Users', 'Throughput (req/s)', 'Avg Latency (s)',
            'P50 (s)', 'P95 (s)', 'P99 (s)', 'Success Rate (%)', 'Apdex Score',
            'Memory (GB)', 'Temp (°C)'
        ])
        
        for result in results:
            data = result['data']
            info = data['benchmark_info']
            gpu_info = data['gpu_info']
            gpu_name = info['gpu_name'].replace('NVIDIA GeForce ', '')
            
            for test_result in data['results']:
                m = test_result['metrics']
                writer.writerow([
                    gpu_name, info['date'], test_result['user_count'],
                    f"{m['throughput_rps']:.2f}", f"{m['avg_latency_s']:.2f}",
                    f"{m['p50_s']:.2f}", f"{m['p95_s']:.2f}", f"{m['p99_s']:.2f}",
                    f"{m['success_rate_pct']:.1f}", f"{m['apdex_score']:.3f}",
                    f"{gpu_info['memory_total_mb']/1024:.1f}", gpu_info['temperature_c']
                ])
    
    print(f"\n✓ CSV exported to: {csv_file}")

def main():
    export_csv = '--csv' in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != '--csv']
    
    if len(args) >= 2:
        results = load_specific_files(args)
    else:
        results = load_benchmark_results()
    
    if not results:
        print("[ERROR] No benchmark results found")
        print("        Run 'make benchmark' first")
        return
    
    if export_csv:
        export_to_csv(results)
    
    if len(args) == 2:
        compare_two_gpus(results[0], results[1])
    else:
        compare_all_results(results)

if __name__ == "__main__":
    main()
