import json
import csv
import glob
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def export_single_result_to_csv(json_file: str):
    """Export a single JSON result to CSV"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    base_name = Path(json_file).stem
    csv_file = REPORTS_DIR / f"{base_name}.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        info = data['benchmark_info']
        writer.writerow(['=== BENCHMARK INFO ===', ''])
        writer.writerow(['Test Date', info['date']])
        writer.writerow(['GPU', info['gpu_name']])
        writer.writerow(['Test Type', info['test_type']])
        writer.writerow([''])
        
        gpu_info = data['gpu_info']
        writer.writerow(['=== GPU INFO ===', ''])
        writer.writerow(['Memory Total (GB)', f"{gpu_info['memory_total_mb']/1024:.1f}"])
        writer.writerow(['Temperature (°C)', gpu_info['temperature_c']])
        writer.writerow([''])
        
        writer.writerow(['=== PERFORMANCE RESULTS ===', ''])
        writer.writerow([
            'Concurrent Users', 'Total Requests', 'Successful', 'Throughput (req/s)',
            'Avg Latency (s)', 'Min Latency (s)', 'Max Latency (s)', 'P50 (s)',
            'P95 (s)', 'P99 (s)', 'Success Rate (%)', 'Apdex Score'
        ])
        
        for result in data['results']:
            m = result['metrics']
            writer.writerow([
                result['user_count'], result['total_requests'], result['total_successful'],
                f"{m['throughput_rps']:.2f}", f"{m['avg_latency_s']:.3f}",
                f"{m['min_latency_s']:.3f}", f"{m['max_latency_s']:.3f}",
                f"{m['p50_s']:.3f}", f"{m['p95_s']:.3f}", f"{m['p99_s']:.3f}",
                f"{m['success_rate_pct']:.1f}", f"{m['apdex_score']:.3f}"
            ])
        
        writer.writerow([''])
        summary = data['summary']
        writer.writerow(['=== SUMMARY ===', ''])
        writer.writerow(['Peak Throughput (req/s)', summary['peak_throughput_rps']])
        writer.writerow(['Best Latency (s)', summary['best_latency_s']])
        writer.writerow(['Max Tested Users', summary['max_tested_users']])
    
    print(f"✓ Exported: {csv_file}")
    return csv_file

def export_all_to_single_csv():
    """Export all results to a single comparison CSV"""
    json_files = glob.glob(str(RESULTS_DIR / "benchmark_*.json"))
    
    if not json_files:
        print("[ERROR] No benchmark results found")
        return
    
    csv_file = REPORTS_DIR / "all_benchmarks_comparison.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            'GPU', 'Date', 'Test Type', 'Users', 'Throughput (req/s)',
            'Avg Latency (s)', 'P50 (s)', 'P95 (s)', 'P99 (s)',
            'Success Rate (%)', 'Apdex Score', 'Memory (GB)', 'Temp (°C)'
        ])
        
        for json_file in sorted(json_files, key=os.path.getmtime, reverse=True):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            info = data['benchmark_info']
            gpu_info = data['gpu_info']
            gpu_name = info['gpu_name'].replace('NVIDIA GeForce ', '')
            
            for result in data['results']:
                m = result['metrics']
                writer.writerow([
                    gpu_name, info['date'], info['test_type'], result['user_count'],
                    f"{m['throughput_rps']:.2f}", f"{m['avg_latency_s']:.2f}",
                    f"{m['p50_s']:.2f}", f"{m['p95_s']:.2f}", f"{m['p99_s']:.2f}",
                    f"{m['success_rate_pct']:.1f}", f"{m['apdex_score']:.3f}",
                    f"{gpu_info['memory_total_mb']/1024:.1f}", gpu_info['temperature_c']
                ])
    
    print(f"\n✓ All results exported to: {csv_file}")
    return csv_file

def main():
    print("\n" + "="*70)
    print("BENCHMARK RESULTS → CSV EXPORT")
    print("="*70)
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        
        if not os.path.isabs(json_file):
            if not os.path.exists(json_file):
                alt_path = RESULTS_DIR / json_file
                if alt_path.exists():
                    json_file = str(alt_path)
        
        if not os.path.exists(json_file):
            print(f"[ERROR] File not found: {json_file}")
            return
        
        print(f"\nExporting: {json_file}")
        export_single_result_to_csv(json_file)
        print("\n✓ Done!\n")
    
    else:
        json_files = glob.glob(str(RESULTS_DIR / "benchmark_*.json"))
        
        if not json_files:
            print("\n[ERROR] No benchmark results found")
            print("        Run 'make benchmark' first\n")
            return
        
        print(f"\nFound {len(json_files)} benchmark result(s)")
        print("\nExporting individual CSVs...")
        
        for json_file in json_files:
            export_single_result_to_csv(json_file)
        
        print("\nCreating combined comparison CSV...")
        export_all_to_single_csv()
        
        print(f"\n{'='*70}")
        print("✓ All exports complete!")
        print(f"{'='*70}")
        print(f"\nCSV files saved to: {REPORTS_DIR}\n")

if __name__ == "__main__":
    main()
