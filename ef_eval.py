import json
import os
from collections import defaultdict
import argparse

input_dir="./output/llama2-7b-chat-hf/nq/"
output_path="./output/llama2-7b-chat-hf/nq/ef_eval.json"
custom_files=[
    "nq_lfd_ef.json",
]

import json
import os
import argparse

def calculate_metrics(data):
    if not data:
        return {
            'avg_input_len': 0,
            'avg_output_len': 0,
            'avg_total_time': 0,
            'avg_latency_ms_per_token': 0,
            'avg_memory_before': 0,
            'avg_peak_memory': 0,
            'avg_throughput': 0,
            'avg_memory_diff': 0,
            'avg_memory_ratio': 0,
            'sample_count': 0
        }
    
    metrics = {
        'input_lens': [],
        'output_lens': [],
        'total_times': [],
        'latencies': [],
        'memory_befores': [],
        'peak_memories': [],
        'throughputs': [],
        'memory_diffs': [],
        'memory_ratios': []
    }
    
    for sample in data:
        m = sample['metrics']
        
        metrics['input_lens'].append(m['input_token_len'])
        metrics['output_lens'].append(m['output_token_len'])
        metrics['total_times'].append(m['total_time'])
        metrics['memory_befores'].append(m['memory_before_mb'])
        metrics['peak_memories'].append(m['peak_memory_mb'])
        
        throughput = m['output_token_len'] / m['total_time'] if m['total_time'] > 0 else 0
        memory_diff = m['peak_memory_mb'] - m['memory_before_mb']
        memory_ratio = memory_diff / m['memory_before_mb'] if m['memory_before_mb'] > 0 else 0
        latency = (m['total_time'] / m['output_token_len'] * 1000) if m['output_token_len'] > 0 else 0
        
        metrics['throughputs'].append(throughput)
        metrics['memory_diffs'].append(memory_diff)
        metrics['memory_ratios'].append(memory_ratio)
        metrics['latencies'].append(latency)
    
    def safe_avg(values):
        return sum(values) / len(values) if values else 0
    
    return {
        'avg_input_len': safe_avg(metrics['input_lens']),
        'avg_output_len': safe_avg(metrics['output_lens']),
        'avg_total_time': safe_avg(metrics['total_times']),
        'avg_latency_ms_per_token': safe_avg(metrics['latencies']),
        'avg_memory_before': safe_avg(metrics['memory_befores']),
        'avg_peak_memory': safe_avg(metrics['peak_memories']),
        'avg_throughput': safe_avg(metrics['throughputs']), 
        'avg_memory_diff': safe_avg(metrics['memory_diffs']),
        'avg_memory_ratio': safe_avg(metrics['memory_ratios']),
        'sample_count': len(data)
    }

def process_files(input_dir):
    results = {}
    
    # noise_result
    noise_files = [f for f in os.listdir(input_dir) if f.startswith('nq_noise_') and f.endswith('.json')]
    
    # custom_result
    for custom_file in custom_files:
        if custom_file in os.listdir(input_dir):
            noise_files.append(custom_file)
    
    for filename in noise_files:
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results[filename] = calculate_metrics(data)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return results

def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")

    
results = process_files(input_dir)
save_results(results, output_path)