#!/usr/bin/env python3
"""
LangSmith Trace Analysis Script for EHS Ingestion Process

This script analyzes LangSmith traces to provide comprehensive insights into
the EHS data ingestion workflow, including LLM operations, token usage,
performance metrics, and cost analysis.

Usage:
    python3 analyze_traces.py <trace_file.json> [--output report.txt]
"""

import json
import argparse
import sys
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import statistics
from pathlib import Path


class TraceAnalyzer:
    """Analyzes LangSmith traces for EHS ingestion processes."""
    
    # GPT-4 pricing per 1K tokens (approximate)
    PRICING = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'claude': {'input': 0.008, 'output': 0.024},
        'default': {'input': 0.01, 'output': 0.03}
    }
    
    def __init__(self, trace_file: str):
        """Initialize the analyzer with a trace file."""
        self.trace_file = Path(trace_file)
        self.traces = []
        self.analysis_results = {}
        
    def load_traces(self) -> None:
        """Load traces from JSON file."""
        try:
            with open(self.trace_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                self.traces = data
            elif isinstance(data, dict):
                if 'traces' in data:
                    self.traces = data['traces']
                elif 'runs' in data:
                    self.traces = data['runs']
                else:
                    # Assume single trace
                    self.traces = [data]
            else:
                raise ValueError(f"Unexpected JSON structure in {self.trace_file}")
                
            print(f"Loaded {len(self.traces)} traces from {self.trace_file}")
            
        except FileNotFoundError:
            print(f"Error: Trace file {self.trace_file} not found")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {self.trace_file}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading traces: {e}")
            sys.exit(1)
    
    def extract_llm_calls(self) -> List[Dict[str, Any]]:
        """Extract all LLM calls from traces."""
        llm_calls = []
        
        def extract_from_run(run: Dict[str, Any]) -> None:
            """Recursively extract LLM calls from a run and its children."""
            # Check if this is an LLM call
            if self._is_llm_call(run):
                llm_calls.append(run)
            
            # Process child runs
            if 'child_runs' in run:
                for child in run['child_runs']:
                    extract_from_run(child)
        
        for trace in self.traces:
            extract_from_run(trace)
        
        return llm_calls
    
    def _is_llm_call(self, run: Dict[str, Any]) -> bool:
        """Determine if a run represents an LLM call."""
        run_type = run.get('run_type', '').lower()
        name = run.get('name', '').lower()
        
        # Common indicators of LLM calls
        llm_indicators = [
            'llm', 'chat', 'openai', 'anthropic', 'claude', 'gpt',
            'completion', 'generate', 'prompt'
        ]
        
        return (
            run_type in ['llm', 'chat_model'] or
            any(indicator in name for indicator in llm_indicators) or
            'model' in run.get('extra', {}).get('metadata', {})
        )
    
    def analyze_operations(self, llm_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze types of operations performed."""
        operations = defaultdict(list)
        
        for call in llm_calls:
            name = call.get('name', 'unknown')
            tags = call.get('tags', [])
            
            # Categorize based on name and tags
            operation_type = self._categorize_operation(name, tags)
            operations[operation_type].append(call)
        
        return dict(operations)
    
    def _categorize_operation(self, name: str, tags: List[str]) -> str:
        """Categorize operation based on name and tags."""
        name_lower = name.lower()
        tags_lower = [tag.lower() for tag in tags]
        
        # EHS-specific categorization
        if any(keyword in name_lower for keyword in ['schema', 'structure']):
            return 'Schema Extraction'
        elif any(keyword in name_lower for keyword in ['entity', 'ner', 'extraction']):
            return 'Entity Extraction'
        elif any(keyword in name_lower for keyword in ['classify', 'classification']):
            return 'Classification'
        elif any(keyword in name_lower for keyword in ['transform', 'convert']):
            return 'Data Transformation'
        elif any(keyword in name_lower for keyword in ['validate', 'validation']):
            return 'Data Validation'
        elif any(keyword in name_lower for keyword in ['summary', 'summarize']):
            return 'Summarization'
        elif any(keyword in name_lower for keyword in ['parse', 'parsing']):
            return 'Document Parsing'
        elif any(tag in tags_lower for tag in ['ehs', 'safety', 'environment', 'health']):
            return 'EHS Processing'
        else:
            return 'General LLM Call'
    
    def analyze_models(self, llm_calls: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze models used and their frequency."""
        models = Counter()
        
        for call in llm_calls:
            # Try different ways to extract model info
            model = None
            
            # Check in extra metadata
            extra = call.get('extra', {})
            metadata = extra.get('metadata', {})
            if 'model' in metadata:
                model = metadata['model']
            elif 'model_name' in metadata:
                model = metadata['model_name']
            
            # Check in serialized fields
            serialized = call.get('serialized', {})
            if not model and 'model' in serialized:
                model = serialized['model']
            
            # Check inputs
            inputs = call.get('inputs', {})
            if not model and 'model' in inputs:
                model = inputs['model']
            
            # Fallback to name parsing
            if not model:
                name = call.get('name', '')
                if 'gpt-4' in name.lower():
                    model = 'gpt-4'
                elif 'gpt-3.5' in name.lower():
                    model = 'gpt-3.5-turbo'
                elif 'claude' in name.lower():
                    model = 'claude'
                else:
                    model = 'unknown'
            
            models[model] += 1
        
        return dict(models)
    
    def analyze_tokens(self, llm_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze token usage statistics."""
        token_stats = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'calls_with_token_info': 0,
            'by_operation': defaultdict(lambda: {'input': 0, 'output': 0, 'total': 0})
        }
        
        for call in llm_calls:
            input_tokens = self._extract_token_count(call, 'input')
            output_tokens = self._extract_token_count(call, 'output')
            
            if input_tokens is not None or output_tokens is not None:
                token_stats['calls_with_token_info'] += 1
                
                if input_tokens:
                    token_stats['total_input_tokens'] += input_tokens
                if output_tokens:
                    token_stats['total_output_tokens'] += output_tokens
                
                # Categorize by operation type
                operation_type = self._categorize_operation(
                    call.get('name', ''), call.get('tags', [])
                )
                if input_tokens:
                    token_stats['by_operation'][operation_type]['input'] += input_tokens
                if output_tokens:
                    token_stats['by_operation'][operation_type]['output'] += output_tokens
                token_stats['by_operation'][operation_type]['total'] += (input_tokens or 0) + (output_tokens or 0)
        
        token_stats['total_tokens'] = token_stats['total_input_tokens'] + token_stats['total_output_tokens']
        return token_stats
    
    def _extract_token_count(self, call: Dict[str, Any], token_type: str) -> Optional[int]:
        """Extract token count from various possible locations in the call data."""
        # Common locations for token information
        locations = [
            call.get('extra', {}).get('metadata', {}),
            call.get('outputs', {}),
            call.get('usage', {}),
            call.get('token_usage', {})
        ]
        
        # Common field names for tokens
        field_names = {
            'input': ['input_tokens', 'prompt_tokens', 'input_token_count'],
            'output': ['output_tokens', 'completion_tokens', 'output_token_count']
        }
        
        for location in locations:
            if not isinstance(location, dict):
                continue
                
            for field_name in field_names.get(token_type, []):
                if field_name in location and isinstance(location[field_name], (int, float)):
                    return int(location[field_name])
        
        return None
    
    def analyze_latency(self, llm_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze latency statistics."""
        latencies = []
        latency_by_operation = defaultdict(list)
        
        for call in llm_calls:
            latency = self._extract_latency(call)
            if latency is not None:
                latencies.append(latency)
                operation_type = self._categorize_operation(
                    call.get('name', ''), call.get('tags', [])
                )
                latency_by_operation[operation_type].append(latency)
        
        stats = {}
        if latencies:
            stats['overall'] = {
                'count': len(latencies),
                'total_seconds': sum(latencies),
                'average_seconds': statistics.mean(latencies),
                'median_seconds': statistics.median(latencies),
                'min_seconds': min(latencies),
                'max_seconds': max(latencies)
            }
            
            # Per-operation statistics
            stats['by_operation'] = {}
            for operation, op_latencies in latency_by_operation.items():
                if op_latencies:
                    stats['by_operation'][operation] = {
                        'count': len(op_latencies),
                        'average_seconds': statistics.mean(op_latencies),
                        'total_seconds': sum(op_latencies)
                    }
        
        return stats
    
    def _extract_latency(self, call: Dict[str, Any]) -> Optional[float]:
        """Extract latency from call data."""
        start_time = call.get('start_time')
        end_time = call.get('end_time')
        
        if start_time and end_time:
            try:
                # Handle different timestamp formats
                if isinstance(start_time, str):
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                else:
                    start_dt = datetime.fromtimestamp(start_time / 1000)
                
                if isinstance(end_time, str):
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                else:
                    end_dt = datetime.fromtimestamp(end_time / 1000)
                
                return (end_dt - start_dt).total_seconds()
            except Exception:
                pass
        
        # Check for explicit duration field
        duration = call.get('duration')
        if duration:
            return duration / 1000 if duration > 1000 else duration
        
        return None
    
    def analyze_errors(self, llm_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze errors and failed operations."""
        errors = []
        
        for call in llm_calls:
            if call.get('status') == 'error' or call.get('error'):
                errors.append({
                    'name': call.get('name'),
                    'error': call.get('error'),
                    'status': call.get('status'),
                    'operation_type': self._categorize_operation(
                        call.get('name', ''), call.get('tags', [])
                    )
                })
        
        error_summary = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(llm_calls) if llm_calls else 0,
            'errors_by_operation': Counter(error['operation_type'] for error in errors),
            'error_details': errors[:5]  # First 5 errors for details
        }
        
        return error_summary
    
    def calculate_costs(self, token_stats: Dict[str, Any], models: Dict[str, int]) -> Dict[str, Any]:
        """Calculate estimated costs based on token usage."""
        costs = {'total_cost': 0, 'by_model': {}, 'by_operation': {}}
        
        # Estimate cost per model (simplified - assumes even distribution)
        total_calls = sum(models.values())
        
        for model, count in models.items():
            model_key = model.lower()
            pricing = self.PRICING.get(model_key, self.PRICING['default'])
            
            # Estimate token distribution for this model
            model_ratio = count / total_calls if total_calls > 0 else 0
            model_input_tokens = token_stats['total_input_tokens'] * model_ratio
            model_output_tokens = token_stats['total_output_tokens'] * model_ratio
            
            input_cost = (model_input_tokens / 1000) * pricing['input']
            output_cost = (model_output_tokens / 1000) * pricing['output']
            model_cost = input_cost + output_cost
            
            costs['by_model'][model] = {
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': model_cost,
                'input_tokens': int(model_input_tokens),
                'output_tokens': int(model_output_tokens)
            }
            costs['total_cost'] += model_cost
        
        # Cost by operation type
        for operation, tokens in token_stats['by_operation'].items():
            # Use average pricing for operation-level estimates
            avg_pricing = self.PRICING['default']
            input_cost = (tokens['input'] / 1000) * avg_pricing['input']
            output_cost = (tokens['output'] / 1000) * avg_pricing['output']
            
            costs['by_operation'][operation] = {
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': input_cost + output_cost,
                'input_tokens': tokens['input'],
                'output_tokens': tokens['output']
            }
        
        return costs
    
    def get_sample_operations(self, operations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Get sample inputs/outputs for each operation type."""
        samples = {}
        
        for op_type, calls in operations.items():
            if calls:
                # Take the first call as a sample
                sample_call = calls[0]
                samples[op_type] = {
                    'count': len(calls),
                    'sample_name': sample_call.get('name'),
                    'sample_input': self._truncate_text(str(sample_call.get('inputs', {})), 200),
                    'sample_output': self._truncate_text(str(sample_call.get('outputs', {})), 200),
                    'tags': sample_call.get('tags', [])
                }
        
        return samples
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max_length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    def find_most_expensive_operations(self, operations: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, float]]:
        """Find operations that took the longest time."""
        operation_times = []
        
        for op_type, calls in operations.items():
            total_time = 0
            call_count = 0
            
            for call in calls:
                latency = self._extract_latency(call)
                if latency is not None:
                    total_time += latency
                    call_count += 1
            
            if call_count > 0:
                avg_time = total_time / call_count
                operation_times.append((op_type, total_time, avg_time, call_count))
        
        # Sort by total time descending
        return sorted(operation_times, key=lambda x: x[1], reverse=True)
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete analysis and return results."""
        print("Starting trace analysis...")
        
        # Load traces
        self.load_traces()
        
        # Extract LLM calls
        print("Extracting LLM calls...")
        llm_calls = self.extract_llm_calls()
        
        # Run analyses
        print("Analyzing operations...")
        operations = self.analyze_operations(llm_calls)
        
        print("Analyzing models...")
        models = self.analyze_models(llm_calls)
        
        print("Analyzing tokens...")
        token_stats = self.analyze_tokens(llm_calls)
        
        print("Analyzing latency...")
        latency_stats = self.analyze_latency(llm_calls)
        
        print("Analyzing errors...")
        error_stats = self.analyze_errors(llm_calls)
        
        print("Calculating costs...")
        costs = self.calculate_costs(token_stats, models)
        
        print("Getting samples...")
        samples = self.get_sample_operations(operations)
        
        print("Finding time-consuming operations...")
        expensive_ops = self.find_most_expensive_operations(operations)
        
        # Compile results
        results = {
            'summary': {
                'total_traces': len(self.traces),
                'total_llm_calls': len(llm_calls),
                'unique_operations': len(operations),
                'models_used': len(models),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'operations': operations,
            'models': models,
            'tokens': token_stats,
            'latency': latency_stats,
            'errors': error_stats,
            'costs': costs,
            'samples': samples,
            'expensive_operations': expensive_ops
        }
        
        self.analysis_results = results
        return results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive human-readable report."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run run_analysis() first.")
        
        results = self.analysis_results
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 80,
            "LANGSMITH TRACE ANALYSIS REPORT - EHS INGESTION PROCESS",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Trace file: {self.trace_file}",
            ""
        ])
        
        # Summary
        summary = results['summary']
        report_lines.extend([
            "EXECUTIVE SUMMARY",
            "-" * 40,
            f"Total Traces Analyzed: {summary['total_traces']:,}",
            f"Total LLM Calls: {summary['total_llm_calls']:,}",
            f"Unique Operation Types: {summary['unique_operations']}",
            f"Models Used: {summary['models_used']}",
            ""
        ])
        
        # Models Analysis
        report_lines.extend([
            "MODELS USED",
            "-" * 40
        ])
        for model, count in sorted(results['models'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / summary['total_llm_calls']) * 100
            report_lines.append(f"{model}: {count:,} calls ({percentage:.1f}%)")
        report_lines.append("")
        
        # Operations Analysis
        report_lines.extend([
            "OPERATION TYPES",
            "-" * 40
        ])
        for op_type, calls in sorted(results['operations'].items(), key=lambda x: len(x[1]), reverse=True):
            count = len(calls)
            percentage = (count / summary['total_llm_calls']) * 100
            report_lines.append(f"{op_type}: {count:,} calls ({percentage:.1f}%)")
        report_lines.append("")
        
        # Token Usage
        tokens = results['tokens']
        report_lines.extend([
            "TOKEN USAGE STATISTICS",
            "-" * 40,
            f"Total Input Tokens: {tokens['total_input_tokens']:,}",
            f"Total Output Tokens: {tokens['total_output_tokens']:,}",
            f"Total Tokens: {tokens['total_tokens']:,}",
            f"Calls with Token Info: {tokens['calls_with_token_info']:,}",
            ""
        ])
        
        # Token usage by operation
        if tokens['by_operation']:
            report_lines.extend([
                "TOKEN USAGE BY OPERATION:",
                ""
            ])
            for op_type, op_tokens in sorted(tokens['by_operation'].items(), key=lambda x: x[1]['total'], reverse=True):
                report_lines.append(f"  {op_type}:")
                report_lines.append(f"    Input: {op_tokens['input']:,} tokens")
                report_lines.append(f"    Output: {op_tokens['output']:,} tokens")
                report_lines.append(f"    Total: {op_tokens['total']:,} tokens")
                report_lines.append("")
        
        # Latency Analysis
        latency = results['latency']
        if 'overall' in latency:
            overall = latency['overall']
            report_lines.extend([
                "LATENCY ANALYSIS",
                "-" * 40,
                f"Total Calls with Timing: {overall['count']:,}",
                f"Total Processing Time: {overall['total_seconds']:.2f} seconds",
                f"Average Latency: {overall['average_seconds']:.3f} seconds",
                f"Median Latency: {overall['median_seconds']:.3f} seconds",
                f"Min Latency: {overall['min_seconds']:.3f} seconds",
                f"Max Latency: {overall['max_seconds']:.3f} seconds",
                ""
            ])
            
            if 'by_operation' in latency:
                report_lines.extend([
                    "AVERAGE LATENCY BY OPERATION:",
                    ""
                ])
                for op_type, op_latency in sorted(latency['by_operation'].items(), 
                                                key=lambda x: x[1]['average_seconds'], reverse=True):
                    report_lines.append(f"  {op_type}: {op_latency['average_seconds']:.3f}s "
                                      f"({op_latency['count']} calls)")
                report_lines.append("")
        
        # Error Analysis
        errors = results['errors']
        report_lines.extend([
            "ERROR ANALYSIS",
            "-" * 40,
            f"Total Errors: {errors['total_errors']:,}",
            f"Error Rate: {errors['error_rate']:.2%}",
            ""
        ])
        
        if errors['errors_by_operation']:
            report_lines.extend([
                "ERRORS BY OPERATION:",
                ""
            ])
            for op_type, error_count in errors['errors_by_operation'].items():
                report_lines.append(f"  {op_type}: {error_count} errors")
            report_lines.append("")
        
        # Cost Analysis
        costs = results['costs']
        report_lines.extend([
            "COST ANALYSIS (ESTIMATED)",
            "-" * 40,
            f"Total Estimated Cost: ${costs['total_cost']:.4f}",
            ""
        ])
        
        if costs['by_model']:
            report_lines.extend([
                "COSTS BY MODEL:",
                ""
            ])
            for model, model_costs in sorted(costs['by_model'].items(), 
                                           key=lambda x: x[1]['total_cost'], reverse=True):
                report_lines.append(f"  {model}:")
                report_lines.append(f"    Input Cost: ${model_costs['input_cost']:.4f}")
                report_lines.append(f"    Output Cost: ${model_costs['output_cost']:.4f}")
                report_lines.append(f"    Total Cost: ${model_costs['total_cost']:.4f}")
                report_lines.append("")
        
        # Most Time-Consuming Operations
        expensive_ops = results['expensive_operations']
        if expensive_ops:
            report_lines.extend([
                "MOST TIME-CONSUMING OPERATIONS",
                "-" * 40
            ])
            for i, (op_type, total_time, avg_time, count) in enumerate(expensive_ops[:5]):
                report_lines.append(f"{i+1}. {op_type}:")
                report_lines.append(f"   Total Time: {total_time:.2f}s")
                report_lines.append(f"   Average Time: {avg_time:.3f}s")
                report_lines.append(f"   Call Count: {count}")
                report_lines.append("")
        
        # Sample Operations
        samples = results['samples']
        if samples:
            report_lines.extend([
                "SAMPLE OPERATIONS",
                "-" * 40
            ])
            for op_type, sample in samples.items():
                report_lines.extend([
                    f"{op_type} (Sample of {sample['count']} calls):",
                    f"  Name: {sample['sample_name']}",
                    f"  Tags: {', '.join(sample['tags']) if sample['tags'] else 'None'}",
                    f"  Sample Input: {sample['sample_input']}",
                    f"  Sample Output: {sample['sample_output']}",
                    ""
                ])
        
        report_lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report


def main():
    """Main function to run the trace analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze LangSmith traces for EHS ingestion process",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 analyze_traces.py traces.json
    python3 analyze_traces.py traces.json --output analysis_report.txt
    python3 analyze_traces.py data/langsmith_traces.json --output reports/trace_analysis.txt
        """
    )
    
    parser.add_argument(
        'trace_file',
        help='Path to the JSON file containing LangSmith traces'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for the analysis report (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.trace_file).exists():
        print(f"Error: Trace file '{args.trace_file}' does not exist")
        sys.exit(1)
    
    try:
        # Initialize analyzer
        analyzer = TraceAnalyzer(args.trace_file)
        
        # Run analysis
        results = analyzer.run_analysis()
        
        # Generate and display report
        report = analyzer.generate_report(args.output)
        
        if not args.output:
            print(report)
        
        print(f"\nAnalysis complete! Processed {results['summary']['total_llm_calls']} LLM calls.")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()