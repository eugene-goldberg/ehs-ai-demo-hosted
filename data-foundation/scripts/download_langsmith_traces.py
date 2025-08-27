#!/usr/bin/env python3
"""
LangSmith Traces Download Script

This script downloads traces from LangSmith for analysis and debugging purposes.
It supports various filtering options and saves traces to JSON files with timestamps.

Usage:
    python3 download_langsmith_traces.py [options]

Examples:
    # Download last 24 hours of traces
    python3 download_langsmith_traces.py --hours 24

    # Download last 100 traces from specific project
    python3 download_langsmith_traces.py --project my-project --limit 100

    # Download traces from specific date range
    python3 download_langsmith_traces.py --start-date 2024-01-01 --end-date 2024-01-02

    # Download only LLM traces
    python3 download_langsmith_traces.py --run-type llm --limit 50

Requirements:
    pip install langsmith python-dotenv tqdm
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from langsmith import Client
    from dotenv import load_dotenv
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required package. Please install with:")
    print("pip install langsmith python-dotenv tqdm")
    sys.exit(1)


class LangSmithTraceDownloader:
    """Downloads traces from LangSmith with various filtering options."""
    
    def __init__(self, api_key: str, api_url: Optional[str] = None):
        """Initialize the LangSmith client."""
        self.client = Client(api_key=api_key, api_url=api_url)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('langsmith_downloader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _create_output_directory(self, output_dir: Path) -> None:
        """Create output directory if it doesn't exist."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {output_dir}")
    
    def _generate_filename(self, project_name: str, filters: Dict[str, Any]) -> str:
        """Generate filename with timestamp and filter info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add filter info to filename
        filter_parts = []
        if filters.get('limit'):
            filter_parts.append(f"limit{filters['limit']}")
        if filters.get('run_type'):
            filter_parts.append(f"type_{filters['run_type']}")
        if filters.get('hours'):
            filter_parts.append(f"last{filters['hours']}h")
        elif filters.get('days'):
            filter_parts.append(f"last{filters['days']}d")
        
        filter_str = "_" + "_".join(filter_parts) if filter_parts else ""
        
        return f"langsmith_traces_{project_name}_{timestamp}{filter_str}.json"
    
    def _parse_datetime(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {date_str}")
        except ValueError as e:
            self.logger.error(f"Date parsing error: {e}")
            raise
    
    def _build_filters(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Build filter dictionary from command line arguments."""
        filters = {}
        
        # Time range filters
        if args.start_date and args.end_date:
            filters['start_time'] = self._parse_datetime(args.start_date)
            filters['end_time'] = self._parse_datetime(args.end_date)
        elif args.hours:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=args.hours)
            filters['start_time'] = start_time
            filters['end_time'] = end_time
            filters['hours'] = args.hours
        elif args.days:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=args.days)
            filters['start_time'] = start_time
            filters['end_time'] = end_time
            filters['days'] = args.days
        
        # Other filters
        if args.run_type:
            filters['run_type'] = args.run_type
        if args.limit:
            filters['limit'] = args.limit
        if args.session_id:
            filters['session_id'] = args.session_id
        if args.filter_string:
            filters['filter'] = args.filter_string
            
        return filters
    
    def _serialize_run_data(self, run: Any) -> Dict[str, Any]:
        """Serialize a run object to a JSON-serializable dictionary."""
        try:
            # Convert the run to a dictionary
            run_dict = {}
            
            # Basic run information
            for attr in ['id', 'name', 'run_type', 'start_time', 'end_time', 
                        'status', 'error', 'execution_order', 'session_id',
                        'parent_run_id', 'child_run_ids', 'tags', 'extra']:
                if hasattr(run, attr):
                    value = getattr(run, attr)
                    if isinstance(value, datetime):
                        run_dict[attr] = value.isoformat()
                    else:
                        run_dict[attr] = value
            
            # Inputs and outputs
            if hasattr(run, 'inputs') and run.inputs:
                run_dict['inputs'] = run.inputs
            if hasattr(run, 'outputs') and run.outputs:
                run_dict['outputs'] = run.outputs
                
            # Events (for streaming runs)
            if hasattr(run, 'events') and run.events:
                run_dict['events'] = [
                    {
                        'name': event.name if hasattr(event, 'name') else str(event),
                        'time': event.time.isoformat() if hasattr(event, 'time') and event.time else None,
                        'data': getattr(event, 'data', None)
                    } for event in run.events
                ]
            
            # Feedback (if available)
            if hasattr(run, 'feedback_stats'):
                run_dict['feedback_stats'] = run.feedback_stats
                
            return run_dict
            
        except Exception as e:
            self.logger.warning(f"Error serializing run {getattr(run, 'id', 'unknown')}: {e}")
            return {
                'id': getattr(run, 'id', 'unknown'),
                'error': f"Serialization error: {str(e)}",
                'raw_type': str(type(run))
            }
    
    def download_traces(self, 
                       project_name: str, 
                       filters: Dict[str, Any], 
                       output_dir: Path) -> Dict[str, Any]:
        """Download traces based on filters and save to JSON file."""
        
        self.logger.info(f"Starting trace download for project: {project_name}")
        self.logger.info(f"Filters: {json.dumps({k: str(v) for k, v in filters.items()}, indent=2)}")
        
        try:
            # Build query parameters
            query_params = {
                'project_name': project_name,
            }
            
            # Add time filters
            if 'start_time' in filters:
                query_params['start_time'] = filters['start_time']
            if 'end_time' in filters:
                query_params['end_time'] = filters['end_time']
                
            # Add other filters
            if 'run_type' in filters:
                query_params['run_type'] = filters['run_type']
            if 'session_id' in filters:
                query_params['session_id'] = filters['session_id']
            if 'filter' in filters:
                query_params['filter'] = filters['filter']
            
            # Get runs from LangSmith
            self.logger.info("Fetching runs from LangSmith...")
            
            runs = []
            run_count = 0
            limit = filters.get('limit', 1000)  # Default limit
            
            # Use list_runs to get runs
            for run in self.client.list_runs(**query_params):
                if run_count >= limit:
                    break
                    
                runs.append(self._serialize_run_data(run))
                run_count += 1
                
                # Show progress
                if run_count % 10 == 0:
                    print(f"Downloaded {run_count} traces...", end='\r')
            
            print(f"\nDownloaded {len(runs)} traces total")
            
            if not runs:
                self.logger.warning("No traces found matching the criteria")
                return {
                    'project_name': project_name,
                    'filters': filters,
                    'trace_count': 0,
                    'traces': [],
                    'summary': 'No traces found'
                }
            
            # Prepare output data
            output_data = {
                'metadata': {
                    'project_name': project_name,
                    'download_time': datetime.now().isoformat(),
                    'filters_applied': filters,
                    'total_traces': len(runs),
                },
                'traces': runs
            }
            
            # Generate filename and save
            filename = self._generate_filename(project_name, filters)
            output_path = output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate summary
            summary = self._generate_summary(runs, filters)
            
            self.logger.info(f"Traces saved to: {output_path}")
            self.logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            return {
                'project_name': project_name,
                'filters': filters,
                'trace_count': len(runs),
                'output_file': str(output_path),
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Error downloading traces: {e}")
            raise
    
    def _generate_summary(self, traces: List[Dict], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the downloaded traces."""
        if not traces:
            return {'message': 'No traces to summarize'}
        
        summary = {
            'total_traces': len(traces),
            'run_types': {},
            'status_counts': {},
            'time_range': {
                'earliest': None,
                'latest': None
            },
            'session_count': 0,
            'error_count': 0
        }
        
        sessions = set()
        earliest_time = None
        latest_time = None
        
        for trace in traces:
            # Count run types
            run_type = trace.get('run_type', 'unknown')
            summary['run_types'][run_type] = summary['run_types'].get(run_type, 0) + 1
            
            # Count statuses
            status = trace.get('status', 'unknown')
            summary['status_counts'][status] = summary['status_counts'].get(status, 0) + 1
            
            # Count errors
            if trace.get('error') or status == 'error':
                summary['error_count'] += 1
            
            # Track sessions
            if trace.get('session_id'):
                sessions.add(trace['session_id'])
            
            # Track time range
            start_time_str = trace.get('start_time')
            if start_time_str:
                try:
                    if isinstance(start_time_str, str):
                        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    else:
                        start_time = start_time_str
                        
                    if earliest_time is None or start_time < earliest_time:
                        earliest_time = start_time
                    if latest_time is None or start_time > latest_time:
                        latest_time = start_time
                except Exception:
                    pass
        
        summary['session_count'] = len(sessions)
        if earliest_time:
            summary['time_range']['earliest'] = earliest_time.isoformat()
        if latest_time:
            summary['time_range']['latest'] = latest_time.isoformat()
        
        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Download traces from LangSmith',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download last 24 hours of traces:
    python3 download_langsmith_traces.py --hours 24

  Download last 100 traces from specific project:
    python3 download_langsmith_traces.py --project my-project --limit 100

  Download traces from specific date range:
    python3 download_langsmith_traces.py --start-date 2024-01-01 --end-date 2024-01-02

  Download only LLM traces:
    python3 download_langsmith_traces.py --run-type llm --limit 50
        """
    )
    
    # Project settings
    parser.add_argument(
        '--project', '-p',
        default='ehs-ai-demo-ingestion',
        help='LangSmith project name (default: ehs-ai-demo-ingestion)'
    )
    
    # Time range filters
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        '--hours',
        type=int,
        help='Download traces from last N hours'
    )
    time_group.add_argument(
        '--days',
        type=int,
        help='Download traces from last N days'
    )
    
    # Date range filters
    parser.add_argument(
        '--start-date',
        help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)'
    )
    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)'
    )
    
    # Other filters
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=1000,
        help='Maximum number of traces to download (default: 1000)'
    )
    parser.add_argument(
        '--run-type',
        choices=['llm', 'chain', 'tool', 'retriever', 'embedding', 'prompt', 'parser'],
        help='Filter by run type'
    )
    parser.add_argument(
        '--session-id',
        help='Filter by specific session ID'
    )
    parser.add_argument(
        '--filter-string',
        help='Custom filter string (LangSmith query syntax)'
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('./traces_output'),
        help='Output directory for trace files (default: ./traces_output)'
    )
    
    # API settings
    parser.add_argument(
        '--api-key',
        help='LangSmith API key (or set LANGSMITH_API_KEY environment variable)'
    )
    parser.add_argument(
        '--api-url',
        help='LangSmith API URL (optional)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger('langsmith_downloader').setLevel(logging.DEBUG)
    
    # Get API key
    api_key = args.api_key or os.getenv('LANGSMITH_API_KEY')
    if not api_key:
        print("Error: LangSmith API key not found!")
        print("Set LANGSMITH_API_KEY environment variable or use --api-key argument")
        sys.exit(1)
    
    # Validate date arguments
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        print("Error: Both --start-date and --end-date must be provided together")
        sys.exit(1)
    
    # Set default time range if none specified
    if not any([args.hours, args.days, args.start_date]):
        args.hours = 24  # Default to last 24 hours
        print("No time range specified, defaulting to last 24 hours")
    
    try:
        # Initialize downloader
        downloader = LangSmithTraceDownloader(api_key, args.api_url)
        
        # Create output directory
        downloader._create_output_directory(args.output_dir)
        
        # Build filters
        filters = downloader._build_filters(args)
        
        # Download traces
        print(f"Downloading traces from project: {args.project}")
        print(f"Output directory: {args.output_dir}")
        print("-" * 50)
        
        result = downloader.download_traces(args.project, filters, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 50)
        print("DOWNLOAD COMPLETE")
        print("=" * 50)
        print(f"Project: {result['project_name']}")
        print(f"Traces downloaded: {result['trace_count']}")
        if result['trace_count'] > 0:
            print(f"Output file: {result['output_file']}")
            
            summary = result['summary']
            print(f"\nSummary:")
            print(f"  Total traces: {summary['total_traces']}")
            print(f"  Sessions: {summary['session_count']}")
            print(f"  Errors: {summary['error_count']}")
            
            print(f"  Run types:")
            for run_type, count in summary['run_types'].items():
                print(f"    {run_type}: {count}")
            
            if summary['time_range']['earliest']:
                print(f"  Time range: {summary['time_range']['earliest']} to {summary['time_range']['latest']}")
        
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()