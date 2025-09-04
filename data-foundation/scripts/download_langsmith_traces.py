#!/usr/bin/env python3
"""
LangSmith Traces Download Script

This script downloads traces from LangSmith for analysis and debugging purposes.
It supports various filtering options and saves traces to JSON files with timestamps.
Optionally, it can extract only the human-LLM conversation parts from the traces.

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

    # Extract only conversation data from LLM traces (defaults: limit=4, hours=1)
    python3 download_langsmith_traces.py --extract-conversations

    # Extract conversations with custom overrides
    python3 download_langsmith_traces.py --extract-conversations --limit 10 --hours 6

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
    
    def _generate_filename(self, project_name: str, filters: Dict[str, Any], conversation_mode: bool = False) -> str:
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
        
        # Add conversation suffix if in conversation mode
        conversation_suffix = "_conversations" if conversation_mode else ""
        
        return f"langsmith_traces_{project_name}_{timestamp}{filter_str}{conversation_suffix}.json"
    
    def _get_fixed_conversation_filename(self) -> str:
        """Get the fixed filename for conversation extraction."""
        return "gpt4_conversations_extracted.md"
    
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
    
    def _extract_conversation_from_run(self, run_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract conversation data from a single LLM run."""
        if run_data.get('run_type') != 'llm':
            return None
        
        conversation = {
            'run_id': run_data.get('id'),
            'session_id': run_data.get('session_id'),
            'start_time': run_data.get('start_time'),
            'end_time': run_data.get('end_time'),
            'status': run_data.get('status'),
            'human_input': None,
            'ai_response': None,
            'error': run_data.get('error')
        }
        
        # Extract human input from inputs
        inputs = run_data.get('inputs', {})
        if inputs:
            # Look for common input patterns
            if 'messages' in inputs:
                messages = inputs['messages']
                
                # Handle LangChain format: messages[0] is an array of message objects
                if isinstance(messages, list) and len(messages) > 0:
                    # Check if first element is another array (LangChain format)
                    if isinstance(messages[0], list):
                        message_list = messages[0]
                        # Look for HumanMessage in the list
                        for msg in message_list:
                            if isinstance(msg, dict):
                                # Handle LangChain message structure
                                if (msg.get('id', [None])[-1] == 'HumanMessage' or 
                                    msg.get('kwargs', {}).get('type') == 'human'):
                                    content = msg.get('kwargs', {}).get('content')
                                    if content:
                                        conversation['human_input'] = content
                                        break
                                # Handle simpler formats
                                elif msg.get('type') == 'human' or msg.get('role') == 'user':
                                    conversation['human_input'] = msg.get('content', str(msg))
                                    break
                    else:
                        # Handle flat message list
                        for message in reversed(messages):
                            if isinstance(message, dict):
                                if message.get('type') == 'human' or message.get('role') == 'user':
                                    conversation['human_input'] = message.get('content', str(message))
                                    break
                                elif 'content' in message:
                                    conversation['human_input'] = message['content']
                                    break
                elif isinstance(messages, dict):
                    conversation['human_input'] = str(messages)
                    
            elif 'input' in inputs:
                conversation['human_input'] = inputs['input']
            elif 'query' in inputs:
                conversation['human_input'] = inputs['query']
            elif 'prompt' in inputs:
                conversation['human_input'] = inputs['prompt']
            elif 'text' in inputs:
                conversation['human_input'] = inputs['text']
            else:
                # Fallback: use the entire input as string
                conversation['human_input'] = str(inputs)
        
        # Extract AI response from outputs
        outputs = run_data.get('outputs', {})
        if outputs:
            # Look for common output patterns
            if 'generations' in outputs:
                generations = outputs['generations']
                if isinstance(generations, list) and len(generations) > 0:
                    gen = generations[0]
                    if isinstance(gen, list) and len(gen) > 0:
                        # LangChain format
                        conversation['ai_response'] = gen[0].get('text', str(gen[0]))
                    elif isinstance(gen, dict):
                        conversation['ai_response'] = gen.get('text', str(gen))
            elif 'content' in outputs:
                conversation['ai_response'] = outputs['content']
            elif 'output' in outputs:
                conversation['ai_response'] = outputs['output']
            elif 'response' in outputs:
                conversation['ai_response'] = outputs['response']
            elif 'text' in outputs:
                conversation['ai_response'] = outputs['text']
            elif 'answer' in outputs:
                conversation['ai_response'] = outputs['answer']
            else:
                # Fallback: use the entire output as string
                conversation['ai_response'] = str(outputs)
        
        # Only return if we have at least some conversation content
        if conversation['human_input'] or conversation['ai_response']:
            return conversation
        
        return None
    
    def _extract_conversations(self, traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conversation data from all traces."""
        conversations = []
        
        for trace in traces:
            conversation = self._extract_conversation_from_run(trace)
            if conversation:
                conversations.append(conversation)
        
        return conversations
    
    def _format_conversations_as_markdown(self, conversations: List[Dict[str, Any]], 
                                         metadata: Dict[str, Any]) -> str:
        """Format conversations as markdown document."""
        md_content = []
        
        # Add header
        md_content.append("# GPT-4 Conversations Extracted from LangSmith")
        md_content.append("")
        
        # Add metadata
        md_content.append("## Extraction Metadata")
        md_content.append("")
        md_content.append(f"- **Project**: {metadata.get('project_name', 'N/A')}")
        md_content.append(f"- **Extraction Time**: {metadata.get('download_time', 'N/A')}")
        md_content.append(f"- **Total Traces Processed**: {metadata.get('total_traces_processed', 0)}")
        md_content.append(f"- **Conversations Extracted**: {metadata.get('conversations_extracted', 0)}")
        md_content.append("")
        
        # Add filter information if available
        if metadata.get('filters_applied'):
            md_content.append("### Applied Filters")
            md_content.append("")
            filters = metadata['filters_applied']
            for key, value in filters.items():
                if key not in ['start_time', 'end_time']:  # Skip datetime objects for readability
                    md_content.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            md_content.append("")
        
        # Add conversations
        md_content.append("## Conversations")
        md_content.append("")
        
        if not conversations:
            md_content.append("*No conversations found matching the criteria.*")
        else:
            for i, conv in enumerate(conversations, 1):
                md_content.append(f"### Conversation {i}")
                md_content.append("")
                
                # Add metadata for each conversation
                md_content.append("**Metadata:**")
                md_content.append(f"- Run ID: `{conv.get('run_id', 'N/A')}`")
                if conv.get('session_id'):
                    md_content.append(f"- Session ID: `{conv.get('session_id')}`")
                md_content.append(f"- Status: `{conv.get('status', 'N/A')}`")
                if conv.get('start_time'):
                    md_content.append(f"- Start Time: {conv.get('start_time')}")
                if conv.get('error'):
                    md_content.append(f"- Error: `{conv.get('error')}`")
                md_content.append("")
                
                # Add human input
                if conv.get('human_input'):
                    md_content.append("**Human Input:**")
                    md_content.append("")
                    human_input = str(conv['human_input']).strip()
                    # Format as code block if it looks like JSON, otherwise as quote
                    if human_input.startswith('{') or human_input.startswith('['):
                        md_content.append("```json")
                        md_content.append(human_input)
                        md_content.append("```")
                    else:
                        # Split into lines and add as blockquote
                        for line in human_input.split('\n'):
                            md_content.append(f"> {line}")
                    md_content.append("")
                
                # Add AI response
                if conv.get('ai_response'):
                    md_content.append("**AI Response:**")
                    md_content.append("")
                    ai_response = str(conv['ai_response']).strip()
                    # Format as code block if it looks like JSON, otherwise as regular text
                    if ai_response.startswith('{') or ai_response.startswith('['):
                        md_content.append("```json")
                        md_content.append(ai_response)
                        md_content.append("```")
                    else:
                        md_content.append(ai_response)
                    md_content.append("")
                
                # Add separator between conversations
                if i < len(conversations):
                    md_content.append("---")
                    md_content.append("")
        
        return '\n'.join(md_content)
    
    def download_traces(self, 
                       project_name: str, 
                       filters: Dict[str, Any], 
                       output_dir: Path,
                       extract_conversations: bool = False) -> Dict[str, Any]:
        """Download traces based on filters and save to JSON file."""
        
        self.logger.info(f"Starting trace download for project: {project_name}")
        if extract_conversations:
            self.logger.info("Conversation extraction mode enabled")
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
            elif extract_conversations:
                # If extracting conversations, automatically filter to LLM runs
                query_params['run_type'] = 'llm'
                
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
            
            # Extract conversations if requested
            if extract_conversations:
                self.logger.info("Extracting conversation data...")
                conversations = self._extract_conversations(runs)
                
                # Prepare metadata
                metadata = {
                    'project_name': project_name,
                    'download_time': datetime.now().isoformat(),
                    'filters_applied': filters,
                    'total_traces_processed': len(runs),
                    'conversations_extracted': len(conversations),
                    'extraction_mode': 'conversations_only'
                }
                
                # Format as markdown
                markdown_content = self._format_conversations_as_markdown(conversations, metadata)
                
                # Use fixed output path for conversations
                fixed_output_dir = Path("/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/traces_output")
                fixed_output_dir.mkdir(parents=True, exist_ok=True)
                output_path = fixed_output_dir / self._get_fixed_conversation_filename()
                
                # Save markdown file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                self.logger.info(f"Conversations saved to: {output_path}")
                self.logger.info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
                self.logger.info(f"Extracted {len(conversations)} conversations from {len(runs)} traces")
                
                return {
                    'project_name': project_name,
                    'filters': filters,
                    'trace_count': len(runs),
                    'conversation_count': len(conversations),
                    'output_file': str(output_path),
                    'summary': f"Extracted {len(conversations)} conversations from {len(runs)} LLM traces"
                }
            
            # Regular trace output (original functionality)
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

  Extract only conversations from LLM traces (defaults: limit=4, hours=1):
    python3 download_langsmith_traces.py --extract-conversations

  Extract conversations with custom filters:
    python3 download_langsmith_traces.py --extract-conversations --hours 12 --session-id abc123
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
        help='Maximum number of traces to download (default: 1000, or 4 when --extract-conversations is used)'
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
    
    # Conversation extraction option
    parser.add_argument(
        '--extract-conversations',
        action='store_true',
        help='Extract only human-AI conversation data from LLM traces (automatically filters to LLM runs, defaults to limit=4, hours=1). Outputs to fixed filename: /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/traces_output/gpt4_conversations_extracted.md'
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('./traces_output'),
        help='Output directory for trace files (default: ./traces_output). Note: --extract-conversations always uses fixed path.'
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
    
    # Apply conversation extraction defaults
    if args.extract_conversations:
        # Set default limit to 4 if not explicitly provided
        if args.limit is None:
            args.limit = 4
            print("Conversation extraction mode: Using default limit=4")
        
        # Set default time range to 1 hour if no time range specified
        if not any([args.hours, args.days, args.start_date]):
            args.hours = 1
            print("Conversation extraction mode: Using default hours=1")
        
        # Inform user about fixed output path
        fixed_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/traces_output/gpt4_conversations_extracted.md"
        print(f"Conversation extraction mode: Output will be saved to {fixed_path}")
    else:
        # Apply regular defaults for non-conversation mode
        if args.limit is None:
            args.limit = 1000
    
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
    
    # Set default time range if none specified (only for non-conversation mode)
    if not args.extract_conversations and not any([args.hours, args.days, args.start_date]):
        args.hours = 24  # Default to last 24 hours
        print("No time range specified, defaulting to last 24 hours")
    
    try:
        # Initialize downloader
        downloader = LangSmithTraceDownloader(api_key, args.api_url)
        
        # Create output directory (only for regular traces, conversations use fixed path)
        if not args.extract_conversations:
            downloader._create_output_directory(args.output_dir)
        
        # Build filters
        filters = downloader._build_filters(args)
        
        # Download traces
        print(f"Downloading traces from project: {args.project}")
        if args.extract_conversations:
            print(f"Mode: Conversation extraction (Markdown format, fixed output path)")
            print(f"Parameters: LLM traces only, limit={args.limit}, hours={args.hours if args.hours else 'custom range'}")
        else:
            print(f"Output directory: {args.output_dir}")
        print("-" * 50)
        
        result = downloader.download_traces(
            args.project, 
            filters, 
            args.output_dir,
            extract_conversations=args.extract_conversations
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("DOWNLOAD COMPLETE")
        print("=" * 50)
        print(f"Project: {result['project_name']}")
        
        if args.extract_conversations:
            print(f"Traces processed: {result['trace_count']}")
            print(f"Conversations extracted: {result.get('conversation_count', 0)}")
            print(f"Output format: Markdown")
        else:
            print(f"Traces downloaded: {result['trace_count']}")
            print(f"Output format: JSON")
        
        if result['trace_count'] > 0:
            print(f"Output file: {result['output_file']}")
            
            if not args.extract_conversations and 'summary' in result:
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
            elif args.extract_conversations:
                print(f"\nConversation extraction summary:")
                print(f"  {result.get('conversation_count', 0)} conversations extracted from {result['trace_count']} LLM traces")
                print(f"  Format: Markdown with structured conversation layout")
        
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()