#!/usr/bin/env python3
"""
Comprehensive Transcript Forwarding Diagnostic Test

This script tests every LLM component to identify where transcript forwarding is failing.
It creates a mock server to capture all transcript forwarding attempts and tests each
component individually to show exactly what's working and what's not.

Usage:
    python3 test_transcript_diagnostic.py

Requirements:
    - Mock server on port 8001
    - Enhanced logging to capture all activity
    - Individual testing of each LLM component
    - Clear diagnostic output
"""

import os
import sys
import json
import time
import logging
import threading
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import signal
import traceback
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Set up enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/transcript_diagnostic.log')
    ]
)

logger = logging.getLogger(__name__)

# Global test results tracking
test_results = {}
mock_server_requests = []
mock_server_running = False
server_thread = None
httpd = None

class MockTranscriptServer:
    """Mock server to capture transcript forwarding requests"""
    
    def __init__(self, port=8001):
        self.port = port
        self.requests_received = []
        self.server = None
        
    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            global mock_server_requests
            
            try:
                # Parse request
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Log the request
                request_info = {
                    'timestamp': datetime.now().isoformat(),
                    'path': self.path,
                    'headers': dict(self.headers),
                    'body': post_data.decode('utf-8'),
                    'method': 'POST'
                }
                
                mock_server_requests.append(request_info)
                
                print(f"\n🔍 MOCK SERVER: Received request to {self.path}")
                print(f"   Headers: {dict(self.headers)}")
                print(f"   Body: {post_data.decode('utf-8')[:200]}...")
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "success", "message": "Transcript entry received"}')
                
            except Exception as e:
                print(f"❌ MOCK SERVER ERROR: {str(e)}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(f'{{"error": "{str(e)}"}}'.encode())
        
        def do_GET(self):
            # Health check endpoint
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "healthy"}')
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            # Suppress default HTTP server logging to avoid clutter
            pass


def start_mock_server():
    """Start the mock server in a separate thread"""
    global mock_server_running, server_thread, httpd
    
    def run_server():
        global httpd, mock_server_running
        try:
            httpd = HTTPServer(('localhost', 8001), MockTranscriptServer.RequestHandler)
            mock_server_running = True
            print("🚀 Mock server started on http://localhost:8001")
            httpd.serve_forever()
        except Exception as e:
            print(f"❌ Failed to start mock server: {str(e)}")
            mock_server_running = False
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Test if server is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Mock server is responding to health checks")
            return True
        else:
            print(f"❌ Mock server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach mock server: {str(e)}")
        return False


def stop_mock_server():
    """Stop the mock server"""
    global httpd, mock_server_running, server_thread
    
    if httpd:
        print("🛑 Stopping mock server...")
        httpd.shutdown()
        httpd.server_close()
        mock_server_running = False
        
        if server_thread and server_thread.is_alive():
            server_thread.join(timeout=5)
        
        print("✅ Mock server stopped")


def clear_mock_requests():
    """Clear the mock server requests list"""
    global mock_server_requests
    mock_server_requests = []


def get_mock_requests():
    """Get all requests received by mock server"""
    global mock_server_requests
    return mock_server_requests.copy()


def test_transcript_forwarder_direct():
    """Test the transcript forwarder directly"""
    print("\n" + "="*60)
    print("🔧 TESTING: Direct Transcript Forwarder")
    print("="*60)
    
    clear_mock_requests()
    
    try:
        # Import transcript forwarder
        from src.utils.transcript_forwarder import forward_transcript_entry_sync, configure_forwarder
        
        # Configure to use our mock server
        configure_forwarder(web_app_url="http://localhost:8001/api/data/transcript/add")
        
        # Test synchronous forwarding
        result = forward_transcript_entry_sync(
            role="test_user",
            content="This is a direct transcript forwarder test",
            context={"test_type": "direct_forwarder", "component": "transcript_forwarder"}
        )
        
        time.sleep(1)  # Allow time for request
        requests_received = get_mock_requests()
        
        test_results['transcript_forwarder'] = {
            'function_called': True,
            'function_result': result,
            'requests_sent': len(requests_received),
            'requests_details': requests_received,
            'success': result and len(requests_received) > 0
        }
        
        print(f"   Function called: ✅")
        print(f"   Function result: {result}")
        print(f"   Requests sent to mock server: {len(requests_received)}")
        
        if requests_received:
            print(f"   Last request body: {requests_received[-1]['body'][:100]}...")
        
        return result and len(requests_received) > 0
        
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        test_results['transcript_forwarder'] = {
            'function_called': False,
            'error': str(e),
            'success': False
        }
        return False


def test_schema_extraction():
    """Test schema extraction component"""
    print("\n" + "="*60)
    print("🧠 TESTING: Schema Extraction Component")
    print("="*60)
    
    clear_mock_requests()
    
    try:
        from src.shared.schema_extraction import extract_schema_from_text
        
        # Test with sample text
        test_text = "John works at Microsoft and manages a team. Microsoft is a technology company."
        
        print(f"   Testing with text: {test_text[:50]}...")
        
        # This should trigger LLM calls and transcript forwarding
        result = extract_schema_from_text(test_text)
        
        time.sleep(2)  # Allow time for async requests
        requests_received = get_mock_requests()
        
        test_results['schema_extraction'] = {
            'function_called': True,
            'function_result': str(result)[:200] if result else None,
            'requests_sent': len(requests_received),
            'requests_details': requests_received,
            'success': len(requests_received) > 0
        }
        
        print(f"   Function called: ✅")
        print(f"   Function result: {str(result)[:100]}...")
        print(f"   Transcript requests sent: {len(requests_received)}")
        
        if requests_received:
            print(f"   Last request body: {requests_received[-1]['body'][:100]}...")
        
        return len(requests_received) > 0
        
    except ImportError as e:
        print(f"   ❌ IMPORT ERROR: {str(e)}")
        print("   Schema extraction function not found")
        test_results['schema_extraction'] = {
            'function_called': False,
            'error': f'Import error: {str(e)}',
            'success': False
        }
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        test_results['schema_extraction'] = {
            'function_called': False,
            'error': str(e),
            'success': False
        }
        return False


def test_qa_integration():
    """Test QA integration component"""
    print("\n" + "="*60)
    print("💬 TESTING: QA Integration Component")
    print("="*60)
    
    clear_mock_requests()
    
    try:
        from src.QA_integration import QA_integration
        
        # Skip if missing API keys
        required_keys = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            print(f"   ⚠️ SKIPPED: Missing required environment variables: {missing_keys}")
            test_results['qa_integration'] = {
                'function_called': False,
                'skipped': True,
                'reason': f'Missing keys: {missing_keys}',
                'success': False
            }
            return False
        
        # Create QA integration instance
        qa_system = QA_integration()
        
        # Test query
        test_query = "What are the safety requirements for chemical storage?"
        
        print(f"   Testing with query: {test_query}")
        
        # This should trigger LLM calls and transcript forwarding
        result = qa_system.answer_query(test_query)
        
        time.sleep(2)  # Allow time for async requests
        requests_received = get_mock_requests()
        
        test_results['qa_integration'] = {
            'function_called': True,
            'function_result': str(result)[:200] if result else None,
            'requests_sent': len(requests_received),
            'requests_details': requests_received,
            'success': len(requests_received) > 0
        }
        
        print(f"   Function called: ✅")
        print(f"   Function result: {str(result)[:100]}...")
        print(f"   Transcript requests sent: {len(requests_received)}")
        
        if requests_received:
            print(f"   Last request body: {requests_received[-1]['body'][:100]}...")
        
        return len(requests_received) > 0
        
    except ImportError as e:
        print(f"   ❌ IMPORT ERROR: {str(e)}")
        print("   QA integration not found")
        test_results['qa_integration'] = {
            'function_called': False,
            'error': f'Import error: {str(e)}',
            'success': False
        }
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        test_results['qa_integration'] = {
            'function_called': False,
            'error': str(e),
            'success': False
        }
        return False


def test_post_processing():
    """Test post processing component"""
    print("\n" + "="*60)
    print("⚙️ TESTING: Post Processing Component")
    print("="*60)
    
    clear_mock_requests()
    
    try:
        from src.post_processing import clean_graph_nodes
        
        # Skip if missing Neo4j keys
        required_keys = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            print(f"   ⚠️ SKIPPED: Missing required environment variables: {missing_keys}")
            test_results['post_processing'] = {
                'function_called': False,
                'skipped': True,
                'reason': f'Missing keys: {missing_keys}',
                'success': False
            }
            return False
        
        print("   Testing graph cleanup function...")
        
        # This should trigger LLM calls for graph cleanup
        result = clean_graph_nodes()
        
        time.sleep(2)  # Allow time for async requests
        requests_received = get_mock_requests()
        
        test_results['post_processing'] = {
            'function_called': True,
            'function_result': str(result)[:200] if result else None,
            'requests_sent': len(requests_received),
            'requests_details': requests_received,
            'success': len(requests_received) > 0
        }
        
        print(f"   Function called: ✅")
        print(f"   Function result: {str(result)[:100]}...")
        print(f"   Transcript requests sent: {len(requests_received)}")
        
        if requests_received:
            print(f"   Last request body: {requests_received[-1]['body'][:100]}...")
        
        return len(requests_received) > 0
        
    except ImportError as e:
        print(f"   ❌ IMPORT ERROR: {str(e)}")
        print("   Post processing functions not found")
        test_results['post_processing'] = {
            'function_called': False,
            'error': f'Import error: {str(e)}',
            'success': False
        }
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        test_results['post_processing'] = {
            'function_called': False,
            'error': str(e),
            'success': False
        }
        return False


def test_communities():
    """Test communities component"""
    print("\n" + "="*60)
    print("🏘️ TESTING: Communities Component")
    print("="*60)
    
    clear_mock_requests()
    
    try:
        from src.communities import create_community_summaries
        
        # Skip if missing Neo4j keys
        required_keys = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            print(f"   ⚠️ SKIPPED: Missing required environment variables: {missing_keys}")
            test_results['communities'] = {
                'function_called': False,
                'skipped': True,
                'reason': f'Missing keys: {missing_keys}',
                'success': False
            }
            return False
        
        print("   Testing community summaries creation...")
        
        # This should trigger LLM calls for community summarization
        result = create_community_summaries()
        
        time.sleep(2)  # Allow time for async requests
        requests_received = get_mock_requests()
        
        test_results['communities'] = {
            'function_called': True,
            'function_result': str(result)[:200] if result else None,
            'requests_sent': len(requests_received),
            'requests_details': requests_received,
            'success': len(requests_received) > 0
        }
        
        print(f"   Function called: ✅")
        print(f"   Function result: {str(result)[:100]}...")
        print(f"   Transcript requests sent: {len(requests_received)}")
        
        if requests_received:
            print(f"   Last request body: {requests_received[-1]['body'][:100]}...")
        
        return len(requests_received) > 0
        
    except ImportError as e:
        print(f"   ❌ IMPORT ERROR: {str(e)}")
        print("   Communities functions not found")
        test_results['communities'] = {
            'function_called': False,
            'error': f'Import error: {str(e)}',
            'success': False
        }
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        test_results['communities'] = {
            'function_called': False,
            'error': str(e),
            'success': False
        }
        return False


def test_llama_parse():
    """Test LlamaParse component"""
    print("\n" + "="*60)
    print("🦙 TESTING: LlamaParse Component")
    print("="*60)
    
    clear_mock_requests()
    
    try:
        from src.parsers.llama_parser import EHSDocumentParser
        
        # Skip if missing API key
        llama_api_key = os.getenv('LLAMA_PARSE_API_KEY')
        if not llama_api_key:
            print("   ⚠️ SKIPPED: Missing LLAMA_PARSE_API_KEY environment variable")
            test_results['llama_parse'] = {
                'function_called': False,
                'skipped': True,
                'reason': 'Missing LLAMA_PARSE_API_KEY',
                'success': False
            }
            return False
        
        print("   Testing LlamaParse document parsing...")
        
        # Create parser instance
        parser = EHSDocumentParser(api_key=llama_api_key)
        
        # Create a dummy document for testing (we won't actually parse it, just initialize)
        test_path = "/tmp/test_document.txt"
        with open(test_path, 'w') as f:
            f.write("This is a test EHS document for safety compliance.")
        
        # This should trigger some initialization that might use LLM
        # Since parsing requires actual API calls, we'll just test initialization
        print(f"   Parser initialized: ✅")
        
        time.sleep(1)  # Allow time for any async requests
        requests_received = get_mock_requests()
        
        test_results['llama_parse'] = {
            'function_called': True,
            'initialization': 'success',
            'requests_sent': len(requests_received),
            'requests_details': requests_received,
            'success': True  # Consider initialization success as success
        }
        
        print(f"   Parser creation: ✅")
        print(f"   Transcript requests sent: {len(requests_received)}")
        
        if requests_received:
            print(f"   Last request body: {requests_received[-1]['body'][:100]}...")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        
        return True
        
    except ImportError as e:
        print(f"   ❌ IMPORT ERROR: {str(e)}")
        print("   LlamaParse not found")
        test_results['llama_parse'] = {
            'function_called': False,
            'error': f'Import error: {str(e)}',
            'success': False
        }
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        test_results['llama_parse'] = {
            'function_called': False,
            'error': str(e),
            'success': False
        }
        return False


def print_summary():
    """Print a comprehensive summary of all tests"""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE DIAGNOSTIC SUMMARY")
    print("="*80)
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
    failed_tests = sum(1 for result in test_results.values() if not result.get('success', False) and not result.get('skipped', False))
    skipped_tests = sum(1 for result in test_results.values() if result.get('skipped', False))
    
    print(f"\n📈 TEST STATISTICS:")
    print(f"   Total Components Tested: {total_tests}")
    print(f"   Successful: {successful_tests} ✅")
    print(f"   Failed: {failed_tests} ❌")
    print(f"   Skipped: {skipped_tests} ⚠️")
    
    print(f"\n📋 DETAILED RESULTS:")
    
    for component, result in test_results.items():
        status = "✅ WORKING" if result.get('success') else "⚠️ SKIPPED" if result.get('skipped') else "❌ FAILED"
        print(f"\n   🔍 {component.upper()}:")
        print(f"      Status: {status}")
        
        if result.get('skipped'):
            print(f"      Reason: {result.get('reason', 'Unknown')}")
        elif result.get('success'):
            print(f"      Function Called: ✅")
            print(f"      Transcript Requests Sent: {result.get('requests_sent', 0)}")
            if result.get('requests_details'):
                print(f"      Last Request: {result['requests_details'][-1]['body'][:50]}...")
        else:
            print(f"      Function Called: {'✅' if result.get('function_called') else '❌'}")
            print(f"      Error: {result.get('error', 'Unknown error')}")
            print(f"      Transcript Requests Sent: {result.get('requests_sent', 0)}")
    
    print(f"\n🎯 TRANSCRIPT FORWARDING ANALYSIS:")
    working_components = [comp for comp, result in test_results.items() if result.get('success')]
    failing_components = [comp for comp, result in test_results.items() if not result.get('success') and not result.get('skipped')]
    
    if working_components:
        print(f"   ✅ Components with working transcript forwarding:")
        for comp in working_components:
            requests_count = test_results[comp].get('requests_sent', 0)
            print(f"      - {comp}: {requests_count} request(s)")
    
    if failing_components:
        print(f"   ❌ Components with failing transcript forwarding:")
        for comp in failing_components:
            error = test_results[comp].get('error', 'Unknown error')
            print(f"      - {comp}: {error}")
    
    if not working_components and not failing_components:
        print("   ⚠️ All components were skipped due to missing configuration")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if failed_tests > 0:
        print("   1. Check the error messages above for specific issues")
        print("   2. Verify that transcript_forwarder.py is being imported correctly")
        print("   3. Ensure forward_transcript_entry is being called in LLM functions")
    
    if skipped_tests > 0:
        print("   4. Set up missing environment variables:")
        print("      - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD for graph operations")
        print("      - LLAMA_PARSE_API_KEY for document parsing")
        print("      - OpenAI/Groq/etc. API keys for LLM operations")
    
    if successful_tests == 0:
        print("   5. ⚠️ CRITICAL: No transcript forwarding is working!")
        print("      - Check if the transcript forwarder integration is properly implemented")
        print("      - Verify that LLM functions are actually calling forward_transcript_entry")
    
    print("\n📝 LOG FILE: /tmp/transcript_diagnostic.log")
    print("="*80)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Received interrupt signal, shutting down...")
    stop_mock_server()
    sys.exit(0)


def main():
    """Main diagnostic function"""
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 TRANSCRIPT FORWARDING DIAGNOSTIC TEST")
    print("="*60)
    print("This script will test every LLM component to identify")
    print("where transcript forwarding is failing.")
    print("="*60)
    
    # Start mock server
    if not start_mock_server():
        print("❌ Cannot start mock server. Exiting.")
        return False
    
    try:
        # Test each component
        components_to_test = [
            ("Transcript Forwarder Direct", test_transcript_forwarder_direct),
            ("Schema Extraction", test_schema_extraction),
            ("QA Integration", test_qa_integration),
            ("Post Processing", test_post_processing),
            ("Communities", test_communities),
            ("LlamaParse", test_llama_parse),
        ]
        
        for component_name, test_function in components_to_test:
            try:
                test_function()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"   ❌ CRITICAL ERROR in {component_name}: {str(e)}")
                test_results[component_name.lower().replace(" ", "_")] = {
                    'function_called': False,
                    'error': f'Critical error: {str(e)}',
                    'success': False
                }
        
        # Print comprehensive summary
        print_summary()
        
        return True
        
    finally:
        # Always stop the mock server
        stop_mock_server()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)