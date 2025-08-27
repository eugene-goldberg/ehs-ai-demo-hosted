#!/usr/bin/env python3
"""
Simple script to list LangSmith projects with details.

This script connects to LangSmith using the API key and lists all available projects
with their details including name, number of runs, and creation/update dates.
"""

import os
import sys
import requests
from datetime import datetime
from typing import Dict, List, Optional


def load_api_key() -> str:
    """Load LangSmith API key from environment variables."""
    api_key = os.getenv('LANGCHAIN_API_KEY')
    if not api_key:
        print("Error: LANGCHAIN_API_KEY environment variable not found")
        print("Please set your LangSmith API key in the environment or .env file")
        sys.exit(1)
    return api_key


def get_langsmith_projects(api_key: str, base_url: str = "https://api.smith.langchain.com") -> Optional[List[Dict]]:
    """
    Fetch all projects from LangSmith API.
    
    Args:
        api_key: LangSmith API key
        base_url: LangSmith API base URL
        
    Returns:
        List of project dictionaries or None if error
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Get projects
        response = requests.get(f"{base_url}/sessions", headers=headers, timeout=30)
        response.raise_for_status()
        
        projects = response.json()
        return projects
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching projects: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def format_date(date_str: Optional[str]) -> str:
    """Format ISO date string to readable format."""
    if not date_str:
        return "Unknown"
    
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return date_str


def display_projects(projects: List[Dict]) -> None:
    """
    Display project information in a formatted way.
    
    Args:
        projects: List of project dictionaries
    """
    if not projects:
        print("No projects found.")
        return
    
    print(f"\nFound {len(projects)} project(s):\n")
    print("=" * 80)
    
    for i, project in enumerate(projects, 1):
        print(f"\nProject #{i}:")
        print(f"  Name: {project.get('name', 'Unknown')}")
        print(f"  ID: {project.get('id', 'Unknown')}")
        print(f"  Description: {project.get('description', 'No description')}")
        print(f"  Run Count: {project.get('run_count', 'Unknown')}")
        print(f"  Created: {format_date(project.get('created_at'))}")
        print(f"  Last Updated: {format_date(project.get('modified_at'))}")
        
        # Additional details if available
        if project.get('tenant_id'):
            print(f"  Tenant ID: {project.get('tenant_id')}")
        
        if project.get('reference_dataset_id'):
            print(f"  Reference Dataset: {project.get('reference_dataset_id')}")
            
        print("-" * 40)


def main():
    """Main function to list LangSmith projects."""
    print("LangSmith Projects Listing Tool")
    print("=" * 40)
    
    # Load API key
    try:
        api_key = load_api_key()
    except SystemExit:
        return
    
    print("Connecting to LangSmith...")
    
    # Get projects
    projects = get_langsmith_projects(api_key)
    
    if projects is None:
        print("Failed to fetch projects. Please check your API key and connection.")
        sys.exit(1)
    
    # Display projects
    display_projects(projects)
    
    print(f"\nTotal projects: {len(projects)}")


if __name__ == "__main__":
    main()