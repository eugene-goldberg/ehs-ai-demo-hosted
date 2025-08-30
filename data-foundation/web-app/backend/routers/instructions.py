from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import os
import asyncio
from pathlib import Path

router = APIRouter()

@router.get("/ingestion-instructions", response_model=Dict[str, Any])
async def get_ingestion_instructions():
    """
    Get ingestion instructions from the markdown file.
    Returns the content of the ingestion_llm_instructions.md file as JSON.
    """
    try:
        # Path to the markdown file relative to the backend directory
        docs_path = Path(__file__).parent.parent.parent.parent / "docs" / "ingestion_llm_instructions.md"
        
        # Check if file exists
        if not docs_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Ingestion instructions file not found at {docs_path}"
            )
        
        # Read the file content
        with open(docs_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Return as JSON
        return {
            "success": True,
            "content": content,
            "file_path": str(docs_path),
            "content_length": len(content)
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Ingestion instructions file not found"
        )
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Permission denied accessing ingestion instructions file"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading ingestion instructions: {str(e)}"
        )