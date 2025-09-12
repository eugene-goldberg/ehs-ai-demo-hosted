"""
LangSmith Conversations API - Simple file reader implementation
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# Create router
router = APIRouter()

# Constants
CONVERSATIONS_OUTPUT_PATH = "/home/azureuser/dev/ehs-ai-demo/data-foundation/traces_output/gpt4_conversations_extracted.md"

@router.get("/")
async def get_conversations_markdown():
    """
    Read the markdown file and return its content as JSON.
    
    Returns:
        JSON response with markdown_content field containing the file contents
    """
    try:
        # Check if file exists
        if not os.path.exists(CONVERSATIONS_OUTPUT_PATH):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "File not found",
                    "message": f"The conversations file does not exist at: {CONVERSATIONS_OUTPUT_PATH}"
                }
            )
        
        # Read the file content
        with open(CONVERSATIONS_OUTPUT_PATH, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        return {
            "markdown_content": markdown_content
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "File not found",
                "message": f"The conversations file does not exist at: {CONVERSATIONS_OUTPUT_PATH}"
            }
        )
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Permission denied",
                "message": f"Cannot read the conversations file: insufficient permissions"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": f"Failed to read conversations file: {str(e)}"
            }
        )
