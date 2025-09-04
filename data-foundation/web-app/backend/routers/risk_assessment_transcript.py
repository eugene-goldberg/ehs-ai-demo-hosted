"""
Router for Risk Assessment Agent Transcript
Serves the markdown content from the docs directory
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_risk_assessment_transcript():
    """Get the Risk Assessment Agent Transcript markdown content"""
    try:
        # Path to the markdown file
        markdown_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/docs/RISK_ASSESSMENT_AGENT_TRANSCRIPT.md"
        
        # Check if file exists
        if not os.path.exists(markdown_path):
            logger.error(f"Risk Assessment Agent Transcript file not found at: {markdown_path}")
            raise HTTPException(
                status_code=404, 
                detail="Risk Assessment Agent Transcript file not found"
            )
        
        # Read the markdown content
        with open(markdown_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        
        logger.info("Successfully read Risk Assessment Agent Transcript")
        
        # Return as JSON response
        return JSONResponse(content={
            "markdown_content": markdown_content,
            "status": "success"
        })
        
    except FileNotFoundError:
        logger.error(f"Risk Assessment Agent Transcript file not found: {markdown_path}")
        raise HTTPException(
            status_code=404, 
            detail="Risk Assessment Agent Transcript file not found"
        )
    except Exception as e:
        logger.error(f"Error reading Risk Assessment Agent Transcript: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading Risk Assessment Agent Transcript: {str(e)}"
        )