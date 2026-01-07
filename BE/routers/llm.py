from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import json

from services.openai_service import OpenAIService
from utils.logger import log_user_activity, log_llm_generation

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response Models
class Step0Data(BaseModel):
    values: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    mustHaveConstraints: List[str] = Field(default_factory=list)
    niceToHaveConstraints: List[str] = Field(default_factory=list)
    concerns: Optional[str] = None

class Step1Data(BaseModel):
    problemDefinition: Optional[str] = None
    internalCues: List[str] = Field(default_factory=list)
    externalCues: List[str] = Field(default_factory=list)
    keyQuestions: List[str] = Field(default_factory=list)

class Step2Data(BaseModel):
    evaluationCriteria: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    informationTemplate: List[Dict[str, str]] = Field(default_factory=list)

class GenerateOptionsRequest(BaseModel):
    sessionId: str
    step0: Step0Data
    step1: Step1Data
    step2: Step2Data

class GenerateOptionsResponse(BaseModel):
    success: bool
    options: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    tokensUsed: Optional[Dict[str, int]] = None

# Initialize OpenAI service
openai_service = OpenAIService()

@router.post("/generate-options", response_model=GenerateOptionsResponse)
async def generate_options(request: GenerateOptionsRequest):
    """
    Generate career/decision options using LLM based on Steps 0-2 data
    """
    try:
        logger.info(f"Received generate options request for session: {request.sessionId}")
        
        # Log user activity
        log_user_activity(
            session_id=request.sessionId,
            activity_type="generate_options_request",
            data={
                "step0": request.step0.model_dump(),
                "step1": request.step1.model_dump(),
                "step2": request.step2.model_dump()
            }
        )
        
        # Call OpenAI service
        result = openai_service.generate_options(
            step0_data=request.step0.model_dump(),
            step1_data=request.step1.model_dump(),
            step2_data=request.step2.model_dump()
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "LLM generation failed"))
        
        # Parse JSON response
        try:
            content = result["content"]
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed_response = json.loads(content)
            options = parsed_response.get("options", [])
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.error(f"Raw response: {result['content']}")
            raise HTTPException(status_code=500, detail="Failed to parse LLM response")
        
        # Log LLM generation
        log_llm_generation(
            session_id=request.sessionId,
            prompt="Steps 0-2 data (see user_activity log)",
            response=content,
            model=result["model"],
            tokens_used=result["tokens_used"]
        )
        
        logger.info(f"Successfully generated {len(options)} options for session: {request.sessionId}")
        
        return GenerateOptionsResponse(
            success=True,
            options=options,
            tokensUsed=result["tokens_used"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_options: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
