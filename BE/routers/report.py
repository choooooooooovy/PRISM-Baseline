from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import logging
from utils.logger import log_report_data

logger = logging.getLogger(__name__)

router = APIRouter()

class ReportDataRequest(BaseModel):
    sessionId: str
    step0: Dict[str, Any]
    step1: Dict[str, Any]
    step2: Dict[str, Any]
    step3: Dict[str, Any]
    step4: Dict[str, Any]

class ReportDataResponse(BaseModel):
    success: bool
    message: str

@router.post("/save-report", response_model=ReportDataResponse)
async def save_report_data(request: ReportDataRequest):
    """
    Save complete worksheet data (Steps 0-4) to log file when user views report
    """
    try:
        logger.info(f"Saving report data for session {request.sessionId}")
        
        # Log the complete report data
        log_report_data(
            session_id=request.sessionId,
            step0_data=request.step0,
            step1_data=request.step1,
            step2_data=request.step2,
            step3_data=request.step3,
            step4_data=request.step4
        )
        
        logger.info(f"Report data saved successfully for session {request.sessionId}")
        
        return ReportDataResponse(
            success=True,
            message="Report data saved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error saving report data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
