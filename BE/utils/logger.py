import logging
import json
import os
from datetime import datetime
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log'),
            logging.StreamHandler()
        ]
    )

def log_user_activity(session_id: str, activity_type: str, data: dict):
    """Log user activity to session-specific JSON file"""
    session_dir = Path("logs") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = session_dir / "user_activity.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "activity_type": activity_type,
        "data": data
    }
    
    # Append to JSON file
    logs = []
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def log_llm_generation(session_id: str, prompt: str, response: str, model: str, tokens_used: dict):
    """Log LLM generation details to session-specific JSON file"""
    session_dir = Path("logs") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = session_dir / "llm_generations.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "tokens_used": tokens_used,
        "prompt": prompt,
        "response": response
    }
    
    logs = []
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def log_report_data(session_id: str, step0_data: dict, step1_data: dict, step2_data: dict, step3_data: dict, step4_data: dict):
    """Log complete report data (Steps 0-4) to session-specific JSON file"""
    session_dir = Path("logs") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = session_dir / "report_data.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step0": step0_data,
        "step1": step1_data,
        "step2": step2_data,
        "step3": step3_data,
        "step4": step4_data
    }
    
    # For report data, we save the complete snapshot (not append)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, ensure_ascii=False, indent=2)
