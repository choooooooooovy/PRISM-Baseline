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
    """Log user activity to JSON file"""
    log_dir = Path("logs")
    log_file = log_dir / f"user_activity_{datetime.now().strftime('%Y%m%d')}.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
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
    """Log LLM generation details"""
    log_dir = Path("logs")
    log_file = log_dir / f"llm_generations_{datetime.now().strftime('%Y%m%d')}.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
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
