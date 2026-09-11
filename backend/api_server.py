"""
FastAPI server for UniBot frontend
Run this instead of app.py when using the custom frontend
"""
import sys
from pathlib import Path
import time
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# Add parent directory to path to import UniBot modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncio
import os
from dotenv import load_dotenv
from agents.unibot import UniBot
from agents.intent_detector import IntentDetector

load_dotenv(override=True)

# Initialize intent detector
intent_detector = IntentDetector()

# Global UniBot instance
unibot: Optional[UniBot] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global unibot
    try:
        # Check for API key before initialization
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            error_msg = (
                "[ERROR] GROQ_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment variables."
            )
            print(error_msg)
            raise ValueError(error_msg)
        
        college_url = os.getenv("COLLEGE_WEBSITE_URL", None)
        print("[INFO] Initializing UniBot...")
        print(f"[INFO] - Groq API Key: {'Set' if groq_api_key else 'Missing'}")
        print(f"[INFO] - College URL: {college_url or 'Not set'}")
        
        unibot = UniBot(college_website_url=college_url)
        await unibot.setup()
        
        print("[OK] UniBot initialized successfully")
        print(f"[INFO] - Using Groq model: openai/gpt-oss-20b (Worker/Evaluator/Intent)")
        print(f"[INFO] - Using embeddings: FastEmbed (BAAI/bge-small-en-v1.5, local, no API calls)")
        
        # Check if knowledge base has data
        try:
            stats = unibot.rag_system.get_stats()
            doc_count = stats.get("total_documents", 0)
            print(f"[INFO] - Knowledge base: {doc_count} documents")
            if doc_count == 0:
                print("[WARNING] Knowledge base is empty. Consider scraping the college website first.")
        except Exception as e:
            print(f"[WARNING] Could not check knowledge base stats: {e}")
        
        if college_url:
            print(f"[INFO] - College website: {college_url}")
        
        print("[OK] API server is ready on http://127.0.0.1:8001")
    except ValueError as e:
        # Re-raise ValueError (API key missing)
        print(f"[ERROR] Configuration error: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Error initializing UniBot: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown
    if unibot:
        try:
            unibot.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")

app = FastAPI(title="UniBot API", lifespan=lifespan)

# Enable CORS for frontend.
# Origins are read from ALLOWED_ORIGINS (comma-separated) so this doesn't
# need a code change per environment - set it once as a platform env var
# pointing at your real deployed frontend URL. Falls back to the local
# Vite dev server ports for local development.
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8081,http://127.0.0.1:8081"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str
    success_criteria: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    history: List[Dict[str, Any]]
    status: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "UniBot API",
        "unibot_ready": unibot is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    global unibot
    status = {
        "status": "healthy" if unibot is not None else "initializing",
        "unibot_ready": unibot is not None
    }
    
    # Add additional info if ready
    if unibot is not None:
        try:
            stats = unibot.rag_system.get_stats()
            status["knowledge_base_documents"] = stats.get("total_documents", 0)
        except:
            pass
    
    return status

def filter_feedback_messages(history):
    """Remove evaluator feedback messages from history"""
    if not history:
        return history
    filtered = []
    for msg in history:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if "Evaluator Feedback" not in str(content):
            filtered.append(msg)
    return filtered

def clean_response(response: str) -> str:
    """
    Remove internal reasoning and chain-of-thought content from LLM responses.
    This function only filters responses that clearly contain internal reasoning patterns.
    Normal responses pass through unchanged.
    """
    if not response or not isinstance(response, str):
        return response
    
    # Patterns that indicate internal reasoning (very specific patterns)
    reasoning_patterns = [
        "with this feedback, i need to re-evaluate",
        "in the turn before the last rejected response",
        "my mistake was in stating",
        "therefore, the correct approach is",
        "let's re-examine the output",
        "let me re-examine the output",
        "the previous rejection was valid",
        "revised plan:",
        "my previous response:",
    ]
    
    # Patterns that indicate actual response content
    response_start_patterns = [
        "i apologize",
        "the knowledge base",
        "based on the",
        "according to the",
        "here are",
        "here is",
        "the following",
        "question:",
    ]
    
    response_lower = response.lower()
    
    # Only filter if we find VERY specific reasoning patterns (not just individual words)
    # This is more conservative to avoid filtering normal responses
    has_reasoning = any(pattern in response_lower for pattern in reasoning_patterns)
    
    # If no reasoning patterns, return original immediately (normal response)
    if not has_reasoning:
        return response
    
    # Response has reasoning - try to find where actual response starts
    lines = response.split('\n')
    
    # Look for response start pattern and include everything from there
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if any(pattern in line_lower for pattern in response_start_patterns):
            # Found response start - include from here
            cleaned = '\n'.join(lines[i:]).strip()
            # Only return if we got something reasonable
            if len(cleaned) >= 20:
                return cleaned
    
    # If no clear response start found, try last paragraph (often contains the answer)
    paragraphs = response.split('\n\n')
    for para in reversed(paragraphs):
        para_lower = para.lower().strip()
        # Use last substantial paragraph that doesn't look like reasoning
        if len(para.strip()) > 50 and not any(pattern in para_lower for pattern in reasoning_patterns):
            if any(pattern in para_lower for pattern in response_start_patterns):
                return para.strip()
    
    # If all else fails, return original (better to show reasoning than lose response entirely)
    return response

def extract_text_content(content):
    """Extract text from content which might be a string, list, or dict"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle list of text blocks (e.g., [{'type': 'text', 'text': '...'}])
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if 'text' in item:
                    text_parts.append(item['text'])
                elif 'content' in item:
                    text_parts.append(extract_text_content(item['content']))
            elif isinstance(item, str):
                text_parts.append(item)
        return '\n'.join(text_parts)
    elif isinstance(content, dict):
        # Handle dict with text field
        if 'text' in content:
            return content['text']
        elif 'content' in content:
            return extract_text_content(content['content'])
        else:
            return str(content)
    else:
        return str(content)

@app.post("/api/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Handle chat messages"""
    global unibot
    request_start = time.perf_counter()
    if not unibot:
        raise HTTPException(status_code=503, detail="UniBot not initialized")
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Validate query is college-domain only
    is_valid, rejection_message = intent_detector.validate_query(request.message)
    if not is_valid:
        total_time = time.perf_counter() - request_start
        print(f"[TIMING] Total /api/chat: {total_time:.2f}s")
        return MessageResponse(
            response=rejection_message,
            history=request.history or [],
            status="rejected"
        )
    
    
    try:
        # Filter existing feedback from history
        history = filter_feedback_messages(request.history or [])
        
        # Process message
        success_criteria = request.success_criteria or "The answer should be clear and accurate"
        updated_history = await unibot.run_superstep(
            request.message,
            success_criteria,
            history,
            thread_id=request.session_id
        )
        
        # Filter feedback from results
        updated_history = filter_feedback_messages(updated_history)
        
        # Get the last assistant message (skip evaluator feedback)
        assistant_messages = [
            msg for msg in updated_history 
            if msg.get("role") == "assistant" and 
            "Evaluator Feedback" not in str(msg.get("content", ""))
        ]
        
        if assistant_messages:
            content = assistant_messages[-1].get("content", "")
            response_text = extract_text_content(content)
        else:
            response_text = "I apologize, but I encountered an issue processing your request."
        
        # Ensure response_text is a string
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Clean response to remove internal reasoning (only if response is not empty)
        if response_text and len(response_text.strip()) > 0:
            response_text = clean_response(response_text)
        
        return MessageResponse(
            response=response_text,
            history=updated_history,
            status="success"
        )
        
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/api/reset")
async def reset():
    """Reset the UniBot instance"""
    global unibot
    
    try:
        if unibot:
            unibot.cleanup()
        
        college_url = os.getenv("COLLEGE_WEBSITE_URL", None)
        unibot = UniBot(college_website_url=college_url)
        await unibot.setup()
        
        return {"status": "success", "message": "UniBot reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting UniBot: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)