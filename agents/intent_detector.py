"""
NLP Intent Detection System for UniBot
Validates queries are college-domain only and detects intent using LLM
"""
from typing import Dict, Optional, Tuple
import re
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time

load_dotenv(override=True)

class QueryClassification(BaseModel):
    """Structured output for query classification"""
    is_college_related: bool = Field(description="True if the query is about college/university topics, False otherwise")
    intent_category: Optional[str] = Field(description="Category: fees, admissions, exams, departments, faculty, hostels, general, or None if out of scope")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class IntentDetector:
    """Detect intent and validate college-domain queries using LLM"""
    
    def __init__(self):
        """Initialize the intent detector with LLM"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        # Use a lightweight model for fast classification
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.1  # Low temperature for consistent classification
        )
        self.classifier_llm = self.llm.with_structured_output(QueryClassification)
    
    def _classify_query(self, query: str) -> QueryClassification:
        """Use LLM to classify if query is college-related and detect intent"""
        system_prompt = """You are a query classifier for a college information assistant.
Your job is to determine if a user's query is related to college/university topics.

College-related topics include:
- Fees, tuition, scholarships, financial aid
- Admissions, applications, eligibility, requirements, cutoffs
- Exams, tests, assessments, semester registration
- Departments, courses, programs, curriculum, syllabus
- Faculty, professors, staff, instructors
- Hostels, accommodation, housing
- Campus facilities, library, labs, events
- General college information

Out-of-scope topics include:
- Weather, news, entertainment
- Personal advice unrelated to college
- General knowledge questions not about the college
- Questions about the college that are not related to the college

Respond with:
- is_college_related: true if the query is about college topics, false otherwise
- intent_category: one of: fees, admissions, exams, departments, faculty, hostels, general, or None if out of scope
- confidence: your confidence in this classification (0.0 to 1.0)"""

        user_prompt = f"Classify this query: {query}"

        try:
            start_time = time.perf_counter()

            result = self.classifier_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            elapsed = time.perf_counter() - start_time
            print(f"[TIMING] Intent classification: {elapsed:.2f}s")

            return result
        except Exception as e:
            # Fallback to basic keyword-based classification if LLM fails
            print(f"Warning: LLM classification failed: {e}, using fallback")
            return self._fallback_classify(query)
    
    def _fallback_classify(self, query: str) -> QueryClassification:
        """Fallback keyword-based classification if LLM fails"""
        query_lower = query.lower()
        
        # Basic college keywords
        college_keywords = [
            'fee', 'fees', 'tuition', 'admission', 'exam', 'department', 
            'faculty', 'professor', 'hostel', 'college', 'university', 
            'campus', 'course', 'program'
        ]
        
        has_college_keyword = any(re.search(r'\b' + re.escape(kw) + r'\b', query_lower) for kw in college_keywords)
        
        # Very basic intent detection
        intent = None
        if has_college_keyword:
            if any(kw in query_lower for kw in ['fee', 'tuition', 'scholarship']):
                intent = 'fees'
            elif any(kw in query_lower for kw in ['admission', 'apply', 'eligibility']):
                intent = 'general'
            elif any(kw in query_lower for kw in ['exam', 'test', 'assessment']):
                intent = 'exams'
            elif any(kw in query_lower for kw in ['department', 'course', 'program']):
                intent = 'departments'
            elif any(kw in query_lower for kw in ['faculty', 'professor', 'teacher']):
                intent = 'faculty'
            elif any(kw in query_lower for kw in ['hostel', 'accommodation']):
                intent = 'hostels'
            else:
                intent = 'general'
        
        return QueryClassification(
            is_college_related=has_college_keyword,
            intent_category=intent,
            confidence=0.7 if has_college_keyword else 0.5
        )
    
    def detect_intent(self, query: str) -> Tuple[Optional[str], float]:
        """
        Detect intent from query using LLM
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (intent_category, confidence_score)
            Returns (None, 0.0) if out of scope
        """
        classification = self._classify_query(query)
        
        if not classification.is_college_related:
            return (None, 0.0)
        
        return (classification.intent_category, classification.confidence)
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if query is college-domain only using LLM
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (is_valid, rejection_message)
            If valid, returns (True, None)
            If invalid, returns (False, polite_rejection_message)
        """
        classification = self._classify_query(query)
        
        if not classification.is_college_related:
            return (False, 
                "I'm sorry, but I'm specifically designed to answer questions about the college, "
                "including fees, admissions, exams, departments, faculty, hostels, and other college-related topics. "
                "Could you please ask me something about the college instead?")
        
        # Query is valid
        return (True, None)
    
    def get_intent_info(self, query: str) -> Dict[str, any]:
        """
        Get detailed intent information using LLM
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with intent, confidence, and validation info
        """
        classification = self._classify_query(query)
        is_valid = classification.is_college_related
        rejection_msg = None if is_valid else (
            "I'm sorry, but I'm specifically designed to answer questions about the college, "
            "including fees, admissions, exams, departments, faculty, hostels, and other college-related topics. "
            "Could you please ask me something about the college instead?"
        )
        
        return {
            'intent': classification.intent_category,
            'confidence': classification.confidence,
            'is_valid': is_valid,
            'rejection_message': rejection_msg
        }
