"""
NLP Intent Detection System for UniBot
Validates queries are college-domain only and detects intent using LLM
"""
from typing import Dict, Optional, Tuple
import re
import os
from langchain_groq import ChatGroq
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
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        # openai/gpt-oss-20b, not the larger gpt-oss-120b used for the
        # Worker/Evaluator: classification is a simple yes/no + category
        # task that doesn't need a large model, so the smaller/faster
        # model is a better fit here regardless of exact quota numbers
        # (which change - verify current limits at console.groq.com if
        # you hit rate limits specifically on classification calls).
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.1  # Low temperature for consistent classification
        )
        self.classifier_llm = self.llm.with_structured_output(QueryClassification)
    
    def _classify_query(self, query: str) -> QueryClassification:
        """Use LLM to classify if query is college-related and detect intent"""
        # Fast path: a confident keyword match skips the Gemini call
        # entirely. Worker + Evaluator already need several Gemini calls
        # per question, so cutting this one out for the common case
        # meaningfully helps stay under the free-tier rate limit. Only
        # skip the LLM for a POSITIVE match - anything ambiguous (no
        # keyword hit) still goes to the LLM below, since wrongly
        # rejecting a real college question is worse than one extra call.
        quick = self._fallback_classify(query)
        if quick.is_college_related:
            print(f"[TIMING] Intent classification: 0.00s (keyword fast-path, no LLM call)")
            return quick

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
            'campus', 'course', 'program',
            # Expanded: previously missing terms meant genuinely common
            # college questions (e.g. "who is the HOD of cse?") wouldn't
            # even match, and were being sent to the LLM as ambiguous
            # every single time.
            'hod', 'head of department', 'department head',
            'syllabus', 'curriculum', 'placement', 'placements',
            'internship', 'scholarship', 'cutoff', 'counselling',
            'counseling', 'seat', 'merit', 'bmsit', 'library', 'lab',
            'laboratory', 'infrastructure', 'facility', 'timetable',
            'rank', 'ranking', 'nirf', 'semester',
            # BMSIT's actual department names/abbreviations
            'cse', 'ece', 'eee', 'mech', 'mechanical', 'civil', 'aiml',
            'csbs', 'mca', 'mba', 'chemistry', 'electronics',
            'communication', 'computer science'
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
            elif any(kw in query_lower for kw in ['faculty', 'professor', 'teacher', 'hod', 'head of department']):
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