import React from 'react'

const SUGGESTED_QUESTIONS = [
  'What is the fee structure for B.Tech?',
  'When is the last date for exam registration?',
  'Tell me about placement statistics',
  'What are the hostel facilities?'
]

function WelcomeScreen({ onSuggestionClick }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-icon">U</div>
      <h2>Welcome to UniBot!</h2>
      <p>Ask me anything about BMSIT - fees, admissions, academics, events, and more.</p>
      
      <div className="suggested-questions">
        {SUGGESTED_QUESTIONS.map((question, index) => (
          <button
            key={index}
            className="suggestion-btn"
            onClick={() => onSuggestionClick(question)}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}

export default WelcomeScreen

