import React, { useState, useEffect, useRef } from 'react'
import Header from './components/Header'
import ChatContainer from './components/ChatContainer'
import InputArea from './components/InputArea'
import WelcomeScreen from './components/WelcomeScreen'
import { formatMessage } from './utils/formatter'
import { checkBackendStatus, sendChatMessage } from './utils/api'

import unibotMascot from './assets/unibot-mascot.png';
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001'

// One session id per browser tab/page load, so LangGraph keeps each
// visitor's conversation in its own thread instead of merging everyone
// into the single instance-wide thread.
const SESSION_ID = (
  typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `session-${Date.now()}-${Math.random().toString(36).slice(2)}`
)

function App() {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [backendStatus, setBackendStatus] = useState({
    online: false,
    text: 'Checking...',
    docCount: 0
  })
  const [conversationHistory, setConversationHistory] = useState([])
  const chatContainerRef = useRef(null)

  useEffect(() => {
    checkBackendStatus(API_BASE_URL, setBackendStatus)
    const statusInterval = setInterval(() => {
      checkBackendStatus(API_BASE_URL, setBackendStatus)
    }, 10000)
    return () => clearInterval(statusInterval)
  }, [])

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return
    const userMessage = inputValue.trim()
    setInputValue('')
    const userMsg = { role: 'user', content: userMessage }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    const thinkingMsg = { role: 'bot', content: 'Thinking...', isThinking: true }
    setMessages(prev => [...prev, thinkingMsg])
    try {
      const response = await sendChatMessage(API_BASE_URL, userMessage, conversationHistory, SESSION_ID)
      setMessages(prev => prev.filter(msg => !msg.isThinking))
      if (response.status === 'success') {
        const botMsg = { role: 'bot', content: response.response }
        setMessages(prev => [...prev, botMsg])
        setConversationHistory(response.history || [])
      } else if (response.status === 'rejected') {
        const botMsg = { role: 'bot', content: response.response }
        setMessages(prev => [...prev, botMsg])
      } else {
        throw new Error('Unexpected response status')
      }
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => prev.filter(msg => !msg.isThinking))
      let errorMessage = 'Sorry, I encountered an error. '
      if (error.name === 'AbortError' || error.message.includes('timeout')) {
        errorMessage += 'The request took too long. Please try again with a simpler question.'
      } else if (error.message.includes('not ready') || error.message.includes('initializing')) {
        errorMessage += error.message
      } else if (error.message.includes('503')) {
        errorMessage += 'The backend server is not ready. Please wait a moment and try again.'
      } else {
        errorMessage += 'Please make sure the backend is running and try again.'
      }
      const errorMsg = { role: 'bot', content: errorMessage }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleSuggestionClick = async (question) => {
    setInputValue('')
    const userMsg = { role: 'user', content: question }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    const thinkingMsg = { role: 'bot', content: 'Thinking...', isThinking: true }
    setMessages(prev => [...prev, thinkingMsg])
    try {
      const response = await sendChatMessage(API_BASE_URL, question, conversationHistory, SESSION_ID)
      setMessages(prev => prev.filter(msg => !msg.isThinking))
      if (response.status === 'success') {
        const botMsg = { role: 'bot', content: response.response }
        setMessages(prev => [...prev, botMsg])
        setConversationHistory(response.history || [])
      } else if (response.status === 'rejected') {
        const botMsg = { role: 'bot', content: response.response }
        setMessages(prev => [...prev, botMsg])
      } else {
        throw new Error('Unexpected response status')
      }
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => prev.filter(msg => !msg.isThinking))
      let errorMessage = 'Sorry, I encountered an error. '
      if (error.name === 'AbortError' || error.message.includes('timeout')) {
        errorMessage += 'The request took too long. Please try again with a simpler question.'
      } else if (error.message.includes('not ready') || error.message.includes('initializing')) {
        errorMessage += error.message
      } else if (error.message.includes('503')) {
        errorMessage += 'The backend server is not ready. Please wait a moment and try again.'
      } else {
        errorMessage += 'Please make sure the backend is running and try again.'
      }
      const errorMsg = { role: 'bot', content: errorMessage }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  const showWelcome = messages.length === 0

  return (
    <div className="container">
      <Header backendStatus={backendStatus} />
      <main className="chat-container" ref={chatContainerRef}>
        {showWelcome && (
          <WelcomeScreen onSuggestionClick={handleSuggestionClick} />
        )}
        <div className="messages">
          {messages.map((msg, index) => (
            <Message key={index} message={msg} />
          ))}
        </div>
      </main>
      <InputArea
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyPress={handleKeyPress}
        onSend={handleSendMessage}
        disabled={isLoading}
      />
    </div>
  )
}

function Message({ message }) {
  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {message.role === 'user' ? 'Y' : <img src={unibotMascot} alt="UniBot" />}
      </div>
      <div className={`message-content ${message.isThinking ? 'thinking' : ''}`}>
        {message.isThinking ? (
          <span className="typing-dots" aria-label="UniBot is typing">
            <span className="typing-dot"></span>
            <span className="typing-dot"></span>
            <span className="typing-dot"></span>
          </span>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }} />
        )}
      </div>
    </div>
  )
}

export default App
