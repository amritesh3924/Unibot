// API utility functions

export async function checkBackendStatus(apiBaseUrl, setBackendStatus) {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3000)
    
    const response = await fetch(`${apiBaseUrl}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-cache',
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (response.ok) {
      const data = await response.json()
      
      if (data.unibot_ready === true || data.unibot_ready === "true" || data.unibot_ready === 1) {
        const docCount = data.knowledge_base_documents || 0
        setBackendStatus({
          online: true,
          text: `Backend online${docCount > 0 ? ` (${docCount} docs)` : ''}`,
          docCount
        })
      } else {
        setBackendStatus({
          online: false,
          text: 'Backend initializing...',
          docCount: 0
        })
      }
    } else {
      throw new Error('Backend not responding')
    }
  } catch (error) {
    if (error.name === 'AbortError' || error.name === 'TimeoutError') {
      setBackendStatus({
        online: false,
        text: 'Backend timeout',
        docCount: 0
      })
    } else {
      setBackendStatus({
        online: false,
        text: 'Backend offline',
        docCount: 0
      })
    }
  }
}

export async function sendChatMessage(apiBaseUrl, message, history = [], sessionId) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 120000) // 2 minute timeout
  
  try {
    const response = await fetch(`${apiBaseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        success_criteria: "The answer should be clear and accurate",
        history,
        session_id: sessionId
      }),
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      if (response.status === 503) {
        throw new Error('UniBot not initialized')
      } else if (response.status === 400) {
        throw new Error('Invalid request')
      } else {
        throw new Error(`Server error: ${response.status}`)
      }
    }
    
    const data = await response.json()
    return data
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('Request timeout - the query took too long to process')
    }
    throw error
  }
}