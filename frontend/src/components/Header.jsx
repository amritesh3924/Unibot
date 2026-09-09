import React from 'react'

function Header({ backendStatus }) {
  const dotStyle = {
    background: backendStatus.online ? 'var(--good)' : 'var(--bad)'
  }

  return (
    <header className="header">
      <div className="header-left">
        <div className="logo">
          <div className="logo-icon">U</div>
        </div>
        <div className="header-text">
          <h1>UniBot</h1>
          <p>BMSIT College AI Assistant</p>
        </div>
      </div>
      <div className="header-right">
        <div className="status-indicator">
          <span className="status-dot" style={dotStyle}></span>
          <span>{backendStatus.text}</span>
        </div>
      </div>
    </header>
  )
}

export default Header

