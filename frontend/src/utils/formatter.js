// Message formatting utility

export function formatMessage(content) {
  if (!content) return ''

  // Split into lines for processing
  const lines = content.split('\n')
  const processedLines = []
  let inList = false
  let listType = null

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmedLine = line.trim()
    
    // Handle empty lines
    if (!trimmedLine) {
      if (inList) {
        processedLines.push(`</${listType}>`)
        inList = false
        listType = null
      }
      // Don't add breaks inside lists or immediately after lists
      if (processedLines.length > 0) {
        const last = processedLines[processedLines.length - 1]
        if (!last.match(/^<\/(ul|ol|p|h[1-6])/)) {
          processedLines.push('<br>')
        }
      }
      continue
    }

    // Check for headers first
    const h3Match = trimmedLine.match(/^###\s+(.+)$/)
    const h2Match = trimmedLine.match(/^##\s+(.+)$/)
    const h1Match = trimmedLine.match(/^#\s+(.+)$/)
    
    if (h3Match) {
      if (inList) {
        processedLines.push(`</${listType}>`)
        inList = false
        listType = null
      }
      processedLines.push(`<h3>${formatInline(h3Match[1])}</h3>`)
      continue
    }
    
    if (h2Match) {
      if (inList) {
        processedLines.push(`</${listType}>`)
        inList = false
        listType = null
      }
      processedLines.push(`<h2>${formatInline(h2Match[1])}</h2>`)
      continue
    }
    
    if (h1Match) {
      if (inList) {
        processedLines.push(`</${listType}>`)
        inList = false
        listType = null
      }
      processedLines.push(`<h1>${formatInline(h1Match[1])}</h1>`)
      continue
    }

    // Check for numbered list items (1. or 1) format)
    const numberedMatch = trimmedLine.match(/^(\d+)[\.\)]\s+(.+)$/)
    
    // Check for bullet points (- or *)
    const bulletMatch = trimmedLine.match(/^[\-\*]\s+(.+)$/)
    
    // Check for bold text followed by colon (like "**Department:** Name")
    // This pattern indicates a structured list item
    const boldColonMatch = trimmedLine.match(/^\*\*(.+?)\*\*:\s*(.+)$/)
    
    if (numberedMatch) {
      // Numbered list item
      if (!inList || listType !== 'ol') {
        if (inList) {
          processedLines.push(`</${listType}>`)
        }
        processedLines.push('<ol>')
        inList = true
        listType = 'ol'
      }
      processedLines.push(`<li>${formatInline(numberedMatch[2])}</li>`)
    } else if (bulletMatch) {
      // Bullet list item
      if (!inList || listType !== 'ul') {
        if (inList) {
          processedLines.push(`</${listType}>`)
        }
        processedLines.push('<ul>')
        inList = true
        listType = 'ul'
      }
      processedLines.push(`<li>${formatInline(bulletMatch[1])}</li>`)
    } else if (boldColonMatch) {
      // Bold text with colon - treat as list item for better formatting
      if (!inList || listType !== 'ul') {
        if (inList) {
          processedLines.push(`</${listType}>`)
        }
        processedLines.push('<ul>')
        inList = true
        listType = 'ul'
      }
      // Format the bold part and the text after colon
      processedLines.push(`<li><strong>${escapeHtml(boldColonMatch[1])}</strong>: ${formatInline(boldColonMatch[2])}</li>`)
    } else {
      // Regular paragraph
      if (inList) {
        processedLines.push(`</${listType}>`)
        inList = false
        listType = null
      }
      processedLines.push(`<p>${formatInline(trimmedLine)}</p>`)
    }
  }
  
  // Close any open list
  if (inList) {
    processedLines.push(`</${listType}>`)
  }

  let formatted = processedLines.join('')
  
  // Clean up multiple consecutive breaks
  formatted = formatted.replace(/(<br>\s*){3,}/g, '<br><br>')
  
  // Clean up breaks before closing tags
  formatted = formatted.replace(/<br>\s*(<\/[^>]+>)/g, '$1')
  
  // Clean up breaks after opening tags (except <br> itself)
  formatted = formatted.replace(/(<[^/>]+>)\s*<br>/g, '$1')
  
  // Clean up empty paragraphs
  formatted = formatted.replace(/<p>\s*<\/p>/g, '')
  
  return formatted
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
  if (!text) return ''
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

// Format inline elements (bold, italic, links, etc.)
// IMPORTANT: Apply markdown formatting BEFORE escaping HTML
//
// PLACEHOLDER SAFETY:
// Placeholders MUST NOT contain characters that any later regex pass in this
// function could match (specifically '_' and '*'). The old token
// `__PLACEHOLDER_N__` used double underscores, which is indistinguishable
// from __bold__ markdown syntax - so when two bold/link segments landed on
// the same line, the __text__ bold pass would re-match across an
// already-inserted placeholder from an earlier pass, corrupting it and
// leaking raw "PLACEHOLDER_N" text into the rendered output (e.g. two
// **bold** fee amounts on one line). Control characters can't appear in
// normal text and can't match \S/\w-based markdown patterns, so they're
// used here instead.
function formatInline(text) {
  if (!text) return ''
  
  let formatted = text
  
  // Use a placeholder approach to protect HTML we create
  const placeholders = []
  let placeholderIndex = 0
  const makePlaceholder = () => `\u0001PH${placeholderIndex++}\u0002`
  
  // Links [text](url) - handle first
  formatted = formatted.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, linkText, url) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(linkText)}</a>`)
    return placeholder
  })
  
  // Bold (**text** or __text__)
  formatted = formatted.replace(/\*\*([^*]+?)\*\*/g, (match, content) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<strong>${escapeHtml(content)}</strong>`)
    return placeholder
  })
  formatted = formatted.replace(/__([^_]+?)__/g, (match, content) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<strong>${escapeHtml(content)}</strong>`)
    return placeholder
  })
  
  // Italic (*text* or _text_) - single asterisks/underscores (not part of bold)
  formatted = formatted.replace(/(?<!\*)\*([^*\s][^*]*?[^*\s])\*(?!\*)/g, (match, content) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<em>${escapeHtml(content)}</em>`)
    return placeholder
  })
  formatted = formatted.replace(/(?<!_)_([^_\s][^_]*?[^_\s])_(?!_)/g, (match, content) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<em>${escapeHtml(content)}</em>`)
    return placeholder
  })
  
  // Dates (format: YYYY-MM-DD or DD/MM/YYYY)
  formatted = formatted.replace(/\b(\d{4}-\d{2}-\d{2})\b/g, (match, date) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<time>${date}</time>`)
    return placeholder
  })
  formatted = formatted.replace(/\b(\d{1,2}\/\d{1,2}\/\d{4})\b/g, (match, date) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<time>${date}</time>`)
    return placeholder
  })
  
  // Escape any remaining text (that wasn't part of markdown)
  formatted = escapeHtml(formatted)
  
  // Restore placeholders (which contain our HTML).
  // Use split/join (replaceAll) rather than a single replace, in case a
  // placeholder somehow appears more than once - a single first-match
  // replace would silently leave later occurrences unresolved.
  placeholders.forEach((html, index) => {
    const token = `\u0001PH${index}\u0002`
    formatted = formatted.split(token).join(html)
  })
  
  return formatted
}