// Message formatting utility

function isTableSeparator(line) {
  const trimmed = line.trim()
  if (!trimmed.includes('|') || !trimmed.includes('-')) return false
  const cells = trimmed.replace(/^\|/, '').replace(/\|$/, '').split('|')
  return cells.length > 0 && cells.every(cell => /^:?-{2,}:?$/.test(cell.trim()))
}

function splitRow(line) {
  let trimmed = line.trim()
  if (trimmed.startsWith('|')) trimmed = trimmed.slice(1)
  if (trimmed.endsWith('|')) trimmed = trimmed.slice(0, -1)
  return trimmed.split('|').map(c => c.trim())
}

export function formatMessage(content) {
  if (!content) return ''

  // Split into lines for processing
  const lines = content.split('\n')
  const processedLines = []
  let inList = false
  let listType = null
  let inTable = false
  let tableHeader = []
  let tableRows = []

  const closeList = () => {
    if (inList) {
      processedLines.push(`</${listType}>`)
      inList = false
      listType = null
    }
  }

  const closeTable = () => {
    if (inTable) {
      const headerHtml = tableHeader.length
        ? `<thead><tr>${tableHeader.map(h => `<th>${formatInline(h)}</th>`).join('')}</tr></thead>`
        : ''
      const rowsHtml = tableRows
        .map(row => `<tr>${row.map(cell => `<td>${formatInline(cell)}</td>`).join('')}</tr>`)
        .join('')
      processedLines.push(
        `<div class="table-wrapper"><table class="message-table">${headerHtml}<tbody>${rowsHtml}</tbody></table></div>`
      )
      inTable = false
      tableHeader = []
      tableRows = []
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmedLine = line.trim()

    // Handle ongoing table
    if (inTable) {
      if (trimmedLine && trimmedLine.includes('|')) {
        if (isTableSeparator(trimmedLine)) {
          continue
        }
        tableRows.push(splitRow(trimmedLine))
        continue
      } else {
        closeTable()
      }
    }

    // Check if new table starts
    if (
      trimmedLine &&
      trimmedLine.includes('|') &&
      i + 1 < lines.length &&
      isTableSeparator(lines[i + 1])
    ) {
      closeList()
      inTable = true
      tableHeader = splitRow(trimmedLine)
      tableRows = []
      i++ // skip separator row
      continue
    }

    // Handle empty lines
    if (!trimmedLine) {
      closeList()
      closeTable()
      // Don't add breaks inside lists or immediately after lists/tables
      if (processedLines.length > 0) {
        const last = processedLines[processedLines.length - 1]
        if (!last.match(/^<\/(ul|ol|p|h[1-6]|table|div)/)) {
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
      closeList()
      closeTable()
      processedLines.push(`<h3>${formatInline(h3Match[1])}</h3>`)
      continue
    }

    if (h2Match) {
      closeList()
      closeTable()
      processedLines.push(`<h2>${formatInline(h2Match[1])}</h2>`)
      continue
    }

    if (h1Match) {
      closeList()
      closeTable()
      processedLines.push(`<h1>${formatInline(h1Match[1])}</h1>`)
      continue
    }

    // Check for numbered list items (1. or 1) format)
    const numberedMatch = trimmedLine.match(/^(\d+)[\.\)]\s+(.+)$/)

    // Check for bullet points (- or *)
    const bulletMatch = trimmedLine.match(/^[\-\*]\s+(.+)$/)

    // Check for bold text followed by colon (like "**Department:** Name")
    const boldColonMatch = trimmedLine.match(/^\*\*(.+?)\*\*:\s*(.+)$/)

    if (numberedMatch) {
      closeTable()
      if (!inList || listType !== 'ol') {
        closeList()
        processedLines.push('<ol>')
        inList = true
        listType = 'ol'
      }
      processedLines.push(`<li>${formatInline(numberedMatch[2])}</li>`)
    } else if (bulletMatch) {
      closeTable()
      if (!inList || listType !== 'ul') {
        closeList()
        processedLines.push('<ul>')
        inList = true
        listType = 'ul'
      }
      processedLines.push(`<li>${formatInline(bulletMatch[1])}</li>`)
    } else if (boldColonMatch) {
      closeTable()
      if (!inList || listType !== 'ul') {
        closeList()
        processedLines.push('<ul>')
        inList = true
        listType = 'ul'
      }
      processedLines.push(
        `<li><strong>${escapeHtml(boldColonMatch[1])}</strong>: ${formatInline(boldColonMatch[2])}</li>`
      )
    } else {
      // Regular paragraph
      closeList()
      closeTable()
      processedLines.push(`<p>${formatInline(trimmedLine)}</p>`)
    }
  }

  // Close any remaining list or table
  closeList()
  closeTable()

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

  // Line breaks <br> or <br/> - preserve model-output breaks
  formatted = formatted.replace(/<br\s*\/?>/gi, () => {
    const placeholder = makePlaceholder()
    placeholders.push('<br>')
    return placeholder
  })

  // Inline code (`code`)
  formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
    const placeholder = makePlaceholder()
    placeholders.push(`<code>${escapeHtml(code)}</code>`)
    return placeholder
  })

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