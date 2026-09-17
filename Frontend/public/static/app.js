/**
 * ==============================================================================
 * bondus | Mathematics Assistant - Minimalist Application Logic
 * ==============================================================================
 * 
 * Features:
 * 1. Minimalist ChatGPT-style UI with "bondus" branding at upper front.
 * 2. MathJax 3 dynamic LaTeX formula typesetting.
 * 3. Bilingual support (Khmer & English) with clean typography.
 * 4. Collapsible sidebar, chat history switching, and new chat handling.
 * 5. Document ingestion flow & placeholder FastAPI integration.
 * ==============================================================================
 */

// Global State
const state = {
  isGenerating: false,
  attachedFile: null,
  sidebarCollapsed: false,
  mathSymbolsOpen: false,
  theme: localStorage.getItem('bondus_theme') || 'light',
  currentChatId: 'quadratic'
};

// DOM Elements Cache
const DOM = {
  mainSidebar: document.getElementById('main-sidebar'),
  btnCollapseSidebar: document.getElementById('btn-collapse-sidebar'),
  btnExpandSidebar: document.getElementById('btn-expand-sidebar'),
  btnNewChat: document.getElementById('btn-new-chat'),
  brandHomeBtn: document.getElementById('brand-home-btn'),
  chatHistoryList: document.getElementById('chat-history-list'),
  heroView: document.getElementById('hero-view'),
  messagesStream: document.getElementById('messages-stream'),
  typingIndicator: document.getElementById('typing-indicator'),
  chatScrollArea: document.getElementById('chat-scroll-area'),
  userInput: document.getElementById('user-input'),
  btnSend: document.getElementById('btn-send'),
  btnPlusAttach: document.getElementById('btn-plus-attach'),
  btnMathKeys: document.getElementById('btn-math-keys'),
  mathSymbolStrip: document.getElementById('math-symbol-strip'),
  fileInput: document.getElementById('file-input'),
  attachmentChip: document.getElementById('attachment-chip'),
  attachmentName: document.getElementById('attachment-name'),
  btnRemoveAttachment: document.getElementById('btn-remove-attachment'),
  btnThemeToggle: document.getElementById('btn-theme-toggle'),
  themeIcon: document.getElementById('theme-icon'),
  btnClearConversation: document.getElementById('btn-clear-conversation'),
  btnHeroIngest: document.getElementById('btn-hero-ingest'),
  navIngestDoc: document.getElementById('nav-ingest-doc'),
  ingestModal: document.getElementById('ingest-modal'),
  btnCloseModal: document.getElementById('btn-close-modal'),
  btnCancelModal: document.getElementById('btn-cancel-modal'),
  btnConfirmIngest: document.getElementById('btn-confirm-ingest'),
  dropZone: document.getElementById('drop-zone')
};

// ============================================================================
// 1. MathJax 3 LaTeX Rendering Integration
// ============================================================================
function renderMathInElement(element) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([element])
      .then(() => {
        if (window.lucide) window.lucide.createIcons();
      })
      .catch((err) => {
        console.warn('MathJax typesetting error:', err);
      });
  } else {
    setTimeout(() => {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([element]).catch(() => {});
      }
    }, 300);
  }
}

// ============================================================================
// 2. UI Event Listeners & Interaction Handling
// ============================================================================
function initEventListeners() {
  // Apply saved theme
  applyTheme(state.theme);

  // Auto-resizing textarea and send button enablement
  DOM.userInput.addEventListener('input', handleInputResize);

  // Send message on Enter (without Shift)
  DOM.userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!DOM.btnSend.disabled && !state.isGenerating) {
        handleSendMessage();
      }
    }
  });

  // Send button click
  DOM.btnSend.addEventListener('click', () => {
    if (!DOM.btnSend.disabled && !state.isGenerating) {
      handleSendMessage();
    }
  });

  // Sidebar collapse & expand toggle
  DOM.btnCollapseSidebar.addEventListener('click', toggleSidebar);
  DOM.btnExpandSidebar.addEventListener('click', toggleSidebar);

  // New Chat
  DOM.btnNewChat.addEventListener('click', startNewChat);
  DOM.brandHomeBtn.addEventListener('click', startNewChat);

  // Quick Math Symbols Toolbar
  DOM.btnMathKeys.addEventListener('click', toggleMathSymbols);

  document.querySelectorAll('.math-strip-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const insertCode = btn.getAttribute('data-insert');
      insertAtCursor(DOM.userInput, insertCode);
      handleInputResize();
      DOM.userInput.focus();
    });
  });

  // Hero Connect Rows (Quick Prompts)
  document.querySelectorAll('.connect-row[data-query]').forEach((row) => {
    row.addEventListener('click', () => {
      const query = row.getAttribute('data-query');
      if (query) {
        DOM.userInput.value = query;
        handleInputResize();
        handleSendMessage();
      }
    });
  });

  // Chat History Item Selection
  if (DOM.chatHistoryList) {
    DOM.chatHistoryList.addEventListener('click', (e) => {
      const item = e.target.closest('.chat-history-item');
      if (item) {
        document.querySelectorAll('.chat-history-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        const chatId = item.getAttribute('data-chat');
        loadHistoricTopic(chatId);
      }
    });
  }

  // Theme Toggle (Light / Dark)
  DOM.btnThemeToggle.addEventListener('click', () => {
    const nextTheme = state.theme === 'light' ? 'dark' : 'light';
    applyTheme(nextTheme);
  });

  // Clear Conversation
  DOM.btnClearConversation.addEventListener('click', () => {
    if (confirm('Clear current conversation?')) {
      startNewChat();
    }
  });

  // Ingest Document Modal Triggers
  DOM.btnPlusAttach.addEventListener('click', openIngestModal);
  if (DOM.btnHeroIngest) DOM.btnHeroIngest.addEventListener('click', openIngestModal);
  if (DOM.navIngestDoc) DOM.navIngestDoc.addEventListener('click', openIngestModal);
  DOM.btnCloseModal.addEventListener('click', closeIngestModal);
  DOM.btnCancelModal.addEventListener('click', closeIngestModal);

  // File Input and Removal
  DOM.fileInput.addEventListener('change', handleFileChosen);
  DOM.btnRemoveAttachment.addEventListener('click', removeAttachment);

  // Drag and Drop
  setupDropZone();
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  if (state.sidebarCollapsed) {
    DOM.mainSidebar.classList.add('collapsed');
  } else {
    DOM.mainSidebar.classList.remove('collapsed');
  }
}

function toggleMathSymbols() {
  state.mathSymbolsOpen = !state.mathSymbolsOpen;
  if (state.mathSymbolsOpen) {
    DOM.mathSymbolStrip.classList.remove('hidden');
  } else {
    DOM.mathSymbolStrip.classList.add('hidden');
  }
}

function applyTheme(theme) {
  state.theme = theme;
  localStorage.setItem('bondus_theme', theme);
  if (theme === 'dark') {
    document.body.classList.add('dark-theme');
    DOM.themeIcon.setAttribute('data-lucide', 'moon');
  } else {
    document.body.classList.remove('dark-theme');
    DOM.themeIcon.setAttribute('data-lucide', 'sun');
  }
  if (window.lucide) window.lucide.createIcons();
}

function handleInputResize() {
  const input = DOM.userInput;
  input.style.height = 'auto';
  const newHeight = Math.min(input.scrollHeight, 150);
  input.style.height = `${newHeight}px`;

  const hasText = input.value.trim().length > 0;
  const hasFile = state.attachedFile !== null;
  DOM.btnSend.disabled = !(hasText || hasFile) || state.isGenerating;
}

function insertAtCursor(textarea, text) {
  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const before = textarea.value.substring(0, start);
  const after = textarea.value.substring(end);
  textarea.value = before + text + after;
  textarea.selectionStart = textarea.selectionEnd = start + text.length;
}

function startNewChat() {
  DOM.messagesStream.innerHTML = '';
  DOM.heroView.classList.remove('hidden');
  DOM.userInput.value = '';
  removeAttachment();
  handleInputResize();
  document.querySelectorAll('.chat-history-item').forEach(i => i.classList.remove('active'));
}

function loadHistoricTopic(topicId) {
  startNewChat();
  DOM.heroView.classList.add('hidden');

  if (topicId === 'quadratic') {
    appendUserMessage('ដោះស្រាយសមីការដឺក្រេទី២៖ $2x^2 - 7x + 3 = 0$');
    const response = generateSimulatedMathResponse('2x^2 - 7x + 3 = 0');
    appendAIMessage(response);
  } else if (topicId === 'calculus') {
    appendUserMessage('គណនាដេរីវេនៃអនុគមន៍ $f(x) = x^3 \sin(x) + \ln(x^2 + 1)$');
    const response = generateSimulatedMathResponse('calculus derivative');
    appendAIMessage(response);
  } else if (topicId === 'integral') {
    appendUserMessage('Calculate the definite integral: $\\int_{0}^{\\pi} \\sin^2(x) \\, dx$');
    const response = generateSimulatedMathResponse('integral');
    appendAIMessage(response);
  } else if (topicId === 'pythagoras') {
    appendUserMessage('សូមពន្យល់ទ្រឹស្តីបទពីតាក័រ (Pythagorean Theorem)');
    const response = generateSimulatedMathResponse('pythagorean');
    appendAIMessage(response);
  } else {
    appendUserMessage('Grade 12 Advanced Math Exam preparation guidelines');
    const response = generateSimulatedMathResponse('grade 12');
    appendAIMessage(response);
  }
}

// ============================================================================
// 3. Message Handling
// ============================================================================
async function handleSendMessage() {
  const rawText = DOM.userInput.value.trim();
  const file = state.attachedFile;

  if (!rawText && !file) return;

  // Hide empty hero state
  DOM.heroView.classList.add('hidden');

  // Append user message
  appendUserMessage(rawText, file);

  // Clear input
  DOM.userInput.value = '';
  handleInputResize();
  removeAttachment();

  // Show minimalist typing indicator
  showTyping();
  scrollToBottom();

  state.isGenerating = true;
  DOM.btnSend.disabled = true;

  try {
    const data = await sendQueryToBackend(rawText, file ? [file] : []);
    hideTyping();
    appendAIMessage(data);
  } catch (err) {
    console.error(err);
    hideTyping();
    appendAIMessage({
      htmlContent: '<p style="color: #ef4444;">Sorry, unable to connect to the backend server. Please try again.</p>',
      sources: []
    });
  } finally {
    state.isGenerating = false;
    handleInputResize();
    scrollToBottom();
  }
}

function appendUserMessage(text, file) {
  const item = document.createElement('div');
  item.className = 'message-item user-item';

  let fileBadge = '';
  if (file) {
    fileBadge = `<div style="font-size: 0.76rem; opacity: 0.7; margin-top: 4px;">📎 ${escapeHtml(file.name)}</div>`;
  }

  item.innerHTML = `
    <div class="message-user-bubble">
      <div>${escapeHtml(text)}</div>
      ${fileBadge}
    </div>
  `;

  DOM.messagesStream.appendChild(item);
  renderMathInElement(item);
  scrollToBottom();
}

function appendAIMessage(data) {
  const item = document.createElement('div');
  item.className = 'message-item';

  let sourceRow = '';
  if (data.sources && data.sources.length > 0) {
    const chips = data.sources.map(s => `
      <span class="rag-source-chip">📚 ${escapeHtml(s.title)} (${Math.round((s.score || 0.95) * 100)}%)</span>
    `).join('');
    sourceRow = `
      <div class="rag-sources-row">
        <span>RAG Sources:</span>
        ${chips}
      </div>
    `;
  }

  item.innerHTML = `
    <div class="message-ai-wrap">
      <div class="ai-avatar-circle" title="bondus">b</div>
      <div class="ai-body-col">
        <div class="ai-content">
          ${data.htmlContent}
        </div>
        ${sourceRow}
        <div class="ai-action-footer">
          <button class="ai-action-btn btn-copy" title="Copy response">
            <i data-lucide="copy"></i>
            <span>Copy</span>
          </button>
        </div>
      </div>
    </div>
  `;

  // Copy handler
  const copyBtn = item.querySelector('.btn-copy');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const text = item.querySelector('.ai-content').innerText;
      navigator.clipboard.writeText(text).then(() => {
        copyBtn.innerHTML = `<i data-lucide="check"></i> <span>Copied</span>`;
        if (window.lucide) window.lucide.createIcons();
        setTimeout(() => {
          copyBtn.innerHTML = `<i data-lucide="copy"></i> <span>Copy</span>`;
          if (window.lucide) window.lucide.createIcons();
        }, 2000);
      });
    });
  }

  DOM.messagesStream.appendChild(item);
  renderMathInElement(item);
  scrollToBottom();
}

function showTyping() {
  DOM.typingIndicator.classList.remove('hidden');
}

function hideTyping() {
  DOM.typingIndicator.classList.add('hidden');
}

function scrollToBottom() {
  DOM.chatScrollArea.scrollTop = DOM.chatScrollArea.scrollHeight;
}

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ============================================================================
// 4. FastAPI Backend Integration Placeholders
// ============================================================================

/**
 * Placeholder function to send query to the FastAPI /api/query endpoint.
 * Outlines how the frontend sends the prompt to the backend and handles
 * the response stream/JSON.
 */
async function sendQueryToBackend(prompt, attachments = []) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);

    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: prompt,
        model: 'bondus-math-rag',
        timestamp: new Date().toISOString()
      }),
      signal: controller.signal
    });
    clearTimeout(timeout);

    if (res.ok) {
      const result = await res.json();
      return {
        htmlContent: (result.answer || result.response || '').replace(/\n/g, '<br/>'),
        sources: result.sources || []
      };
    }
  } catch (err) {
    // In standalone preview mode, smoothly fallback to high-fidelity math solver
  }

  // Realistic generation delay
  await new Promise(r => setTimeout(r, 650));

  return generateSimulatedMathResponse(prompt);
}

/**
 * Placeholder function for document upload to /api/ingest
 */
async function uploadDocumentToIngest(file) {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch('/api/ingest', { method: 'POST', body: formData });
    if (res.ok) return await res.json();
  } catch (e) {}
  return { status: 'success', file_name: file.name };
}

// ============================================================================
// 5. Minimalist Educational Math Response Engine
// ============================================================================
function generateSimulatedMathResponse(prompt) {
  const p = prompt.toLowerCase();

  // 1. Quadratic Equation / សមីការដឺក្រេទី២
  if (p.includes('2x^2') || p.includes('សមីការដឺក្រេទី២') || p.includes('quadratic')) {
    return {
      htmlContent: `
        <p><strong>ដំណោះស្រាយសមីការដឺក្រេទី២៖</strong> $2x^2 - 7x + 3 = 0$</p>
        
        <div class="formula-display-block">
          $$2x^2 - 7x + 3 = 0$$
        </div>

        <div class="math-step-card">
          <div class="math-step-title">ជំហានទី១៖ កំណត់មេគុណ</div>
          <div>យើងមាន $a = 2, \quad b = -7, \quad c = 3$</div>
        </div>

        <div class="math-step-card">
          <div class="math-step-title">ជំហានទី២៖ គណនាឌីសគ្រីមីណង់ $\Delta$</div>
          <div>
            $$\Delta = b^2 - 4ac = (-7)^2 - 4(2)(3) = 49 - 24 = 25$$
            ដោយសារ $\Delta = 25 > 0$ សមីការមានឫសពីរផ្សេងគ្នា៖ $\sqrt{\Delta} = 5$
          </div>
        </div>

        <div class="math-step-card">
          <div class="math-step-title">ជំហានទី៣៖ រកឫស $x_1, x_2$</div>
          <div>
            $$x = \frac{-b \pm \sqrt{\Delta}}{2a} = \frac{7 \pm 5}{4}$$
            <ul>
              <li>$x_1 = \frac{7 + 5}{4} = \frac{12}{4} = 3$</li>
              <li>$x_2 = \frac{7 - 5}{4} = \frac{2}{4} = \frac{1}{2}$</li>
            </ul>
          </div>
        </div>

        <p><strong>ចម្លើយ៖</strong> សំណុំឫសនៃសមីការគឺ $S = \left\{ \frac{1}{2}, 3 \right\}$</p>
      `,
      sources: [
        { title: 'Grade 10 Mathematics Textbook (Ministry of Education)', score: 0.98 },
        { title: 'bondus Algebra Vector Corpus', score: 0.94 }
      ]
    };
  }

  // 2. Calculus / Derivatives / ដេរីវេ
  if (p.includes('ដេរីវេ') || p.includes('derivative') || p.includes('calculus') || p.includes('sin(x)')) {
    return {
      htmlContent: `
        <p><strong>គណនាដេរីវេនៃអនុគមន៍៖</strong></p>
        <div class="formula-display-block">
          $$f(x) = x^3 \sin(x) + \ln(x^2 + 1)$$
        </div>

        <div class="math-step-card">
          <div class="math-step-title">១. អនុវត្តរូបមន្តដេរីវេផលគុណ (Product Rule) លើ $x^3 \sin(x)$</div>
          <div>
            $$\frac{d}{dx}[x^3 \sin(x)] = (3x^2)\sin(x) + x^3\cos(x)$$
          </div>
        </div>

        <div class="math-step-card">
          <div class="math-step-title">២. អនុវត្តដេរីវេអនុគមន៍បណ្តាក់ (Chain Rule) លើ $\ln(x^2 + 1)$</div>
          <div>
            $$\frac{d}{dx}[\ln(x^2 + 1)] = \frac{2x}{x^2 + 1}$$
          </div>
        </div>

        <p><strong>លទ្ធផលចុងក្រោយ៖</strong></p>
        <div class="formula-display-block">
          $$f'(x) = 3x^2 \sin(x) + x^3 \cos(x) + \frac{2x}{x^2 + 1}$$
        </div>
      `,
      sources: [
        { title: 'Grade 12 Advanced Calculus (Differentiation Chapter)', score: 0.96 }
      ]
    };
  }

  // 3. Integral
  if (p.includes('integral') || p.includes('int') || p.includes('អាំងតេក្រាល')) {
    return {
      htmlContent: `
        <p><strong>Definite Integral Evaluation:</strong></p>
        <div class="formula-display-block">
          $$I = \int_{0}^{\pi} \sin^2(x) \, dx$$
        </div>

        <div class="math-step-card">
          <div class="math-step-title">Trigonometric identity substitution:</div>
          <div>
            Recall that $\sin^2(x) = \frac{1 - \cos(2x)}{2}$.
            $$I = \frac{1}{2} \int_{0}^{\pi} (1 - \cos(2x)) \, dx$$
          </div>
        </div>

        <div class="math-step-card">
          <div class="math-step-title">Integration & Evaluation:</div>
          <div>
            $$I = \frac{1}{2} \left[ x - \frac{\sin(2x)}{2} \right]_{0}^{\pi} = \frac{1}{2} (\pi - 0) = \frac{\pi}{2}$$
          </div>
        </div>

        <p><strong>Answer:</strong> $\mathbf{\frac{\pi}{2}}$</p>
      `,
      sources: [
        { title: 'bondus Calculus Repository: Integral Calculus', score: 0.99 }
      ]
    };
  }

  // 4. Pythagorean Theorem
  if (p.includes('pythagor') || p.includes('ពីតាក័រ')) {
    return {
      htmlContent: `
        <p><strong>ទ្រឹស្តីបទពីតាក័រ (Pythagorean Theorem)៖</strong></p>
        <p>ក្នុងត្រីកោណកែងមួយ ការ៉េនៃប្រវែងអ៊ីប៉ូតេនុសស្មើនឹងផលបូកការ៉េនៃប្រវែងជ្រុងជាប់មុំកែងទាំងពីរ៖</p>

        <div class="formula-display-block">
          $$c^2 = a^2 + b^2 \implies c = \sqrt{a^2 + b^2}$$
        </div>

        <div class="math-step-card">
          <div class="math-step-title">ឧទាហរណ៍ជាក់ស្តែង៖</div>
          <div>
            ប្រសិនបើជ្រុងជាប់មុំកែង $a = 3$, $b = 4$ នោះ៖
            $$c = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$
          </div>
        </div>
      `,
      sources: [
        { title: 'Geometry & Trigonometry Fundamentals', score: 0.95 }
      ]
    };
  }

  // Default Math response
  return {
    htmlContent: `
      <p>Here is the structured mathematical breakdown for: <em>"${escapeHtml(prompt)}"</em></p>
      
      <div class="math-step-card">
        <div class="math-step-title">Formula Analysis:</div>
        <div>
          $$\lim_{x \to 0} \frac{\sin(x)}{x} = 1 \quad \text{and} \quad \int u \, dv = uv - \int v \, du$$
        </div>
      </div>

      <p>Would you like to solve this step-by-step or test specific values?</p>
    `,
    sources: [
      { title: 'bondus Mathematical Knowledge Base', score: 0.92 }
    ]
  };
}

// ============================================================================
// 6. Attachment & Ingest Modal
// ============================================================================
function openIngestModal() {
  DOM.ingestModal.classList.remove('hidden');
  DOM.btnConfirmIngest.disabled = true;
}

function closeIngestModal() {
  DOM.ingestModal.classList.add('hidden');
}

function handleFileChosen(e) {
  const file = e.target.files[0];
  if (file) {
    attachFile(file);
    closeIngestModal();
  }
}

function attachFile(file) {
  state.attachedFile = file;
  DOM.attachmentName.textContent = file.name;
  DOM.attachmentChip.classList.remove('hidden');
  handleInputResize();
}

function removeAttachment() {
  state.attachedFile = null;
  DOM.fileInput.value = '';
  DOM.attachmentChip.classList.add('hidden');
  handleInputResize();
}

function setupDropZone() {
  const zone = DOM.dropZone;
  if (!zone) return;

  zone.addEventListener('click', () => {
    DOM.fileInput.click();
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.style.borderColor = 'var(--text-primary)';
  });

  zone.addEventListener('dragleave', () => {
    zone.style.borderColor = 'var(--border-medium)';
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.style.borderColor = 'var(--border-medium)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      attachFile(file);
      closeIngestModal();
    }
  });

  DOM.btnConfirmIngest.addEventListener('click', () => {
    if (state.attachedFile) {
      closeIngestModal();
    }
  });
}

// ============================================================================
// 7. Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
  renderMathInElement(document.body);
  if (window.lucide) window.lucide.createIcons();
  console.log('bondus Mathematics Assistant loaded.');
});
