/* === app.js — RAG Enterprise Assistant === */

// ──────────────────────────────────────────────
// Theme Toggle
// ──────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('theme');
  if (saved === 'light') {
    document.body.classList.add('theme-light');
  }
  updateThemeIcons();

  document.getElementById('theme-toggle').addEventListener('click', () => {
    const isLight = document.body.classList.toggle('theme-light');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    updateThemeIcons();
  });
}

function updateThemeIcons() {
  const isLight = document.body.classList.contains('theme-light');
  document.getElementById('theme-icon-dark').style.display = isLight ? 'none' : '';
  document.getElementById('theme-icon-light').style.display = isLight ? '' : 'none';
}

// ──────────────────────────────────────────────
// SSEClient — Proper SSE parser
// ──────────────────────────────────────────────
class SSEClient {
  constructor(url, {onStep, onSource, onToken, onDone, onError}) {
    this._url = url;
    this._callbacks = {onStep, onSource, onToken, onDone, onError};
    this._abort = null;
  }

  async connect(body) {
    this._abort = new AbortController();
    let timer = null;
    try {
      const resp = await fetch(this._url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
        signal: this._abort.signal,
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      let lastActivity = Date.now();
      timer = setInterval(() => {
        if (Date.now() - lastActivity > 45000) { this._abort.abort(); }
      }, 5000);

      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        lastActivity = Date.now();
        buffer += decoder.decode(value, {stream: true});

        const normalized = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        const parts = normalized.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (!part.trim()) continue;
          const event = this._parseEvent(part);
          if (!event) continue;
          this._dispatch(event);
        }
      }
      clearInterval(timer);
      this._callbacks.onDone?.();
    } catch (err) {
      if (timer) clearInterval(timer);
      if (err.name !== 'AbortError') {
        this._callbacks.onError?.('Network error: ' + err.message);
      } else {
        this._callbacks.onError?.('Request timed out.');
      }
    }
  }

  _parseEvent(text) {
    const event = {event: 'message', data: ''};
    const lines = text.split('\n');
    let dataLines = [];
    for (const line of lines) {
      if (line.startsWith('event:')) { event.event = line.slice(6).trim(); }
      else if (line.startsWith('data:')) { dataLines.push(line.slice(6)); }
    }
    event.data = dataLines.join('\n');
    return event;
  }

  _dispatch(event) {
    switch (event.event) {
      case 'step': this._callbacks.onStep?.(event.data); break;
      case 'source': this._callbacks.onSource?.(event.data); break;
      case 'token': this._callbacks.onToken?.(event.data); break;
      case 'error': this._callbacks.onError?.(event.data); break;
      case 'done': this._callbacks.onDone?.(); break;
    }
  }

  cancel() { this._abort?.abort(); }
}

// ──────────────────────────────────────────────
// ThinkingChain — Gemini-style reasoning display
// ──────────────────────────────────────────────
class ThinkingChain {
  constructor(container) {
    this._el = document.createElement('div');
    this._el.className = 'thinking-chain expanded';
    this._steps = [];
    this._startTime = Date.now();
    this._sourceCount = 0;
    this._chunkCount = 0;

    const header = document.createElement('div');
    header.className = 'thinking-chain-header';
    header.innerHTML = `
      <svg class="chain-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
      <span class="chain-title">Reasoning</span>
      <span class="chain-spinner"><span class="chain-dot"></span><span class="chain-dot"></span><span class="chain-dot"></span></span>`;
    header.addEventListener('click', () => this._toggle());
    this._el.appendChild(header);

    this._body = document.createElement('div');
    this._body.className = 'thinking-chain-body';
    this._el.appendChild(this._body);

    this._titleEl = header.querySelector('.chain-title');
    this._spinnerEl = header.querySelector('.chain-spinner');
    container.appendChild(this._el);
  }

  addStep(id, text) {
    for (const s of this._steps) { if (s.status === 'active') s.status = 'done'; }
    const existingIdx = this._steps.findIndex(s => s.id === id);
    if (existingIdx >= 0) {
      this._steps[existingIdx] = {id, text, status: 'active'};
    } else {
      this._steps.push({id, text, status: 'active'});
    }
    if (id === 'rewrite') {
      for (const f of [{id:'retrieve',text:'Searching documents'},{id:'generate',text:'Generating answer'},{id:'check',text:'Verifying accuracy'}]) {
        if (!this._steps.some(s => s.id === f.id)) this._steps.push({...f, status:'pending'});
      }
    }
    this._render();
  }

  updateStep(id, text) {
    const step = this._steps.find(s => s.id === id);
    if (step) { step.text = text; step.status = 'done'; }
    this._render();
  }

  setSources(sourceCount, chunkCount) {
    this._sourceCount = sourceCount;
    this._chunkCount = chunkCount;
    this.updateStep('retrieve', `Searched ${sourceCount} document${sourceCount!==1?'s':''}, found ${chunkCount} chunk${chunkCount!==1?'s':''}`);
  }

  finish() {
    for (const s of this._steps) { if (s.status==='pending'||s.status==='active') s.status='done'; }
    const elapsed = ((Date.now()-this._startTime)/1000).toFixed(1);
    this._titleEl.textContent = `Searched ${this._sourceCount} doc${this._sourceCount!==1?'s':''} · ${this._chunkCount} source${this._chunkCount!==1?'s':''} · ${elapsed}s`;
    this._spinnerEl.style.display = 'none';
    this._render();
    setTimeout(() => { this._el.classList.remove('expanded'); this._el.classList.add('collapsed'); }, 800);
  }

  _toggle() {
    if (this._el.classList.contains('collapsed')) { this._el.classList.remove('collapsed'); this._el.classList.add('expanded'); }
    else if (this._el.classList.contains('expanded')) { this._el.classList.remove('expanded'); this._el.classList.add('collapsed'); }
  }

  _render() {
    this._body.innerHTML = this._steps.map(s => {
      let icon = s.status==='done'?'✓':s.status==='active'?'•':'·';
      return `<div class="thinking-chain-step ${s.status}"><span class="step-icon">${icon}</span><span class="step-text">${esc(s.text)}</span></div>`;
    }).join('');
  }
}

// ──────────────────────────────────────────────
// View Router
// ──────────────────────────────────────────────
class ViewRouter {
  constructor() {
    this._panels = document.querySelectorAll('.view-panel');
    this._navItems = document.querySelectorAll('[data-view]');
    const params = new URLSearchParams(window.location.search);
    this.switch(params.get('view') || 'chat', false);

    this._navItems.forEach(item => {
      item.addEventListener('click', () => {
        const view = item.dataset.view;
        this.switch(view, true);
        const url = new URL(window.location);
        url.searchParams.set('view', view);
        window.history.replaceState({}, '', url);
      });
    });
    window.addEventListener('popstate', () => {
      const p = new URLSearchParams(window.location.search);
      this.switch(p.get('view') || 'chat', false);
    });
  }

  switch(viewName, animate=true) {
    const targetPanel = document.getElementById('view-' + viewName);
    if (!targetPanel) return;
    this._panels.forEach(p => {
      if (!animate) { p.style.transition='none'; p.classList.toggle('active', p===targetPanel); p.offsetHeight; p.style.transition=''; }
      else { p.classList.toggle('active', p===targetPanel); }
    });
    this._navItems.forEach(item => item.classList.toggle('active', item.dataset.view === viewName));
    if (viewName === 'chat') {
      setTimeout(() => {
        const inp = hasMessages() ? document.getElementById('question-input') : document.getElementById('question-input-centered');
        inp?.focus();
      }, 350);
    }
    if (viewName === 'documents') loadDocs();
  }
}

// ──────────────────────────────────────────────
// Chat State
// ──────────────────────────────────────────────
let _currentChain = null;
let _currentAssistant = null;
let _messageCount = 0;

function hasMessages() { return _messageCount > 0; }

function transitionToMessages() {
  if (_messageCount > 0) return;
  _messageCount = 1;
  document.getElementById('welcome-center').classList.add('hidden');
  const bar = document.getElementById('chat-form');
  bar.style.display = 'flex';
  const centeredInput = document.getElementById('question-input-centered');
  document.getElementById('question-input').value = centeredInput.value;
  setTimeout(() => document.getElementById('question-input').focus(), 400);
}

function sendMessage(question) {
  if (!question.trim()) return;

  transitionToMessages();

  const sendBtn = document.getElementById('send-btn');
  const input = document.getElementById('question-input');
  sendBtn.disabled = true;
  input.disabled = true;

  addUserMessage(question);
  const msg = document.createElement('div');
  msg.className = 'message assistant';
  const messagesEl = document.getElementById('chat-messages');

  // Inner container for centered content within full-width row
  const inner = document.createElement('div');
  inner.className = 'msg-inner';

  _currentChain = new ThinkingChain(inner);
  const body = document.createElement('div');
  body.className = 'msg-body';
  inner.appendChild(body);
  const sources = document.createElement('div');
  sources.className = 'msg-sources';
  inner.appendChild(sources);
  msg.appendChild(inner);
  messagesEl.appendChild(msg);
  _currentAssistant = {msg, body, sources};
  scrollToBottom();

  function cleanup() {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }

  const sse = new SSEClient('/api/query/stream', {
    onStep(data) {
      const m = {
        'Rewriting query...':{id:'rewrite',text:'Reformulating query'},
        'Searching documents...':{id:'retrieve',text:'Searching documents'},
        'Generating answer...':{id:'generate',text:'Generating answer'},
        'Verifying accuracy...':{id:'check',text:'Verifying accuracy'},
      }[data];
      if (m) _currentChain.addStep(m.id, m.text);
    },
    onSource(text) {
      try {
        if (!sources.querySelector(`[data-text="${escAttr(text).substring(0,20)}"]`)) {
          const chip = document.createElement('span');
          chip.className = 'source-chip';
          chip.textContent = text.length > 80 ? text.substring(0,80)+'...' : text;
          chip.setAttribute('data-text', text.substring(0,20));
          sources.appendChild(chip);
        }
        _currentChain._sourceCount = Math.max(_currentChain._sourceCount, 1);
        _currentChain._chunkCount = sources.querySelectorAll('.source-chip').length;
        _currentChain.setSources(_currentChain._sourceCount, _currentChain._chunkCount);
      } catch(e) {}
    },
    onToken(char) { body.textContent += char; scrollToBottom(); },
    onDone() {
      if (_currentChain) { _currentChain.finish(); _currentChain = null; }
      _currentAssistant = null;
      cleanup();
    },
    onError(msg) {
      body.textContent = 'Sorry, something went wrong. ' + msg;
      if (_currentChain) { _currentChain.finish(); _currentChain = null; }
      _currentAssistant = null;
      cleanup();
    },
  });

  sse.connect({question, namespace:'default'}).catch(err => {
    body.textContent = 'Connection failed: ' + err.message;
    if (_currentChain) { _currentChain.finish(); _currentChain = null; }
    _currentAssistant = null;
    cleanup();
  });
}

function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message user';
  const inner = document.createElement('div');
  inner.className = 'msg-inner';
  inner.textContent = text;
  row.appendChild(inner);
  document.getElementById('chat-messages').appendChild(row);
  scrollToBottom();
}

function scrollToBottom() {
  const el = document.getElementById('chat-messages');
  requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
}

// ──────────────────────────────────────────────
// Documents
// ──────────────────────────────────────────────
async function loadDocs() {
  try {
    const resp = await fetch('/api/documents');
    const docs = await resp.json();
    const tbody = document.getElementById('docs-tbody');
    tbody.innerHTML = '';
    if (docs.length === 0) {
      tbody.innerHTML = '<tr class="empty-row"><td colspan="4">No documents indexed yet.</td></tr>';
      return;
    }
    docs.forEach(d => {
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td><span class="doc-name">${esc(d.filename||d.source_id)}</span></td>` +
        `<td><span class="type-tag">${esc(d.file_type||'?')}</span></td>` +
        `<td>${d.chunk_count}</td>` +
        `<td><button class="btn-sm btn-danger" data-delete="${escAttr(d.source_id)}">Delete</button></td>`;
      tbody.appendChild(tr);
    });
    tbody.querySelectorAll('[data-delete]').forEach(btn => {
      btn.addEventListener('click', async () => {
        if (!confirm('Delete this document and all its chunks?')) return;
        await fetch('/api/documents/' + btn.dataset.delete, {method:'DELETE'});
        await loadDocs();
      });
    });
  } catch(e) {
    document.getElementById('docs-tbody').innerHTML = '<tr class="empty-row"><td colspan="4">Failed to load documents.</td></tr>';
  }
}

function setupDocuments() {
  const fi = document.getElementById('file-input');
  const uz = document.getElementById('upload-zone');
  const us = document.getElementById('upload-status');
  uz.addEventListener('click', () => fi.click());
  uz.addEventListener('dragover', e => { e.preventDefault(); uz.classList.add('drag-over'); });
  uz.addEventListener('dragleave', () => uz.classList.remove('drag-over'));
  uz.addEventListener('drop', e => { e.preventDefault(); uz.classList.remove('drag-over'); doUpload(e.dataTransfer.files); });
  fi.addEventListener('change', () => doUpload(fi.files));

  async function doUpload(files) {
    for (const file of files) {
      us.innerHTML = `<div class="upload-progress"><div class="spinner"></div>Uploading ${esc(file.name)}...</div>`;
      const form = new FormData(); form.append('file', file);
      try {
        const resp = await fetch('/api/documents', {method:'POST', body:form});
        const data = await resp.json();
        us.innerHTML = data.status==='duplicate'
          ? `<div class="status-msg status-warn">${esc(file.name)} already indexed</div>`
          : `<div class="status-msg status-ok">${esc(file.name)} — ${data.chunk_count} chunks indexed</div>`;
      } catch(err) {
        us.innerHTML = `<div class="status-msg status-err">Upload failed: ${esc(err.message)}</div>`;
      }
      fi.value = '';
      await loadDocs();
      setTimeout(() => { if (us.firstChild) us.firstChild.style.opacity='0'; }, 4000);
    }
  }
  loadDocs();
}

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────
function esc(s) { const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function escAttr(s) { return s.replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  new ViewRouter();
  setupDocuments();

  // Auto-grow textarea
  function autoGrow(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }

  // Centered input (welcome state)
  const centeredForm = document.getElementById('chat-form-centered');
  const centeredInput = document.getElementById('question-input-centered');
  centeredInput.addEventListener('input', () => autoGrow(centeredInput));
  centeredInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); centeredForm.dispatchEvent(new Event('submit')); }
  });
  centeredForm.addEventListener('submit', e => {
    e.preventDefault();
    const q = centeredInput.value.trim();
    if (!q) return;
    centeredInput.value = '';
    autoGrow(centeredInput);
    sendMessage(q);
  });

  // Bottom input bar
  const bottomForm = document.getElementById('chat-form');
  const bottomInput = document.getElementById('question-input');
  bottomInput.addEventListener('input', () => autoGrow(bottomInput));
  bottomInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); bottomForm.dispatchEvent(new Event('submit')); }
  });
  bottomForm.addEventListener('submit', e => {
    e.preventDefault();
    const q = bottomInput.value.trim();
    if (!q) return;
    bottomInput.value = '';
    autoGrow(bottomInput);
    sendMessage(q);
  });
});
