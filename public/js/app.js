// ─────────────────────────────────────────────
//  AXIOM — app.js
//  Main chat page logic
// ─────────────────────────────────────────────

// ── STATE ──────────────────────────────────────
let currentUser     = null;
let currentUsername = null;
let isGuest         = false;
let convos          = {};   // id → { title, subject, messages[] }
let currentId       = null;
let currentSubject  = 'General';
let busy            = false;
let lastQuestion    = '';

// ── DOM ────────────────────────────────────────
const $ = id => document.getElementById(id);
const messagesEl  = $('messages');
const inputArea   = $('inputArea');
const sendBtn     = $('sendBtn');
const topbarTitle = $('topbarTitle');
const historyList = $('historyList');
const teachToast  = $('teachToast');
const teachInput  = $('teachInput');
const modal       = $('modal');
const modalTitle  = $('modalTitle');
const modalBody   = $('modalBody');
const sidebar     = $('sidebar');
const sidebarOverlay = $('sidebarOverlay');

// ── BOOT ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  setupListeners();

  // Check if actually logged in first
  currentUser = await sbGetCurrentUser();

  if (currentUser) {
    // Logged in — clear any leftover guest flag
    sessionStorage.removeItem('axiom_guest');
    isGuest = false;
    currentUsername = await sbGetUsername(currentUser.id);
  } else {
    // Not logged in — check guest flag
    isGuest = sessionStorage.getItem('axiom_guest') === '1';
    if (!isGuest) { window.location.href = 'login.html'; return; }
  }

  updateProfileBtn();
  await bootChat();
});

async function bootChat() {
  if (isGuest) { newChat(); return; } // guest — skip everything

  currentUser = await sbGetCurrentUser();
  if (!currentUser) { window.location.href = 'login.html'; return; }

  await sbEnsureProfile(currentUser);
  currentUsername = await sbGetUsername(currentUser.id);
  updateProfileBtn(); // refresh username in sidebar after OAuth

  const chats = await sbLoadChats(currentUser.id);
  if (chats.length > 0) {
    chats.forEach(c => convos[c.id] = { title: c.title, subject: c.subject, messages: [] });
    renderHistory();
    await switchChat(chats[0].id);
  } else {
    newChat();
  }
}

// ── PROFILE BUTTON ─────────────────────────────
function updateProfileBtn() {
  const btn = $('btnProfile');
  const label = isGuest ? 'Guest' : (currentUsername || 'Profile');
  btn.innerHTML = `
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <circle cx="7.5" cy="5.5" r="2.5" stroke="currentColor" stroke-width="1.3"/>
      <path d="M2 13c0-2.5 2.5-4 5.5-4s5.5 1.5 5.5 4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
    </svg>
    ${esc(label)}`;
}

// ── LISTENERS ──────────────────────────────────
function setupListeners() {
  $('newChatBtn').addEventListener('click', newChat);
  sendBtn.addEventListener('click', sendMessage);
  inputArea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  inputArea.addEventListener('input', () => autoResize(inputArea));

  document.querySelectorAll('.pill').forEach(p => p.addEventListener('click', () => setPill(p)));

  $('btnProfile').addEventListener('click',  () => openModal('profile'));
  $('btnHistory').addEventListener('click',  () => openModal('history'));
  $('btnSettings').addEventListener('click', () => openModal('settings'));

  $('modalClose').addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  $('teachClose').addEventListener('click', hideTeachToast);
  $('teachSubmit').addEventListener('click', submitTeachAxiom);

  // Mobile sidebar toggle
  $('menuBtn').addEventListener('click', openSidebar);
  sidebarOverlay.addEventListener('click', closeSidebar);
}

function openSidebar()  { sidebar.classList.add('open'); sidebarOverlay.classList.add('show'); }
function closeSidebar() { sidebar.classList.remove('open'); sidebarOverlay.classList.remove('show'); }

// ── HELPERS ────────────────────────────────────
function uid() { return Math.random().toString(36).slice(2, 9); }
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}
function useSuggestion(text) {
  inputArea.value = text;
  autoResize(inputArea);
  inputArea.focus();
}
function setPill(el) {
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  currentSubject = el.dataset.subject;
  if (currentId && currentUser) sbSaveChat(currentId, convos[currentId].title, currentSubject, currentUser.id);
}

// ── NEW CHAT ───────────────────────────────────
function newChat() {
  closeSidebar();
  const id = uid();
  convos[id] = { title: 'New Chat', subject: 'General', messages: [] };
  currentId = id;
  currentSubject = 'General';
  document.querySelectorAll('.pill').forEach((p, i) => p.classList.toggle('active', i === 0));
  topbarTitle.textContent = 'New Chat';
  showEmptyState();
  renderHistory();
  inputArea.value = '';
  inputArea.style.height = 'auto';
  inputArea.focus();
}

// ── HISTORY ────────────────────────────────────
function renderHistory() {
  historyList.innerHTML = '';
  Object.keys(convos).reverse().forEach(id => {
    const c = convos[id];
    const d = document.createElement('div');
    d.className = 'chat-item' + (id === currentId ? ' active' : '');
    d.innerHTML = `<div class="ci-dot"></div><div class="ci-text">${esc(c.title)}</div>`;
    d.addEventListener('click', () => { switchChat(id); closeSidebar(); });
    historyList.appendChild(d);
  });
}

async function switchChat(id) {
  currentId = id;
  currentSubject = convos[id].subject || 'General';
  document.querySelectorAll('.pill').forEach(p => p.classList.toggle('active', p.dataset.subject === currentSubject));
  topbarTitle.textContent = convos[id].title;
  renderHistory();

  if (currentUser && convos[id].messages.length === 0) {
    const msgs = await sbLoadMessages(id);
    convos[id].messages = msgs.map(m => ({ role: m.role, content: m.content }));
  }

  messagesEl.innerHTML = '';
  if (convos[id].messages.length === 0) { showEmptyState(); return; }
  convos[id].messages.forEach(m => messagesEl.appendChild(makeRow(m.role, formatResponse(m.content))));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── EMPTY STATE ────────────────────────────────
function showEmptyState() {
  const guestNote = isGuest
    ? `<div class="guest-notice">
         <span>💾</span>
         <p>You're in <strong>guest mode</strong> — chats won't be saved.
         <a href="login.html">Create an account</a> to keep your history.</p>
       </div>`
    : '';

  messagesEl.innerHTML = `
  <div class="empty-state">
    ${guestNote}
    <div class="empty-heading">
      <h1>What are we<br>studying <em>today?</em></h1>
      <p>Ask me anything — concepts, homework, equations, problems.</p>
    </div>
    <div class="sug-grid">
      <div class="sug-card" onclick="useSuggestion('Explain photosynthesis simply')">
        <div class="sug-icon">🌱</div>
        <div class="sug-title">Explain a concept</div>
        <div class="sug-sub">Break down any topic</div>
      </div>
      <div class="sug-card" onclick="useSuggestion('Solve: 3x + 5 = 2x + 6')">
        <div class="sug-icon">📐</div>
        <div class="sug-title">Solve an equation</div>
        <div class="sug-sub">Step-by-step math help</div>
      </div>
      <div class="sug-card" onclick="useSuggestion('Quiz me on World War 1 causes')">
        <div class="sug-icon">🧠</div>
        <div class="sug-title">Quiz me</div>
        <div class="sug-sub">Test yourself on any topic</div>
      </div>
      <div class="sug-card" onclick="useSuggestion('Essay outline on climate change')">
        <div class="sug-icon">✍️</div>
        <div class="sug-title">Essay help</div>
        <div class="sug-sub">Outlines, structure, ideas</div>
      </div>
    </div>
  </div>`;
}

// ── MESSAGE RENDERING ──────────────────────────
const AVATAR = `<svg width="16" height="16" viewBox="0 0 34 34" fill="none">
  <path d="M17 5L29 27H5L17 5Z" stroke="#f46d2a" stroke-width="2" stroke-linejoin="round" fill="none"/>
  <path d="M10.5 22h13" stroke="#f46d2a" stroke-width="1.8" stroke-linecap="round"/>
  <circle cx="17" cy="5" r="2" fill="#f46d2a"/>
</svg>`;

function makeRow(role, html) {
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;
  if (role === 'ai') {
    const av = document.createElement('div');
    av.className = 'ai-avatar'; av.innerHTML = AVATAR;
    row.appendChild(av);
  }
  const b = document.createElement('div');
  b.className = `bubble ${role}`; b.innerHTML = html;
  row.appendChild(b);
  return row;
}

function appendRow(role, html) {
  messagesEl.querySelector('.empty-state')?.remove();
  messagesEl.appendChild(makeRow(role, html));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping() {
  messagesEl.querySelector('.empty-state')?.remove();
  const row = document.createElement('div');
  row.className = 'msg-row ai'; row.id = 'typingRow';
  const av = document.createElement('div');
  av.className = 'ai-avatar'; av.innerHTML = AVATAR;
  const b = document.createElement('div');
  b.className = 'bubble ai';
  b.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  row.appendChild(av); row.appendChild(b);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function hideTyping() { $('typingRow')?.remove(); }

// ── SEND ───────────────────────────────────────
async function sendMessage() {
  if (busy) return;
  const text = inputArea.value.trim();
  if (!text) return;

  inputArea.value = ''; inputArea.style.height = 'auto';
  sendBtn.disabled = true; busy = true;
  lastQuestion = text;
  hideTeachToast();

  // Auto-title
  if (convos[currentId].title === 'New Chat') {
    const title = text.length > 36 ? text.slice(0, 33) + '…' : text;
    convos[currentId].title = title;
    topbarTitle.textContent = title;
    if (currentUser) await sbSaveChat(currentId, title, currentSubject, currentUser.id);
    renderHistory();
  }

  convos[currentId].messages.push({ role: 'user', content: text });
  appendRow('user', esc(text));
  if (currentUser) await sbSaveMessage(currentId, 'user', text, currentUser.id);
  showTyping();

  try {
    const history = convos[currentId].messages.slice(0, -1);
    const res = await fetch(`${CONFIG.RENDER_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        subject: currentSubject,
        history,
        user_id: currentUser?.id || null,
        username: currentUsername || null
      })
    });

    const data = await res.json();
    hideTyping();

    if (data.unknown) {
      const msg = "🤔 Hmm, I don't have an answer for that yet! Help me learn below.";
      convos[currentId].messages.push({ role: 'ai', content: msg });
      appendRow('ai', formatResponse(msg));
      if (currentUser) await sbSaveMessage(currentId, 'ai', msg, currentUser.id);
      showTeachToast();
    } else {
      const reply = data.response || 'Sorry, something went wrong. Please try again.';
      convos[currentId].messages.push({ role: 'ai', content: reply });
      appendRow('ai', formatResponse(reply));
      if (currentUser) await sbSaveMessage(currentId, 'ai', reply, currentUser.id);
    }
  } catch (err) {
    hideTyping();
    const msg = '⚠️ Could not reach Axiom\'s backend. Is Render running?';
    appendRow('ai', formatResponse(msg));
    console.error(err);
  }

  sendBtn.disabled = false; busy = false;
  inputArea.focus();
}

// ── FORMAT ─────────────────────────────────────
function formatResponse(raw) {
  let t = esc(raw);
  t = t.replace(/```([\s\S]*?)```/g, (_, c) => `<pre>${c.trim()}</pre>`);
  t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  t = t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/_(.+?)_/g, '<em>$1</em>');
  t = t.replace(/~---~/g, '<hr class="separator">');
  t = t.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  return t;
}

// ── TEACH AXIOM ────────────────────────────────
function showTeachToast()  { teachToast.classList.add('show'); teachInput.focus(); }
function hideTeachToast()  { teachToast.classList.remove('show'); teachInput.value = ''; }

async function submitTeachAxiom() {
  const answer = teachInput.value.trim();
  if (!answer) return;
  await sbSaveSuggestion(lastQuestion, answer, currentUser?.id || null);
  hideTeachToast();
  const msg = '✅ Thanks! We review suggestions weekly and add them to Axiom. 🙏';
  convos[currentId].messages.push({ role: 'ai', content: msg });
  appendRow('ai', formatResponse(msg));
}

// ── MODALS ─────────────────────────────────────
function openModal(type) {
  modal.classList.add('open');

  if (type === 'profile') {
    modalTitle.textContent = isGuest ? 'Guest Mode' : 'Profile';
    if (isGuest) {
      modalBody.innerHTML = `
        <p>You're currently in <strong>guest mode</strong>. Your chats are not being saved.</p>
        <p>Create an account to save your chat history and access Axiom from any device.</p>
        <a href="login.html" class="btn-primary" style="margin-top:8px;display:flex;">Create Account</a>`;
    } else {
      modalBody.innerHTML = `
        <p><strong>Username</strong><br>${esc(currentUsername || '—')}</p>
        <p><strong>Email</strong><br>${esc(currentUser?.email || '—')}</p>
        <p><strong>Version</strong><br>Axiom Aurora v1</p>
        <br>
        <button onclick="handleLogout()" class="btn-secondary">Sign Out</button>`;
    }

  } else if (type === 'settings') {
    modalTitle.textContent = 'Settings';
    modalBody.innerHTML = `
      <p><strong>Version</strong><br>Axiom Aurora v1</p>
      <p><strong>AI Model</strong><br>qwen3 32b via Groq</p>
      <p><strong>Subject Mode</strong><br>Use the pills in the top bar to focus Axiom on a specific subject.</p>
      <p><strong>Shortcuts</strong><br>Enter to send · Shift+Enter for new line</p>
      <p style="color:var(--muted2);font-size:12px;margin-top:16px;">More settings coming in a future update ✦</p>`;

  } else if (type === 'history') {
    modalTitle.textContent = 'All Chats';
    const ids = Object.keys(convos).reverse();
    modalBody.innerHTML = ids.length === 0
      ? '<p>No chats yet — start a new one!</p>'
      : ids.map(id => `
          <div class="hist-item" onclick="switchChat('${id}');closeModal()">
            <span class="hist-label">${esc(convos[id].title)}</span>
            <span class="hist-tag">${esc(convos[id].subject || 'General')}</span>
          </div>`).join('');
  }
}

async function handleLogout() {
  await sbSignOut();
  window.location.href = 'login.html';
}

function closeModal() { modal.classList.remove('open'); }
function esc(t) {
  return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
