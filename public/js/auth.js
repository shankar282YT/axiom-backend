// ─────────────────────────────────────────────
//  AXIOM — auth.js
//  Login / Signup page logic
// ─────────────────────────────────────────────

// ── BOOT ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // If already logged in redirect to chat
  const user = await sbGetCurrentUser();
  if (user) { window.location.href = 'index.html'; return; }

  setupTabs();
  renderLoginForm();
});

// ── TABS ───────────────────────────────────────
function setupTabs() {
  document.querySelectorAll('.auth-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      clearMessages();
      if (tab.dataset.tab === 'login')  renderLoginForm();
      if (tab.dataset.tab === 'signup') renderSignupForm();
    });
  });
}

function setActiveTab(name) {
  document.querySelectorAll('.auth-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
}

// ── FORMS ──────────────────────────────────────
function renderLoginForm() {
  document.getElementById('authForm').innerHTML = `
    <div class="field">
      <label>Email</label>
      <input type="email" id="fEmail" placeholder="you@email.com" autocomplete="email"/>
    </div>
    <div class="field">
      <label>Password</label>
      <input type="password" id="fPassword" placeholder="Your password" autocomplete="current-password"/>
    </div>
    <div class="forgot-link"><a onclick="showForgotForm()">Forgot password?</a></div>
    <button class="btn-primary" id="submitBtn" onclick="handleLogin()">Sign In</button>`;

  document.getElementById('fPassword').addEventListener('keydown', e => {
    if (e.key === 'Enter') handleLogin();
  });
}

function renderSignupForm() {
  document.getElementById('authForm').innerHTML = `
    <div class="field">
      <label>Username</label>
      <input type="text" id="fUsername" placeholder="coolstudent42" autocomplete="username"/>
    </div>
    <div class="field">
      <label>Email</label>
      <input type="email" id="fEmail" placeholder="you@email.com" autocomplete="email"/>
    </div>
    <div class="field">
      <label>Password</label>
      <input type="password" id="fPassword" placeholder="Min. 8 characters" autocomplete="new-password"/>
    </div>
    <button class="btn-primary" id="submitBtn" onclick="handleSignup()">Create Account</button>`;

  document.getElementById('fPassword').addEventListener('keydown', e => {
    if (e.key === 'Enter') handleSignup();
  });
}

function showForgotForm() {
  document.getElementById('authForm').innerHTML = `
    <p style="font-size:13.5px;color:var(--muted);margin-bottom:16px;line-height:1.6;">
      Enter your email and we'll send you a link to reset your password.
    </p>
    <div class="field">
      <label>Email</label>
      <input type="email" id="fEmail" placeholder="you@email.com" autocomplete="email"/>
    </div>
    <button class="btn-primary" id="submitBtn" onclick="handleForgot()">Send Reset Link</button>
    <button class="btn-secondary" style="margin-top:8px;" onclick="renderLoginForm();setActiveTab('login')">Back to Sign In</button>`;
}

// ── HANDLERS ───────────────────────────────────
async function handleLogin() {
  const email    = val('fEmail');
  const password = val('fPassword');
  if (!email || !password) { showError('Please fill in all fields.'); return; }

  setLoading('Signing in...');
  const result = await sbSignIn(email, password);
  if (result.error) { showError(result.error); setLoading(null, 'Sign In'); return; }

  window.location.href = 'index.html';
}

async function handleSignup() {
  const username = val('fUsername');
  const email    = val('fEmail');
  const password = val('fPassword');

  if (!username || !email || !password) { showError('Please fill in all fields.'); return; }
  if (username.length < 3) { showError('Username must be at least 3 characters.'); return; }
  if (password.length < 8) { showError('Password must be at least 8 characters.'); return; }

  setLoading('Creating account...');
  const result = await sbSignUp(email, password, username);
  
  if (result.error) { 
    showError(result.error); 
    setLoading(null, 'Create Account'); 
    return; 
  }

  if (result.pending) {
    // Email confirmation required — most common on Supabase free tier
    showSuccess('✅ Account created! Check your inbox to confirm your email, then sign in.');
    setTimeout(() => { setActiveTab('login'); renderLoginForm(); }, 3000);
    return;
  }

  // Auto-confirmed — go straight to app
  window.location.href = 'index.html';
}

async function handleForgot() {
  const email = val('fEmail');
  if (!email) { showError('Please enter your email.'); return; }

  setLoading('Sending...');
  const result = await sbResetPassword(email);
  if (result.error) { showError(result.error); setLoading(null, 'Send Reset Link'); return; }

  showSuccess('Reset link sent! Check your inbox.');
}

async function handleOAuth(provider) {
  const result = await sbSignInWithOAuth(provider);
  if (result.error) showError(result.error);
  // OAuth redirects automatically on success
}

function handleGuest() {
  // Set guest flag in sessionStorage and go to chat
  sessionStorage.setItem('axiom_guest', '1');
  window.location.href = 'index.html';
}

// ── HELPERS ────────────────────────────────────
function val(id) {
  return (document.getElementById(id)?.value || '').trim();
}

function setLoading(loadingText, resetText) {
  const btn = document.getElementById('submitBtn');
  if (!btn) return;
  if (loadingText) {
    btn.textContent = loadingText;
    btn.disabled = true;
  } else {
    btn.textContent = resetText || 'Submit';
    btn.disabled = false;
  }
}

function showError(msg) {
  const el = document.getElementById('authError');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  el.classList.remove('msg-success');
  el.classList.add('msg-error');
}

function showSuccess(msg) {
  const el = document.getElementById('authError');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  el.classList.remove('msg-error');
  el.classList.add('msg-success');
}

function clearMessages() {
  const el = document.getElementById('authError');
  if (!el) return;
  el.classList.remove('show', 'msg-error', 'msg-success');
  el.textContent = '';
}
