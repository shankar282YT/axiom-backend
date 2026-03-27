// ─────────────────────────────────────────────
//  AXIOM — supabase.js
// ─────────────────────────────────────────────

const _sb = supabase.createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_ANON_KEY);

// ══ AUTH ══════════════════════════════════════

async function sbSignUp(email, password, username) {
  const { data, error } = await _sb.auth.signUp({ email, password });

  if (error) {
    // Friendlier messages for common Supabase errors
    if (error.status === 429) return { error: 'Too many sign-up attempts. Please wait a few minutes and try again.' };
    if (error.message.includes('already registered')) return { error: 'An account with this email already exists. Try signing in.' };
    return { error: error.message };
  }

  // Supabase returns a user but with no session if email confirmation is required
  // data.session will be null in that case
  if (!data.session) {
    // Still try to insert the profile row — user exists, just unconfirmed
    if (data.user) {
      await _sb.from('users').insert([{ id: data.user.id, email, username }]);
    }
    return { pending: true }; // Signal to UI to show "check your email"
  }

  // Auto-confirmed (e.g. you disabled email confirmation in Supabase dashboard)
  const { error: pe } = await _sb.from('users').insert([{ id: data.user.id, email, username }]);
  if (pe && !pe.message.includes('duplicate')) console.warn('Profile insert:', pe.message);
  return { user: data.user };
}

async function sbSignIn(email, password) {
  const { data, error } = await _sb.auth.signInWithPassword({ email, password });

  if (error) {
    if (error.status === 400) {
      // Most common cause: wrong password OR email not confirmed yet
      return { error: 'Invalid email or password. If you just signed up, please confirm your email first.' };
    }
    if (error.status === 429) return { error: 'Too many attempts. Please wait a few minutes.' };
    return { error: error.message };
  }

  return { user: data.user };
}

async function sbSignInWithOAuth(provider) {
  const { error } = await _sb.auth.signInWithOAuth({
    provider,
    options: { redirectTo: window.location.origin + '/index.html' }
  });
  if (error) return { error: error.message };
  return {};
}

async function sbResetPassword(email) {
  const { error } = await _sb.auth.resetPasswordForEmail(email, {
    redirectTo: window.location.origin + '/reset.html'
  });
  if (error) return { error: error.message };
  return {};
}

async function sbSignOut() { await _sb.auth.signOut(); }

async function sbGetCurrentUser() {
  const { data } = await _sb.auth.getUser();
  return data?.user || null;
}

async function sbGetUsername(userId) {
  const { data } = await _sb.from('users').select('username').eq('id', userId).single();
  return data?.username || null;
}

// ══ CHATS ═════════════════════════════════════

async function sbSaveChat(chatId, title, subject, userId) {
  if (!userId) return;
  const { error } = await _sb.from('chats').upsert([{ id: chatId, title, subject, user_id: userId }]);
  if (error) console.warn('saveChat:', error.message);
}

async function sbLoadChats(userId) {
  if (!userId) return [];
  const { data, error } = await _sb.from('chats').select('*').eq('user_id', userId).order('created_at', { ascending: false });
  if (error) { console.warn('loadChats:', error.message); return []; }
  return data || [];
}

async function sbDeleteChat(chatId) {
  const { error } = await _sb.from('chats').delete().eq('id', chatId);
  if (error) console.warn('deleteChat:', error.message);
}

// ══ MESSAGES ══════════════════════════════════

async function sbSaveMessage(chatId, role, content, userId) {
  if (!userId) return;
  const { error } = await _sb.from('messages').insert([{ chat_id: chatId, role, content, user_id: userId }]);
  if (error) console.warn('saveMessage:', error.message);
}

async function sbLoadMessages(chatId) {
  const { data, error } = await _sb.from('messages').select('*').eq('chat_id', chatId).order('created_at', { ascending: true });
  if (error) { console.warn('loadMessages:', error.message); return []; }
  return data || [];
}

// ══ SUGGESTIONS ═══════════════════════════════

async function sbSaveSuggestion(question, answer, userId) {
  const row = { question, answer };
  if (userId) row.user_id = userId;
  const { error } = await _sb.from('suggestions').insert([row]);
  if (error) console.warn('saveSuggestion:', error.message);
}

async function sbEnsureProfile(user) {
  // Check if profile exists
  const { data } = await _sb.from('users').select('username').eq('id', user.id).single();
  if (data) return; // already exists

  // Generate username from Google display name or email
  const rawName = user.user_metadata?.full_name || user.email.split('@')[0];
  // Clean it up — remove spaces, lowercase, add random suffix
  const username = rawName.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '') + '_' + Math.random().toString(36).slice(2, 5);

  await _sb.from('users').insert([{
    id: user.id,
    email: user.email,
    username
  }]);
}
