-- ─────────────────────────────────────────────
--  AXIOM — Supabase Schema
--  Run this in your Supabase SQL editor
-- ─────────────────────────────────────────────

-- Chat sessions
CREATE TABLE chats (
  id          TEXT PRIMARY KEY,
  title       TEXT NOT NULL DEFAULT 'New Chat',
  subject     TEXT NOT NULL DEFAULT 'General',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Individual messages
CREATE TABLE messages (
  id          BIGSERIAL PRIMARY KEY,
  chat_id     TEXT REFERENCES chats(id) ON DELETE CASCADE,
  role        TEXT NOT NULL CHECK (role IN ('user', 'ai')),
  content     TEXT NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- "Teach Axiom" suggestions from users
CREATE TABLE suggestions (
  id          BIGSERIAL PRIMARY KEY,
  question    TEXT NOT NULL,
  answer      TEXT NOT NULL,
  reviewed    BOOLEAN DEFAULT FALSE,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX idx_messages_chat_id ON messages(chat_id);
CREATE INDEX idx_suggestions_reviewed ON suggestions(reviewed);
