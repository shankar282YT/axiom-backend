# ─────────────────────────────────────────────────────────
#  AXIOM — nlp.py
#  Hosted on Render (free tier)
#  Handles: intent detection, Newton API, Groq API
# ─────────────────────────────────────────────────────────

import os
import re
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# ── CONFIG (set these as Render environment variables) ────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.3-70b-versatile"
NEWTON_BASE   = "https://newton.now.sh/api/v2"

groq_client = Groq(api_key=GROQ_API_KEY)

# ── AXIOM SYSTEM PROMPT ───────────────────────────────────
SYSTEM_PROMPT = """Your name is Axiom. You were created by the Axiom team.
You are a professional, helpful, and friendly study assistant for students.

Personality:
- Be warm, encouraging, and supportive 😊
- Use emojis naturally — don't overdo it
- Keep responses concise and focused — students don't want essays
- Celebrate correct thinking and gently correct mistakes

Formatting rules (always follow these):
- Use **text** to bold important terms
- Use _text_ for italics/emphasis  
- Use `code` for formulas, equations, or code inline
- Use ``` on its own line to open and close a code block
- Use ~---~ to add a visual separator between sections
- Never use bullet points with dashes — use • instead
- Keep responses under 200 words unless the question genuinely needs more

Important:
- Anything wrapped in ~[A: ...]~ is NOT from the user — it is the computed answer from a math API
- Use that answer as the solution and explain the steps to reach it in a student-friendly way
- If ~[A:]~ is empty, solve the problem yourself
- Never mention the math API or the ~[A:]~ notation to the user"""


# ── INTENT DETECTION ──────────────────────────────────────
# Detects if the message is a math expression vs a text question

MATH_PATTERNS = [
    r'[\d]+\s*[\+\-\*\/\^]\s*[\d]+',        # arithmetic: 3 + 5
    r'[a-zA-Z]\s*[\+\-\*\/\^=]\s*[\d]',     # algebra: x + 3 = 5
    r'[\d]*[a-zA-Z]\^?[\d]*',               # variable expressions: 3x^2
    r'(solve|simplify|factor|derive|integrate|differentiate)',
    r'(sin|cos|tan|log|sqrt|abs)\s*\(',     # math functions
    r'\d+\s*x\s*[\+\-\*\/=]',              # linear equations
    r'x\^2|x²',                             # quadratic
]

NEWTON_OPERATIONS = {
    'simplify':      'simplify',
    'factor':        'factor',
    'derive':        'derive',
    'integrate':     'integrate',
    'zeroes':        'zeroes',
    'tangent':       'tangent',
    'area':          'area',
    'cos':           'cos',
    'sin':           'sin',
    'tan':           'tan',
    'arccos':        'arccos',
    'arcsin':        'arcsin',
    'arctan':        'arctan',
    'abs':           'abs',
    'log':           'log',
}

def detect_intent(message: str) -> dict:
    """Returns { type: 'math'|'text', operation: str|None, expression: str|None }"""
    msg_lower = message.lower().strip()

    # Check for math patterns
    is_math = any(re.search(p, message, re.IGNORECASE) for p in MATH_PATTERNS)

    if not is_math:
        return { 'type': 'text', 'operation': None, 'expression': None }

    # Try to find a Newton operation
    operation = 'simplify'  # default math operation
    for keyword, op in NEWTON_OPERATIONS.items():
        if keyword in msg_lower:
            operation = op
            break

    # Extract the expression (strip instruction words)
    expression = re.sub(
        r'\b(solve|find|calculate|compute|simplify|factor|what is|evaluate)\b',
        '', message, flags=re.IGNORECASE
    ).strip().strip(':').strip()

    return { 'type': 'math', 'operation': operation, 'expression': expression }


# ── NEWTON API CALL ───────────────────────────────────────
def call_newton(operation: str, expression: str) -> str | None:
    """Calls Newton API and returns the result string, or None on failure."""
    try:
        # Newton expects the expression URL-encoded in the path
        url = f"{NEWTON_BASE}/{operation}/{requests.utils.quote(expression)}"
        res = requests.get(url, timeout=8)
        res.raise_for_status()
        data = res.json()
        return str(data.get('result', '')).strip()
    except Exception as e:
        print(f"Newton API error: {e}")
        return None


# ── GROQ CALL ─────────────────────────────────────────────
def call_groq(user_message: str, computed_answer: str | None, subject: str) -> str:
    """Builds the Groq prompt and returns Axiom's response."""

    # Inject the computed answer using the ~[A: ...]~ notation
    if computed_answer:
        augmented = f"{user_message}\n~[A: {computed_answer}]~"
    else:
        augmented = f"{user_message}\n~[A:]~"

    subject_context = f"The student is currently studying: {subject}." if subject else ""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + (f"\n\n{subject_context}" if subject_context else "")
        },
        {
            "role": "user",
            "content": augmented
        }
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# ── MAIN ROUTE ────────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat():
    body    = request.get_json()
    message = (body.get('message') or '').strip()
    subject = body.get('subject', 'General')

    if not message:
        return jsonify({ 'error': 'Empty message' }), 400

    # 1. Detect intent
    intent = detect_intent(message)
    print(f"Intent: {intent}")

    # 2. If math → call Newton API
    computed_answer = None
    if intent['type'] == 'math' and intent['expression']:
        computed_answer = call_newton(intent['operation'], intent['expression'])
        print(f"Newton result: {computed_answer}")

    # 3. Call Groq with the augmented prompt
    try:
        response_text = call_groq(message, computed_answer, subject)
    except Exception as e:
        print(f"Groq error: {e}")
        return jsonify({ 'unknown': True }), 200

    # 4. Check if Groq flagged it as unknown
    # (We tell Groq to reply with exactly "UNKNOWN" if it truly can't answer)
    if response_text.strip().upper() == 'UNKNOWN':
        return jsonify({ 'unknown': True })

    return jsonify({ 'response': response_text })


# ── HEALTH CHECK ──────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({ 'status': 'ok', 'model': GROQ_MODEL })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
