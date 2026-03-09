# ============================================================
#  AXIOM · engine.py
#  Rule-based response engine
#  Input : intent JSON from NLP
#  Output: response string
# ============================================================

import random

# ── Response banks ───────────────────────────────────────────
# Multiple responses per intent keep things feeling natural

RESPONSES = {

    "greet": [
        "Hey! Good to have you here. What's on your mind?",
        "Hi there! I'm AXIOM — ask me anything.",
        "Hello! Always great to see you. How can I help?",
        "Hey! What can I do for you today?",
        "Hi! Ready when you are.",
    ],

    "who_are_you": [
        "I'm AXIOM — a hybrid AI that combines a neural network with a rule-based engine to understand and respond to you.",
        "The name's AXIOM. I'm an AI built with a mix of natural language processing and rule-based logic to give you accurate, structured responses.",
        "I'm AXIOM, your AI assistant. I use a hybrid approach — NLP to understand what you're saying, and a rule engine to decide how to respond.",
        "AXIOM, at your service! I'm a hybrid AI — part neural network, part rule-based system. Think of me as structured intelligence.",
        "Good question! I'm AXIOM — an AI that understands natural language and responds using a rule-based engine underneath.",
    ],

    "who_made_you": [
        "I was built by AXIOM — the same name I carry. Pretty fitting, right?",
        "AXIOM created me! My architecture, my training, my rules — all crafted by the AXIOM team.",
        "My creator? That would be AXIOM. They designed both my brain and my logic engine.",
        "I was made by AXIOM. They built me from the ground up — NLP model, rule engine and all.",
        "AXIOM is behind everything I am. They put in the work to make me think and respond the way I do.",
    ],

    "what_can_you_do": [
        "Right now I'm getting started, so I'm best at introducing myself! More capabilities — like homework help — are coming soon.",
        "At the moment I'm great at telling you who I am and how I work. Homework helper features are on the way!",
        "I'm currently in my early phase — I can have a conversation and introduce myself. Expect a lot more from me soon!",
        "For now, I can chat with you and tell you all about myself. I'm being trained to help with homework too — stay tuned!",
        "I'm still growing! Today I can introduce myself and chat. Tomorrow? Homework help, problem solving, and more.",
    ],

    "farewell": [
        "See you later! Come back anytime.",
        "Goodbye! It was great talking to you.",
        "Take care! I'll be here whenever you need me.",
        "Catch you next time! 👋",
        "Bye! Don't hesitate to come back if you need anything.",
    ],

    "fallback": [
        "Hmm, I'm not sure I can help with that just yet — I'm still learning! Try asking me who I am or what I can do.",
        "That's a bit outside what I know right now. I'm best at introductions — ask me about myself!",
        "I don't have an answer for that one yet. I'm growing fast though! For now, feel free to ask me about AXIOM.",
        "Interesting question, but that's beyond my current abilities. Ask me who I am — I'm much better at that!",
        "I haven't learned that yet! Try asking what I can do, and I'll tell you what I'm capable of right now.",
    ],
}

# Confidence threshold — below this, always fallback
CONFIDENCE_THRESHOLD = 0.55


# ── Engine ───────────────────────────────────────────────────

def execute(intent_json: dict) -> str:
    """
    intent_json: {
        "intent":     str,
        "confidence": float,
        "all_probs":  dict   (optional)
    }
    Returns a response string.
    """
    intent     = intent_json.get("intent", "fallback")
    confidence = intent_json.get("confidence", 0.0)

    # Low confidence → fallback regardless of predicted intent
    if confidence < CONFIDENCE_THRESHOLD:
        intent = "fallback"

    # Pick a response from the bank
    bank = RESPONSES.get(intent, RESPONSES["fallback"])
    return random.choice(bank)
