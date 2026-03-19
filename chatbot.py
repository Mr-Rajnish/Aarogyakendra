import requests

# 🔥 PUT YOUR GROQ API KEY HERE


import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# =========================
# 🤖 AI CALL (GROQ)
# =========================
def call_ai(prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)

        print("Groq Status:", response.status_code)

        result = response.json()

        print("Groq Response:", result)  # 🔥 debug

        # ✅ SAFE CHECK
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]

        # ❌ If API error
        if "error" in result:
            raise Exception(result["error"])

        # ❌ Unknown response
        raise Exception("Invalid response from AI")

    except Exception as e:
        print("❌ AI Error:", e)
        return "AI is currently unavailable. Please try again."


# =========================
# 🧠 MAIN FUNCTION
# =========================
def generate_health_assessment(symptoms_text, report_text=None):

    prompt = f"""
You are a medical assistant.

Symptoms: {symptoms_text}
Report: {report_text if report_text else "None"}

Give:
- possible diseases
- precautions
- lifestyle changes
- medications (with disclaimer)

Keep answer simple.
"""

    try:
        response = call_ai(prompt)

        return {
            "summary": response,
            "diseases": ["AI-based prediction"],
            "precautions": ["Consult doctor"],
            "lifestyle_changes": ["Healthy lifestyle"],
            "medications": [],
            "medication_disclaimer": "Consult a doctor before taking any medication",
            "model_used": "groq"
        }

    except Exception as e:
        print("AI Error:", e)

        return {
            "summary": "AI not available",
            "diseases": [],
            "precautions": ["Consult doctor"],
            "lifestyle_changes": [],
            "medications": [],
            "medication_disclaimer": "Consult a doctor before taking any medication",
            "model_used": "fallback"
        }


# =========================
# 💬 CHAT FORMAT
# =========================
def assessment_to_chat_markdown(data):
    return f"""
**Summary:** {data.get("summary")}

**Precautions:**
- {"".join(data.get("precautions", []))}

**Disclaimer:** {data.get("medication_disclaimer")}
"""