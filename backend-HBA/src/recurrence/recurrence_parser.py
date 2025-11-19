import json
import re
from src.deepseek_llm import DeepSeekLLM
from src.recurrence.recurrence_prompt import RECURRENCE_PROMPT

def extract_recurrence(user_input: str) -> dict:
    print(f"[Recurrence Parser] Received input: {user_input}")

    # ✅ Step 1: Handle small talk / unrelated queries early
    small_talk = {"hi", "hello", "hey", "thanks", "thank you"}
    if user_input.lower().strip() in small_talk:
        print("[Recurrence Parser] Detected small talk, skipping LLM call")
        return {"is_recurring": False, "reason": "small_talk"}

    llm = DeepSeekLLM()
    prompt = RECURRENCE_PROMPT.format(user_input=user_input)
    print(f"[Recurrence Parser] Generated prompt for LLM:\n{prompt}")

    try:
        # ✅ Step 2: Try LLM call safely
        raw = llm._call(prompt)
        print(f"[Recurrence Parser] Raw LLM response:\n{raw}")

        cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        print(f"[Recurrence Parser] Cleaned JSON string:\n{cleaned}")

        parsed = json.loads(cleaned)
        print(f"[Recurrence Parser] Parsed recurrence data: {parsed}")
        return parsed

    except json.JSONDecodeError:
        print("[Recurrence Parser] Failed to parse JSON, returning default non-recurring response")
        return {"is_recurring": False, "reason": "json_error"}

    except Exception as e:
        # ✅ Step 3: Catch API errors (429, timeouts, etc.)
        print(f"[Recurrence Parser] LLM call failed: {e}")
        return {"is_recurring": False, "reason": "llm_unavailable"}
