RECURRENCE_PROMPT = """
You are an intelligent assistant that extracts recurrence patterns from booking requests.

From the following user request:
"{user_input}"

Detect if it contains a recurring booking pattern.
If yes, output the rule in strict JSON:
{{
  "is_recurring": true,
  "frequency": "daily" | "weekly" | "monthly",
  "days_of_week": ["Monday", "Wednesday"],   // only for weekly
  "start_time": "HH:MM",
  "room_name": "<room_name>",
  "module_code": "<module_code>",
  "end_time": "HH:MM",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD"  // last day of recurrence
}}

If no recurrence is found, return:
{{
  "is_recurring": false
}}

Respond in JSON only.
"""
