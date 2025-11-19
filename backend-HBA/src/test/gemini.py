from langchain.llms.base import LLM
from pydantic import BaseModel, Field  # Use Pydantic directly
import requests
import os
from typing import Optional, List, Any
from dotenv import load_dotenv
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
# Load environment variables
load_dotenv()
print(f"🔐 Loaded API Key: {os.getenv('GEMINI_API_KEY3')}")

class Gemini(BaseLLM):
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY3"))
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "gpt-oss-20b:free" # Replace with the correct model name for DeepSeek

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Make a call to the Gemini model (via OpenRouter API).
        """
        payload = {
           "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract structured data from room availability questions. "
                        "Output must be JSON with: room_name, date (yyyy-mm-dd), "
                        "start_time and end_time (HH:MM 24-hour format)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {os.getenv('GEMINI_API_KEY3')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",   # Optional - can update for production
            "X-Title": "HBA"            # Optional - can customize
        }

        # data = {
        #     "model": self.model,
        #     "messages": [{"role": "user", "content": prompt}],
        # }

        try:
            # response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
            response = requests.post(self.base_url, headers=headers, json=payload)

            response.raise_for_status()  # Raises an HTTPError if the status code is 4xx or 5xx
            return response.json()['choices'][0]['message']['content']

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to call Gemini API: {e}")

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        """
        Implemented to satisfy BaseLLM abstract class requirements.
        """
        generations = []
        for prompt in prompts:
            response_text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=response_text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "gemini"
