from pydantic import BaseModel, Field
import requests
from typing import Optional, List, Any

from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation

from config.app_config import get_settings
from utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class DeepSeekLLM(BaseLLM):
    
    api_key: str = Field(default_factory=lambda: settings.OPENAI_API_KEY2)
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "z-ai/glm-4.5-air:free"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
      
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract structured data from questions. "
                        "Output must be JSON with: room_name, date (yyyy-mm-dd), "
                        "start_time and end_time (HH:MM 24-hour format)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "HBA"
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.RequestException as e:
            logger.error(f"DeepSeek API request failed: {e}")
            raise RuntimeError(f"Failed to call DeepSeek API: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid API response format: {e}")
            raise RuntimeError(f"Invalid DeepSeek API response: {e}")
    
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
       
        generations = []
        for prompt in prompts:
            response_text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=response_text)])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "deepseek_llm"