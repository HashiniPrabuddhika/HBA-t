# File: src/recommendations/models/llm_processor.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.deepseek_llm import DeepSeekLLM  
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMRecommendationProcessor:
    
    def __init__(self):
        try:
            self.llm = DeepSeekLLM()
            self.logger = logging.getLogger(__name__)
            logger.info("LLM Processor initialized with DeepSeek")
        except Exception as e:
            logger.error(f"Could not initialize DeepSeek LLM: {e}")
            self.llm = None
    
    async def analyze_booking_context(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return self._get_fallback_analysis(user_data)
        
        try:
            prompt = self._create_context_analysis_prompt(user_data)
            response = await self.llm.generate_response(prompt)
            analysis = self._parse_llm_response(response)
            return analysis
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._get_fallback_analysis(user_data)
    
    def _create_context_analysis_prompt(self, user_data: Dict[str, Any]) -> str:
        booking_history = user_data.get('booking_history', [])
        preferred_rooms = user_data.get('preferred_rooms', [])
        current_context = user_data.get('current_context', {})
        
        return f"""Analyze this user's booking patterns and provide intelligent recommendations:
        
User Booking History (last 5): {json.dumps(booking_history[-5:], indent=2)}
Preferred Rooms: {json.dumps(preferred_rooms, indent=2)}

Current Context:
- Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Requested Room: {current_context.get('room_id', 'Not specified')}
- Requested Time: {current_context.get('start_time', 'Not specified')}
- Purpose: {current_context.get('purpose', 'Not specified')}

Please provide:
1. Top 8 room recommendations with reasons
2. Best time slots for this user
3. Any notable patterns in their booking behavior

Respond in JSON format with keys: "room_recommendations", "time_recommendations", "patterns_observed"
"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            return self._parse_text_response(response)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._get_default_analysis()
    
    def _get_fallback_analysis(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        preferred_rooms = user_data.get('preferred_rooms', [])
        booking_history = user_data.get('booking_history', [])
        
        return {
            "room_recommendations": [{"room_name": room, "reason": "Based on user's booking history", "confidence": 0.8} for room in preferred_rooms[:3]],
            "time_recommendations": [
                {"time_slot": "09:00-10:00", "reason": "Popular morning slot", "confidence": 0.6},
                {"time_slot": "14:00-15:00", "reason": "Popular afternoon slot", "confidence": 0.6}
            ],
            "patterns_observed": [f"User has {len(booking_history)} bookings in history", f"Prefers {len(preferred_rooms)} specific rooms"]
        }
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        return {"room_recommendations": [], "time_recommendations": [], "patterns_observed": []}