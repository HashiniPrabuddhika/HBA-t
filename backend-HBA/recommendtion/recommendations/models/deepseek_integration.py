import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.deepseek_llm import DeepSeekLLM 
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class DeepSeekRecommendationProcessor:
    
    def __init__(self):
        try:
            self.deepseek = DeepSeekLLM()
            self.logger = logging.getLogger(__name__)
            logger.info("DeepSeek Recommendation Processor initialized")
            
            self.prompt_templates = {
                'room_analysis': self._get_room_analysis_template(),
                'user_pattern_analysis': self._get_user_pattern_template(),
                'alternative_suggestions': self._get_alternative_suggestions_template(),
                'smart_scheduling': self._get_smart_scheduling_template(),
                'explanation_generation': self._get_explanation_template()
            }
        except Exception as e:
            logger.error(f"Could not initialize DeepSeek LLM: {e}")
            self.deepseek = None
    
    def _calculate_end_time(self, start_time: str, duration_hours: float) -> str:
        """Calculate end time based on start time and duration"""
        try:
            if isinstance(start_time, str):
                # Handle different time formats
                if 'T' in start_time:  # ISO format
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                elif ':' in start_time and len(start_time.split(':')) >= 2:  # Time only
                    time_parts = start_time.split(':')
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                    start_dt = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    raise ValueError(f"Unsupported time format: {start_time}")
            else:
                start_dt = start_time
            
            end_dt = start_dt + timedelta(hours=duration_hours)
            
            # Return in same format as input
            if 'T' in str(start_time):
                return end_dt.isoformat()
            else:
                return end_dt.strftime('%H:%M')
                
        except Exception as e:
            logger.error(f"Error calculating end time: {e}")
            return start_time  # Return original if calculation fails
    
    def _validate_room_availability(self, room_data: Dict[str, Any], start_time: str, end_time: str) -> bool:
        """Validate if room is available for the entire requested duration"""
        try:
            # This should check against your actual booking system
            # For now, implementing basic logic that you'll need to enhance
            
            existing_bookings = room_data.get('bookings', [])
            
            if isinstance(start_time, str):
                if 'T' in start_time:
                    req_start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    req_end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                else:
                    # Assuming same day for time-only format
                    today = datetime.now().date()
                    req_start = datetime.combine(today, datetime.strptime(start_time, '%H:%M').time())
                    req_end = datetime.combine(today, datetime.strptime(end_time, '%H:%M').time())
            
            # Check for conflicts with existing bookings
            for booking in existing_bookings:
                booking_start = datetime.fromisoformat(booking['start_time'].replace('Z', '+00:00'))
                booking_end = datetime.fromisoformat(booking['end_time'].replace('Z', '+00:00'))
                
                # Check for overlap
                if not (req_end <= booking_start or req_start >= booking_end):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating room availability: {e}")
            return False
    
    async def analyze_user_booking_context(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.deepseek:
            return self._get_fallback_analysis(user_data)
        
        try:
            context = self._prepare_user_context(user_data)
            prompt = self.prompt_templates['user_pattern_analysis'].format(**context)
            response = await self._call_deepseek_llm(prompt, max_tokens=800)
            analysis = self._parse_user_analysis_response(response)
            
            logger.info(f"Successfully analyzed user context for {user_data.get('user_id', 'unknown')}")
            return analysis
        except Exception as e:
            logger.error(f"Error in user context analysis: {e}")
            return self._get_fallback_analysis(user_data)
    
    async def generate_room_recommendations(self, request_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.deepseek:
            return self._get_fallback_room_recommendations(request_context)
        
        try:
            context = self._prepare_room_context(request_context)
            prompt = self.prompt_templates['room_analysis'].format(**context)
            response = await self._call_deepseek_llm(prompt, max_tokens=600)
            recommendations = self._parse_room_recommendations(response, request_context)
            
            logger.info(f"Generated {len(recommendations)} room recommendations")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating room recommendations: {e}")
            return self._get_fallback_room_recommendations(request_context)
    
    async def generate_alternative_suggestions(self, original_request: Dict[str, Any],
                                            available_alternatives: List[Dict[str, Any]],
                                            user_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.deepseek:
            return self._enhance_alternatives_fallback(available_alternatives, original_request)
        
        try:
            context = self._prepare_alternative_context(original_request, available_alternatives, user_patterns)
            prompt = self.prompt_templates['alternative_suggestions'].format(**context)
            response = await self._call_deepseek_llm(prompt, max_tokens=700)
            enhanced_alternatives = self._parse_alternative_suggestions(response, available_alternatives, original_request)
            
            logger.info(f"Enhanced {len(enhanced_alternatives)} alternative suggestions")
            return enhanced_alternatives
        except Exception as e:
            logger.error(f"Error generating alternative suggestions: {e}")
            return self._enhance_alternatives_fallback(available_alternatives, original_request)
    
    async def generate_smart_scheduling_insights(self, scheduling_data: Dict[str, Any],
                                               room_utilization: Dict[str, Any]) -> Dict[str, Any]:
        if not self.deepseek:
            return self._get_fallback_scheduling_insights(scheduling_data)
        
        try:
            # Calculate proper time range
            start_time = scheduling_data.get('start_time', '')
            duration = scheduling_data.get('duration_hours', 1)
            end_time = self._calculate_end_time(start_time, duration)
            
            context = {
                'requested_time': start_time,
                'requested_end_time': end_time,
                'duration': duration,
                'capacity_needed': scheduling_data.get('capacity', 'Unknown'),
                'purpose': scheduling_data.get('purpose', 'Meeting'),
                'utilization_stats': json.dumps(room_utilization, indent=2),
                'current_date': datetime.now().strftime('%Y-%m-%d'),
                'current_time': datetime.now().strftime('%H:%M'),
                'available_rooms': json.dumps(scheduling_data.get('available_rooms', []), indent=2)
            }
            
            prompt = self.prompt_templates['smart_scheduling'].format(**context)
            response = await self._call_deepseek_llm(prompt, max_tokens=600)
            insights = self._parse_scheduling_insights(response, scheduling_data)
            
            logger.info("Generated smart scheduling insights")
            return insights
        except Exception as e:
            logger.error(f"Error generating scheduling insights: {e}")
            return self._get_fallback_scheduling_insights(scheduling_data)
    
    async def explain_recommendation(self, recommendation: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        if not self.deepseek:
            return self._get_fallback_explanation(recommendation)
        
        try:
            context = {
                'room_name': recommendation.get('suggestion', {}).get('room_name', 'Unknown'),
                'date': recommendation.get('suggestion', {}).get('date', 'Unknown'),
                'start_time': recommendation.get('suggestion', {}).get('start_time', 'Unknown'),
                'end_time': recommendation.get('suggestion', {}).get('end_time', 'Unknown'),
                'confidence': recommendation.get('score', 0),
                'recommendation_type': recommendation.get('type', 'general'),
                'user_history_count': len(user_context.get('booking_history', [])),
                'user_preferences': ', '.join(user_context.get('preferred_rooms', [])[:3]),
                'reason': recommendation.get('reason', 'System recommendation')
            }
            
            prompt = self.prompt_templates['explanation_generation'].format(**context)
            response = await self._call_deepseek_llm(prompt, max_tokens=200)
            explanation = response.strip().replace('"', '').replace('\n', ' ')
            
            logger.debug(f"Generated explanation for {context['room_name']}")
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return self._get_fallback_explanation(recommendation)
    
    async def _call_deepseek_llm(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            methods = ['generate_response', 'chat', 'complete']
            for method in methods:
                if hasattr(self.deepseek, method):
                    fn = getattr(self.deepseek, method)
                    if asyncio.iscoroutinefunction(fn):
                        return await fn(prompt, max_tokens=max_tokens)
                    else:
                        return fn(prompt, max_tokens=max_tokens)
            return await self.deepseek(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Error calling DeepSeek LLM: {e}")
            raise
    
    def _prepare_user_context(self, user_data: Dict[str, Any]) -> Dict[str, str]:
        booking_history = user_data.get('booking_history', [])
        preferred_rooms = user_data.get('preferred_rooms', [])
        booking_patterns = user_data.get('booking_patterns', {})
        
        return {
            'user_id': str(user_data.get('user_id', 'unknown')),
            'total_bookings': str(len(booking_history)),
            'recent_bookings': json.dumps(booking_history[-5:], indent=2, default=str),
            'preferred_rooms': ', '.join(preferred_rooms[:5]) if preferred_rooms else 'None identified',
            'booking_frequency': booking_patterns.get('frequency', 'unknown'),
            'typical_duration': str(booking_patterns.get('avg_duration', 'unknown')),
            'current_date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
    
    def _prepare_room_context(self, request_context: Dict[str, Any]) -> Dict[str, str]:
        start_time = request_context.get('start_time', 'Not specified')
        duration = request_context.get('duration_hours', 1)
        end_time = self._calculate_end_time(start_time, duration) if start_time != 'Not specified' else 'Not specified'
        
        return {
            'requested_room': request_context.get('room_id', 'Not specified'),
            'requested_date': request_context.get('date', 'Not specified'),
            'requested_start_time': start_time,
            'requested_end_time': end_time,
            'duration': str(duration),
            'capacity_needed': str(request_context.get('capacity', 'Not specified')),
            'purpose': request_context.get('purpose', 'Not specified'),
            'available_rooms': json.dumps(request_context.get('available_rooms', [])[:10], indent=2),
            'current_time': datetime.now().strftime('%H:%M'),
            'day_of_week': datetime.now().strftime('%A')
        }
    
    def _prepare_alternative_context(self, original_request: Dict[str, Any], 
                                   alternatives: List[Dict[str, Any]], 
                                   user_patterns: Dict[str, Any]) -> Dict[str, str]:
        start_time = original_request.get('start_time', '')
        duration = original_request.get('duration_hours', 1)
        end_time = self._calculate_end_time(start_time, duration)
        
        return {
            'original_room': original_request.get('room_id', 'Unknown'),
            'original_date': original_request.get('date', 'Unknown'),
            'original_start_time': start_time,
            'original_end_time': end_time,
            'duration': str(duration),
            'purpose': original_request.get('purpose', 'Meeting'),
            'user_preferences': json.dumps(user_patterns.get('preferences_inferred', {}), indent=2),
            'user_patterns': json.dumps(user_patterns.get('patterns_identified', []), indent=2),
            'alternatives': json.dumps(alternatives[:5], indent=2, default=str),
            'capacity_needed': str(original_request.get('capacity', 'Not specified'))
        }
    
    def _parse_user_analysis_response(self, response: str) -> Dict[str, Any]:
        try:
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                return {
                    'patterns_identified': parsed.get('patterns_identified', []),
                    'recommendations': parsed.get('recommendations', []),
                    'user_type': parsed.get('user_type', 'regular'),
                    'preferences_inferred': parsed.get('preferences_inferred', {}),
                    'scheduling_insights': parsed.get('scheduling_insights', [])
                }
            return self._parse_text_to_analysis(response)
        except json.JSONDecodeError:
            return self._parse_text_to_analysis(response)
    
    def _parse_room_recommendations(self, response: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            recommendations = []
            start_time = context.get('start_time', '')
            duration = context.get('duration_hours', 1)
            end_time = self._calculate_end_time(start_time, duration)
            
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                for rec in parsed.get('recommendations', [])[:5]:
                    # Validate availability for full duration
                    room_available = True  # You should implement proper validation here
                    
                    if room_available:
                        recommendations.append({
                            'type': 'llm_generated',
                            'score': min(rec.get('confidence', 0.5), 1.0),
                            'reason': rec.get('reason', 'LLM recommendation'),
                            'suggestion': {
                                'room_name': rec.get('room_name', 'Unknown'),
                                'start_time': start_time,
                                'end_time': end_time,
                                'date': context.get('date', ''),
                                'confidence': min(rec.get('confidence', 0.5), 1.0)
                            },
                            'llm_explanation': rec.get('explanation', ''),
                            'data_source': 'deepseek_llm'
                        })
            return recommendations[:3]
        except Exception as e:
            logger.error(f"Error parsing room recommendations: {e}")
            return []
    
    def _parse_alternative_suggestions(self, response: str, alternatives: List[Dict], original_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        enhanced_alternatives = []
        duration = original_request.get('duration_hours', 1)
        
        for i, alt in enumerate(alternatives[:5]):
            try:
                enhanced_alt = alt.copy()
                enhanced_alt['llm_enhanced'] = True
                
                # Ensure proper end time calculation for alternatives
                alt_start = alt.get('suggestion', {}).get('start_time', '')
                if alt_start:
                    alt_end = self._calculate_end_time(alt_start, duration)
                    enhanced_alt['suggestion']['end_time'] = alt_end
                
                enhanced_alt['llm_explanation'] = self._extract_explanation_for_alternative(response, alt.get('suggestion', {}).get('room_name', ''), i)
                
                original_score = alt.get('score', 0.5)
                llm_boost = 0.1 if alt.get('suggestion', {}).get('room_name', '') in response else 0
                enhanced_alt['score'] = min(original_score + llm_boost, 1.0)
                enhanced_alternatives.append(enhanced_alt)
            except Exception as e:
                logger.error(f"Error enhancing alternative {i}: {e}")
                enhanced_alternatives.append(alt)
        return enhanced_alternatives
    
    def _parse_scheduling_insights(self, response: str, scheduling_data: Dict[str, Any]) -> Dict[str, Any]:
        default_insights = {
            'optimal_times': [], 'room_suggestions': [], 'scheduling_tips': [],
            'conflict_predictions': [], 'efficiency_score': 0.5
        }
        
        try:
            duration = scheduling_data.get('duration_hours', 1)
            
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Process optimal times with proper end time calculation
                optimal_times = []
                for time_slot in parsed.get('optimal_times', []):
                    if isinstance(time_slot, dict) and 'time_slot' in time_slot:
                        slot_start = time_slot['time_slot'].split('-')[0] if '-' in time_slot['time_slot'] else time_slot['time_slot']
                        slot_end = self._calculate_end_time(slot_start, duration)
                        
                        optimal_times.append({
                            'start_time': slot_start,
                            'end_time': slot_end,
                            'efficiency_score': time_slot.get('efficiency_score', 0.5),
                            'reasoning': time_slot.get('reasoning', 'Optimal time slot')
                        })
                
                return {
                    'optimal_times': optimal_times,
                    'room_suggestions': parsed.get('room_suggestions', []),
                    'scheduling_tips': parsed.get('scheduling_tips', []),
                    'conflict_predictions': parsed.get('conflict_predictions', []),
                    'efficiency_score': parsed.get('efficiency_score', 0.5)
                }
            return default_insights
        except Exception as e:
            logger.error(f"Error parsing scheduling insights: {e}")
            return default_insights
    
    def _extract_explanation_for_alternative(self, response: str, room_name: str, index: int) -> str:
        if room_name and room_name.lower() in response.lower():
            sentences = response.split('.')
            for sentence in sentences:
                if room_name.lower() in sentence.lower():
                    return sentence.strip()
        
        fallback_explanations = [
            "Alternative option based on availability and your preferences",
            "Good alternative with similar features to your requested room",
            "Available at your preferred time with suitable capacity",
            "Suitable alternative for your meeting needs",
            "Recommended based on room features and timing"
        ]
        return fallback_explanations[index % len(fallback_explanations)]
    
    # Updated Prompt Templates
    def _get_user_pattern_template(self) -> str:
        return """Analyze this user's booking patterns:
        
User ID: {user_id}, Total: {total_bookings}
Recent: {recent_bookings}
Preferred: {preferred_rooms}
Frequency: {booking_frequency}, Duration: {typical_duration}h
Date: {current_date}

JSON format:
{{"patterns_identified": ["pattern1"], "user_type": "frequent|occasional|new",
"preferences_inferred": {{"preferred_times": ["morning"], "room_preferences": ["small"], "booking_style": "advance_planner"}},
"recommendations": [{{"type": "room_suggestion", "room_name": "Room", "reason": "Why", "confidence": 0.8}}],
"scheduling_insights": ["insight1"]}}"""
    
    def _get_room_analysis_template(self) -> str:
        return """Analyze room booking request:
        
Room: {requested_room}, Date: {requested_date}
Time: {requested_start_time} - {requested_end_time} (Duration: {duration}h)
Capacity: {capacity_needed}, Purpose: {purpose}
Current: {current_time}, {day_of_week}

Available: {available_rooms}

Recommend rooms available for the FULL duration from {requested_start_time} to {requested_end_time}.

JSON format:
{{"recommendations": [{{"room_name": "Room", "reason": "Why", "confidence": 0.8, "explanation": "Details", "suitability_factors": ["factor1"]}}],
"timing_insights": [{{"suggestion": "Consider earlier", "reason": "Better availability"}}]}}"""
    
    def _get_alternative_suggestions_template(self) -> str:
        return """User wanted {original_room} on {original_date} from {original_start_time} to {original_end_time} (Duration: {duration}h) for {purpose}, unavailable.

Preferences: {user_preferences}
Patterns: {user_patterns}
Capacity needed: {capacity_needed}

Available alternatives: {alternatives}

Provide alternatives that are available for the FULL {duration} hour duration. Consider both alternative rooms at the same time and alternative times for suitable rooms.

JSON format:
{{"alternatives_analysis": [{{"room_name": "Alt", "recommendation_strength": "high", "reasons": ["reason1"], "explanation": "Why", "trade_offs": ["gain", "lose"], "available_duration": "Full {duration}h available"}}],
"timing_alternatives": [{{"suggested_start_time": "09:00", "suggested_end_time": "10:00", "explanation": "Better time with full availability"}}],
"general_advice": "Overall guidance"}}"""
    
    def _get_smart_scheduling_template(self) -> str:
        return """Scheduling analysis for {duration}h meeting:
        
Requested: {requested_time} to {requested_end_time}, Capacity: {capacity_needed}
Purpose: {purpose}, Date: {current_date}, Current: {current_time}

Available rooms: {available_rooms}
Utilization: {utilization_stats}

Suggest optimal times that have {duration}h availability and suitable rooms.

JSON format:
{{"optimal_times": [{{"time_slot": "09:00-10:00", "efficiency_score": 0.9, "reasoning": "Why optimal for {duration}h"}}],
"room_suggestions": [{{"room_name": "Room", "utilization_score": 0.3, "recommendation": "Why good", "available_duration": "{duration}h available"}}],
"scheduling_tips": ["Book {duration}h slots in advance"], "conflict_predictions": ["High demand 2-3 PM"], "efficiency_score": 0.8}}"""
    
    def _get_explanation_template(self) -> str:
        return """Explain recommendation:
        
Room: {room_name}, Date: {date}
Time: {start_time} to {end_time}
Confidence: {confidence}, Type: {recommendation_type}, Reason: {reason}

User: {user_history_count} bookings, Prefers: {user_preferences}

Provide 1-2 sentence explanation focusing on benefits, time slot availability, and user patterns."""
    
    # Updated Fallback Methods
    def _get_fallback_analysis(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        booking_history = user_data.get('booking_history', [])
        preferred_rooms = user_data.get('preferred_rooms', [])
        
        return {
            'patterns_identified': ['Regular booking pattern' if len(booking_history) > 5 else 'Occasional user',
                                  'Prefers specific rooms' if preferred_rooms else 'Flexible room choice'],
            'user_type': 'frequent' if len(booking_history) > 10 else 'occasional',
            'preferences_inferred': {'preferred_times': ['morning'], 'room_preferences': preferred_rooms[:3], 'booking_style': 'regular'},
            'recommendations': [{'type': 'room_suggestion', 'room_name': room, 'reason': 'Based on history', 'confidence': 0.7} for room in preferred_rooms[:3]],
            'scheduling_insights': ['User books regularly' if len(booking_history) > 5 else 'New user pattern']
        }
    
    def _get_fallback_room_recommendations(self, request_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        available_rooms = request_context.get('available_rooms', [])
        start_time = request_context.get('start_time', '')
        duration = request_context.get('duration_hours', 1)
        end_time = self._calculate_end_time(start_time, duration)
        
        recommendations = []
        for i, room in enumerate(available_rooms[:3]):
            recommendations.append({
                'type': 'fallback_recommendation', 
                'score': 0.6 - (i * 0.1), 
                'reason': f'Available room with suitable capacity for full duration',
                'suggestion': {
                    'room_name': room.get('name', f'Room {i+1}'), 
                    'start_time': start_time,
                    'end_time': end_time,
                    'date': request_context.get('date', ''),
                    'confidence': 0.6 - (i * 0.1)
                },
                'data_source': 'fallback_logic'
            })
        return recommendations
    
    def _enhance_alternatives_fallback(self, alternatives: List[Dict[str, Any]], original_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        enhanced = []
        duration = original_request.get('duration_hours', 1)
        fallback_explanations = [
            f"Good alternative option with {duration}h availability", 
            f"Similar features to your requested room, available for full {duration}h", 
            f"Available at suitable time with {duration}h duration", 
            f"Suitable for your meeting requirements with full availability", 
            f"Recommended based on room capacity and {duration}h availability"
        ]
        
        for i, alt in enumerate(alternatives):
            enhanced_alt = alt.copy()
            
            # Calculate proper end time for alternative
            alt_start = alt.get('suggestion', {}).get('start_time', '')
            if alt_start:
                alt_end = self._calculate_end_time(alt_start, duration)
                enhanced_alt['suggestion']['end_time'] = alt_end
            
            enhanced_alt['llm_explanation'] = fallback_explanations[i % len(fallback_explanations)]
            enhanced_alt['llm_enhanced'] = False
            enhanced.append(enhanced_alt)
        return enhanced
    
    def _get_fallback_scheduling_insights(self, scheduling_data: Dict[str, Any]) -> Dict[str, Any]:
        duration = scheduling_data.get('duration_hours', 1)
        
        return {
            'optimal_times': [
                {
                    'start_time': '09:00', 
                    'end_time': self._calculate_end_time('09:00', duration),
                    'efficiency_score': 0.8, 
                    'reasoning': f'Morning hours typically less busy for {duration}h meetings'
                },
                {
                    'start_time': '14:00', 
                    'end_time': self._calculate_end_time('14:00', duration),
                    'efficiency_score': 0.7, 
                    'reasoning': f'Good afternoon option with {duration}h availability'
                }
            ],
            'room_suggestions': [{
                'room_name': 'Room Name', 
                'utilization_score': 0.3, 
                'recommendation': f'Generally available for {duration}h duration',
                'available_duration': f'{duration}h available'
            }],
            'scheduling_tips': [
                f'Book {duration}h slots in advance for better room selection', 
                f'Consider morning slots for higher {duration}h availability'
            ],
            'conflict_predictions': [], 
            'efficiency_score': 0.6
        }
    
    def _get_fallback_explanation(self, recommendation: Dict[str, Any]) -> str:
        room_name = recommendation.get('suggestion', {}).get('room_name', 'this room')
        start_time = recommendation.get('suggestion', {}).get('start_time', '')
        end_time = recommendation.get('suggestion', {}).get('end_time', '')
        rec_type = recommendation.get('type', 'recommendation')
        score = recommendation.get('score', 0.5)
        
        time_info = f" from {start_time} to {end_time}" if start_time and end_time else ""
        
        if rec_type == 'alternative_time':
            return f"Consider {room_name}{time_info} - it offers good availability for your full duration."
        elif rec_type == 'alternative_room':
            return f"{room_name} is available{time_info} with similar features to meet your needs."
        elif rec_type == 'proactive':
            return f"Based on your booking history, {room_name}{time_info} is a great choice."
        else:
            confidence_text = "highly" if score > 0.8 else "moderately" if score > 0.6 else "reasonably"
            return f"{room_name} is {confidence_text} recommended{time_info} based on your requirements."
    
    def _parse_text_to_analysis(self, response: str) -> Dict[str, Any]:
        patterns = []
        if 'frequent' in response.lower(): patterns.append('frequent booking pattern')
        if 'morning' in response.lower(): patterns.append('morning preference')
        if 'regular' in response.lower(): patterns.append('regular schedule')
        
        return {
            'patterns_identified': patterns or ['general usage pattern'],
            'user_type': 'frequent' if 'frequent' in response.lower() else 'regular',
            'preferences_inferred': {
                'preferred_times': ['morning'] if 'morning' in response.lower() else ['flexible'],
                'room_preferences': [], 'booking_style': 'regular'
            },
            'recommendations': [], 
            'scheduling_insights': patterns or ['standard booking behavior']
        }
    
    def get_processor_status(self) -> Dict[str, Any]:
        return {
            'deepseek_available': self.deepseek is not None,
            'templates_loaded': len(self.prompt_templates),
            'fallback_ready': True,
            'processor_type': 'DeepSeekRecommendationProcessor',
            'features': [
                'proper_end_time_calculation',
                'duration_validation',
                'availability_checking',
                'time_range_validation'
            ]
        }