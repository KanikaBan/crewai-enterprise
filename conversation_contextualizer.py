from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Optional
import json
import re
from collections import Counter
import math

class ConversationContextualizerInput(BaseModel):
    """Input schema for Conversation Contextualizer Tool."""
    current_project_idea: str = Field(..., description="The current project idea to find context for")
    user_id: str = Field(..., description="The user ID to filter relevant conversations")
    context_type: str = Field(..., description="Type of context: 'similar_projects', 'user_history', 'lessons_learned', 'related_discussions'")
    conversation_data: Optional[str] = Field(default="[]", description="JSON string of conversation data (optional, defaults to empty)")
    max_results: Optional[int] = Field(default=5, description="Maximum number of results to return")

class ConversationContextualizerTool(BaseTool):
    """Tool for extracting relevant context from past conversations using keyword matching and similarity scoring."""

    name: str = "conversation_contextualizer"
    description: str = (
        "Analyzes conversation data to provide relevant context for current project discussions. "
        "Supports finding similar projects, user history, lessons learned, and related discussions "
        "with confidence scores and relevance rankings."
    )
    args_schema: Type[BaseModel] = ConversationContextualizerInput

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using basic NLP techniques."""
        try:
            # Convert to lowercase and remove special characters
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
            
            # Common stop words to filter out
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
            }
            
            # Extract words and filter stop words
            words = [word for word in cleaned_text.split() if len(word) > 2 and word not in stop_words]
            
            # Return most common words as keywords
            word_counts = Counter(words)
            return [word for word, count in word_counts.most_common(20)]
            
        except Exception as e:
            return []

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic similarity score between two texts."""
        try:
            keywords1 = set(self._extract_keywords(text1))
            keywords2 = set(self._extract_keywords(text2))
            
            if not keywords1 or not keywords2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Cosine similarity based on keyword frequency
            all_keywords = keywords1.union(keywords2)
            vector1 = [1 if keyword in keywords1 else 0 for keyword in all_keywords]
            vector2 = [1 if keyword in keywords2 else 0 for keyword in all_keywords]
            
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(a * a for a in vector2))
            
            cosine_similarity = 0.0
            if magnitude1 > 0 and magnitude2 > 0:
                cosine_similarity = dot_product / (magnitude1 * magnitude2)
            
            # Combine similarities with weights
            final_similarity = (jaccard_similarity * 0.6) + (cosine_similarity * 0.4)
            return min(max(final_similarity, 0.0), 1.0)
            
        except Exception as e:
            return 0.0

    def _analyze_similar_projects(self, current_idea: str, conversations: List[Dict], user_id: str) -> List[Dict]:
        """Find conversations about similar project ideas."""
        results = []
        
        for conv in conversations:
            try:
                content = conv.get('content', '')
                conv_user_id = conv.get('user_id', '')
                timestamp = conv.get('timestamp', '')
                
                similarity = self._calculate_similarity(current_idea, content)
                
                if similarity > 0.1:  # Minimum threshold
                    results.append({
                        'conversation_id': conv.get('id', ''),
                        'content': content[:200] + '...' if len(content) > 200 else content,
                        'similarity_score': round(similarity, 3),
                        'confidence': round(similarity * 0.9, 3),
                        'timestamp': timestamp,
                        'user_id': conv_user_id,
                        'relevance_type': 'similar_project'
                    })
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    def _analyze_user_history(self, current_idea: str, conversations: List[Dict], user_id: str) -> List[Dict]:
        """Provide user's relevant past innovation discussions."""
        results = []
        
        # Filter conversations by user_id
        user_conversations = [conv for conv in conversations if conv.get('user_id') == user_id]
        
        for conv in user_conversations:
            try:
                content = conv.get('content', '')
                timestamp = conv.get('timestamp', '')
                
                similarity = self._calculate_similarity(current_idea, content)
                
                if similarity > 0.05:  # Lower threshold for user's own conversations
                    results.append({
                        'conversation_id': conv.get('id', ''),
                        'content': content[:200] + '...' if len(content) > 200 else content,
                        'similarity_score': round(similarity, 3),
                        'confidence': round(similarity * 0.95, 3),  # Higher confidence for user's own data
                        'timestamp': timestamp,
                        'user_id': user_id,
                        'relevance_type': 'user_history'
                    })
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

    def _extract_lessons_learned(self, current_idea: str, conversations: List[Dict], user_id: str) -> List[Dict]:
        """Extract insights and lessons from previous similar projects."""
        results = []
        
        # Look for conversations containing lesson indicators
        lesson_keywords = ['lesson', 'learned', 'mistake', 'error', 'problem', 'solution', 'insight', 'experience', 'avoid', 'recommend']
        
        for conv in conversations:
            try:
                content = conv.get('content', '').lower()
                timestamp = conv.get('timestamp', '')
                conv_user_id = conv.get('user_id', '')
                
                # Check for lesson indicators
                lesson_score = sum(1 for keyword in lesson_keywords if keyword in content) / len(lesson_keywords)
                
                # Calculate similarity to current idea
                similarity = self._calculate_similarity(current_idea, conv.get('content', ''))
                
                # Combine scores
                combined_score = (lesson_score * 0.4) + (similarity * 0.6)
                
                if combined_score > 0.1:
                    results.append({
                        'conversation_id': conv.get('id', ''),
                        'content': conv.get('content', '')[:200] + '...' if len(conv.get('content', '')) > 200 else conv.get('content', ''),
                        'similarity_score': round(similarity, 3),
                        'lesson_score': round(lesson_score, 3),
                        'confidence': round(combined_score * 0.8, 3),
                        'timestamp': timestamp,
                        'user_id': conv_user_id,
                        'relevance_type': 'lessons_learned'
                    })
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _find_related_discussions(self, current_idea: str, conversations: List[Dict], user_id: str) -> List[Dict]:
        """Find tangentially related conversations that might provide insights."""
        results = []
        
        current_keywords = set(self._extract_keywords(current_idea))
        
        for conv in conversations:
            try:
                content = conv.get('content', '')
                timestamp = conv.get('timestamp', '')
                conv_user_id = conv.get('user_id', '')
                
                conv_keywords = set(self._extract_keywords(content))
                
                # Look for partial keyword overlap (tangential relationship)
                overlap = len(current_keywords.intersection(conv_keywords))
                relatedness = overlap / max(len(current_keywords), 1) if current_keywords else 0
                
                # Avoid exact matches (those are similar_projects)
                similarity = self._calculate_similarity(current_idea, content)
                
                if 0.05 < relatedness < 0.8 and similarity < 0.7:
                    results.append({
                        'conversation_id': conv.get('id', ''),
                        'content': content[:200] + '...' if len(content) > 200 else content,
                        'similarity_score': round(similarity, 3),
                        'relatedness_score': round(relatedness, 3),
                        'confidence': round(relatedness * 0.7, 3),
                        'timestamp': timestamp,
                        'user_id': conv_user_id,
                        'relevance_type': 'related_discussion'
                    })
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _run(self, current_project_idea: str, user_id: str, context_type: str, 
             conversation_data: Optional[str] = "[]", max_results: Optional[int] = 5) -> str:
        """Execute the conversation contextualization."""
        try:
            # Parse conversation data
            try:
                conversations = json.loads(conversation_data) if conversation_data else []
            except json.JSONDecodeError:
                conversations = []
            
            # Validate context_type
            valid_types = ['similar_projects', 'user_history', 'lessons_learned', 'related_discussions']
            if context_type not in valid_types:
                return json.dumps({
                    'error': f'Invalid context_type. Must be one of: {", ".join(valid_types)}',
                    'valid_types': valid_types
                })
            
            # Execute appropriate analysis
            if context_type == 'similar_projects':
                results = self._analyze_similar_projects(current_project_idea, conversations, user_id)
            elif context_type == 'user_history':
                results = self._analyze_user_history(current_project_idea, conversations, user_id)
            elif context_type == 'lessons_learned':
                results = self._extract_lessons_learned(current_project_idea, conversations, user_id)
            elif context_type == 'related_discussions':
                results = self._find_related_discussions(current_project_idea, conversations, user_id)
            
            # Limit results
            limited_results = results[:max_results]
            
            # Prepare response
            response = {
                'context_type': context_type,
                'current_project_idea': current_project_idea,
                'user_id': user_id,
                'total_found': len(results),
                'returned_count': len(limited_results),
                'contexts': limited_results,
                'summary': {
                    'highest_confidence': max([r.get('confidence', 0) for r in limited_results], default=0),
                    'average_confidence': round(sum([r.get('confidence', 0) for r in limited_results]) / len(limited_results), 3) if limited_results else 0,
                    'keywords_analyzed': self._extract_keywords(current_project_idea)[:10]
                }
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({
                'error': f'Error processing conversation context: {str(e)}',
                'context_type': context_type,
                'user_id': user_id
            })