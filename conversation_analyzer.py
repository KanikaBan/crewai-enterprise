from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Optional
import json
from collections import Counter, defaultdict
import re

class ConversationAnalysisRequest(BaseModel):
    """Input schema for Conversation Analyzer Tool."""
    user_id: str = Field(description="The ID of the user to analyze conversations for")
    analysis_type: str = Field(
        description="Type of analysis to perform: 'user_patterns', 'project_trends', 'success_analysis', 'similar_ideas'"
    )
    conversations_data: Optional[str] = Field(
        default="[]",
        description="JSON string containing conversation data to analyze. If not provided, will use mock data for demonstration."
    )

class ConversationAnalyzerTool(BaseTool):
    """Tool for analyzing patterns in stored conversations to provide insights."""

    name: str = "conversation_analyzer"
    description: str = (
        "Analyzes patterns in stored conversations to provide insights about user behavior, "
        "project trends, success patterns, and similar ideas. Supports analysis types: "
        "'user_patterns', 'project_trends', 'success_analysis', 'similar_ideas'. "
        "Returns structured analysis with insights and recommendations."
    )
    args_schema: Type[BaseModel] = ConversationAnalysisRequest

    def _run(self, user_id: str, analysis_type: str, conversations_data: str = "[]") -> str:
        """
        Analyze conversation patterns based on the specified analysis type.
        
        Args:
            user_id: The user ID to analyze
            analysis_type: Type of analysis to perform
            conversations_data: JSON string of conversation data
        
        Returns:
            JSON string with analysis results and insights
        """
        try:
            # Validate analysis type
            valid_types = ['user_patterns', 'project_trends', 'success_analysis', 'similar_ideas']
            if analysis_type not in valid_types:
                return json.dumps({
                    "error": f"Invalid analysis_type. Must be one of: {valid_types}",
                    "status": "failed"
                })

            # Parse conversations data
            try:
                conversations = json.loads(conversations_data)
                if not conversations:
                    # Use mock data for demonstration
                    conversations = self._get_mock_conversations()
            except json.JSONDecodeError:
                return json.dumps({
                    "error": "Invalid JSON format for conversations_data",
                    "status": "failed"
                })

            # Perform analysis based on type
            if analysis_type == 'user_patterns':
                result = self._analyze_user_patterns(user_id, conversations)
            elif analysis_type == 'project_trends':
                result = self._analyze_project_trends(conversations)
            elif analysis_type == 'success_analysis':
                result = self._analyze_success_patterns(conversations)
            elif analysis_type == 'similar_ideas':
                result = self._find_similar_ideas(conversations)

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Analysis failed: {str(e)}",
                "status": "failed"
            })

    def _get_mock_conversations(self) -> List[Dict]:
        """Generate mock conversation data for demonstration."""
        return [
            {
                "id": "conv_001",
                "user_id": "user_123",
                "timestamp": "2024-01-15T10:30:00",
                "content": "I want to create a mobile app for food delivery. The app should connect restaurants with customers.",
                "project_type": "mobile_app",
                "status": "approved",
                "tags": ["food", "delivery", "mobile", "marketplace"]
            },
            {
                "id": "conv_002", 
                "user_id": "user_456",
                "timestamp": "2024-01-16T14:20:00",
                "content": "Let's build a web platform for online learning. It should have video courses and quizzes.",
                "project_type": "web_app",
                "status": "implemented",
                "tags": ["education", "learning", "web", "platform"]
            },
            {
                "id": "conv_003",
                "user_id": "user_123", 
                "timestamp": "2024-01-17T09:15:00",
                "content": "I need a data analytics dashboard for tracking sales performance across different regions.",
                "project_type": "dashboard",
                "status": "rejected",
                "tags": ["analytics", "sales", "dashboard", "reporting"]
            },
            {
                "id": "conv_004",
                "user_id": "user_789",
                "timestamp": "2024-01-18T16:45:00", 
                "content": "Can we develop a food ordering system for restaurants? Customers should be able to browse menus.",
                "project_type": "mobile_app",
                "status": "approved",
                "tags": ["food", "ordering", "restaurant", "mobile"]
            }
        ]

    def _analyze_user_patterns(self, user_id: str, conversations: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns for a specific user."""
        user_conversations = [conv for conv in conversations if conv.get('user_id') == user_id]
        
        if not user_conversations:
            return {
                "analysis_type": "user_patterns",
                "user_id": user_id,
                "status": "insufficient_data",
                "message": "No conversations found for this user",
                "insights": [],
                "recommendations": ["Start more conversations to build analysis data"]
            }

        # Analyze patterns
        project_types = Counter(conv.get('project_type', 'unknown') for conv in user_conversations)
        status_distribution = Counter(conv.get('status', 'unknown') for conv in user_conversations)
        
        # Extract keywords from content
        all_content = ' '.join(conv.get('content', '') for conv in user_conversations)
        keywords = self._extract_keywords(all_content)
        
        # Calculate success rate
        total_projects = len(user_conversations)
        successful_projects = len([conv for conv in user_conversations 
                                 if conv.get('status') in ['approved', 'implemented']])
        success_rate = (successful_projects / total_projects * 100) if total_projects > 0 else 0

        insights = [
            f"User has initiated {total_projects} conversations",
            f"Most preferred project type: {project_types.most_common(1)[0][0] if project_types else 'N/A'}",
            f"Success rate: {success_rate:.1f}%",
            f"Top keywords: {', '.join([kw[0] for kw in keywords[:5]])}"
        ]

        recommendations = []
        if success_rate < 50:
            recommendations.append("Consider refining project ideas for better approval rates")
        if len(set(conv.get('project_type') for conv in user_conversations)) == 1:
            recommendations.append("Try exploring different project types for diverse experience")

        return {
            "analysis_type": "user_patterns",
            "user_id": user_id,
            "status": "success",
            "data": {
                "total_conversations": total_projects,
                "project_types": dict(project_types),
                "status_distribution": dict(status_distribution),
                "success_rate": success_rate,
                "top_keywords": keywords[:10]
            },
            "insights": insights,
            "recommendations": recommendations
        }

    def _analyze_project_trends(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Analyze trending topics and themes across all conversations."""
        if len(conversations) < 2:
            return {
                "analysis_type": "project_trends",
                "status": "insufficient_data",
                "message": "Need at least 2 conversations for trend analysis",
                "insights": [],
                "recommendations": ["Collect more conversation data"]
            }

        # Analyze project types
        project_types = Counter(conv.get('project_type', 'unknown') for conv in conversations)
        
        # Extract and count all tags
        all_tags = []
        for conv in conversations:
            all_tags.extend(conv.get('tags', []))
        trending_tags = Counter(all_tags)
        
        # Analyze content keywords
        all_content = ' '.join(conv.get('content', '') for conv in conversations)
        keywords = self._extract_keywords(all_content)
        
        # Time-based analysis (mock implementation)
        monthly_counts = defaultdict(int)
        for conv in conversations:
            timestamp = conv.get('timestamp', '2024-01-01T00:00:00')
            month = timestamp[:7]  # Extract YYYY-MM
            monthly_counts[month] += 1

        insights = [
            f"Total conversations analyzed: {len(conversations)}",
            f"Most popular project type: {project_types.most_common(1)[0][0] if project_types else 'N/A'}",
            f"Top trending tag: {trending_tags.most_common(1)[0][0] if trending_tags else 'N/A'}",
            f"Most active month: {max(monthly_counts, key=monthly_counts.get) if monthly_counts else 'N/A'}"
        ]

        recommendations = [
            f"Focus on {project_types.most_common(1)[0][0]} projects as they're most popular",
            f"Consider projects involving {trending_tags.most_common(1)[0][0]} as it's trending"
        ]

        return {
            "analysis_type": "project_trends",
            "status": "success",
            "data": {
                "project_types": dict(project_types.most_common()),
                "trending_tags": dict(trending_tags.most_common(10)),
                "top_keywords": keywords[:15],
                "monthly_activity": dict(monthly_counts)
            },
            "insights": insights,
            "recommendations": recommendations
        }

    def _analyze_success_patterns(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in successful vs unsuccessful projects."""
        if len(conversations) < 3:
            return {
                "analysis_type": "success_analysis",
                "status": "insufficient_data",
                "message": "Need at least 3 conversations for success pattern analysis",
                "insights": [],
                "recommendations": ["Collect more conversation data with status information"]
            }

        # Categorize conversations by success
        successful = [conv for conv in conversations if conv.get('status') in ['approved', 'implemented']]
        unsuccessful = [conv for conv in conversations if conv.get('status') in ['rejected', 'cancelled']]
        
        if not successful or not unsuccessful:
            return {
                "analysis_type": "success_analysis",
                "status": "insufficient_data",
                "message": "Need both successful and unsuccessful examples",
                "insights": [],
                "recommendations": ["Ensure conversations have varied status outcomes"]
            }

        # Analyze successful patterns
        successful_types = Counter(conv.get('project_type') for conv in successful)
        successful_tags = []
        for conv in successful:
            successful_tags.extend(conv.get('tags', []))
        successful_tag_counts = Counter(successful_tags)
        
        # Analyze unsuccessful patterns
        unsuccessful_types = Counter(conv.get('project_type') for conv in unsuccessful)
        unsuccessful_tags = []
        for conv in unsuccessful:
            unsuccessful_tags.extend(conv.get('tags', []))
        unsuccessful_tag_counts = Counter(unsuccessful_tags)
        
        # Calculate success rates by project type
        success_rates = {}
        for proj_type in set(conv.get('project_type') for conv in conversations):
            total = len([c for c in conversations if c.get('project_type') == proj_type])
            success = len([c for c in successful if c.get('project_type') == proj_type])
            success_rates[proj_type] = (success / total * 100) if total > 0 else 0

        insights = [
            f"Overall success rate: {len(successful) / len(conversations) * 100:.1f}%",
            f"Most successful project type: {max(success_rates, key=success_rates.get) if success_rates else 'N/A'}",
            f"Most common tag in successful projects: {successful_tag_counts.most_common(1)[0][0] if successful_tag_counts else 'N/A'}",
            f"Least successful project type: {min(success_rates, key=success_rates.get) if success_rates else 'N/A'}"
        ]

        recommendations = []
        best_type = max(success_rates, key=success_rates.get) if success_rates else None
        if best_type:
            recommendations.append(f"Focus on {best_type} projects for higher success rates")
        
        if successful_tag_counts:
            top_success_tag = successful_tag_counts.most_common(1)[0][0]
            recommendations.append(f"Include '{top_success_tag}' elements in projects")

        return {
            "analysis_type": "success_analysis",
            "status": "success",
            "data": {
                "total_conversations": len(conversations),
                "successful_count": len(successful),
                "unsuccessful_count": len(unsuccessful),
                "success_rates_by_type": success_rates,
                "successful_tags": dict(successful_tag_counts.most_common(5)),
                "unsuccessful_tags": dict(unsuccessful_tag_counts.most_common(5))
            },
            "insights": insights,
            "recommendations": recommendations
        }

    def _find_similar_ideas(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Find conversations with similar project ideas."""
        if len(conversations) < 2:
            return {
                "analysis_type": "similar_ideas",
                "status": "insufficient_data",
                "message": "Need at least 2 conversations to find similarities",
                "insights": [],
                "recommendations": ["Add more conversations to detect similarities"]
            }

        similar_groups = []
        processed = set()
        
        for i, conv1 in enumerate(conversations):
            if i in processed:
                continue
                
            similar_group = [conv1]
            tags1 = set(conv1.get('tags', []))
            content1 = conv1.get('content', '').lower()
            
            for j, conv2 in enumerate(conversations[i+1:], i+1):
                if j in processed:
                    continue
                    
                tags2 = set(conv2.get('tags', []))
                content2 = conv2.get('content', '').lower()
                
                # Calculate similarity
                tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2)) if tags1.union(tags2) else 0
                content_similarity = self._calculate_content_similarity(content1, content2)
                
                # Consider similar if high tag overlap or content similarity
                if tag_similarity > 0.3 or content_similarity > 0.4:
                    similar_group.append(conv2)
                    processed.add(j)
            
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
                processed.add(i)

        insights = [
            f"Found {len(similar_groups)} groups of similar ideas",
            f"Total conversations analyzed: {len(conversations)}",
            f"Unique ideas: {len(conversations) - sum(len(group)-1 for group in similar_groups)}"
        ]

        recommendations = []
        if similar_groups:
            recommendations.append("Review similar ideas before starting new projects to avoid duplication")
            recommendations.append("Consider combining similar ideas into more comprehensive projects")
        else:
            recommendations.append("All ideas appear to be unique - good diversity in project concepts")

        # Format similar groups for output
        formatted_groups = []
        for i, group in enumerate(similar_groups):
            formatted_groups.append({
                "group_id": i + 1,
                "count": len(group),
                "conversations": [
                    {
                        "id": conv.get('id'),
                        "user_id": conv.get('user_id'),
                        "project_type": conv.get('project_type'),
                        "tags": conv.get('tags', []),
                        "content_preview": conv.get('content', '')[:100] + "..." if len(conv.get('content', '')) > 100 else conv.get('content', '')
                    } for conv in group
                ]
            })

        return {
            "analysis_type": "similar_ideas",
            "status": "success",
            "data": {
                "total_conversations": len(conversations),
                "similar_groups": formatted_groups,
                "unique_ideas": len(conversations) - sum(len(group)-1 for group in similar_groups)
            },
            "insights": insights,
            "recommendations": recommendations
        }

    def _extract_keywords(self, text: str, top_n: int = 20) -> List[tuple]:
        """Extract keywords from text using basic frequency analysis."""
        # Clean text and split into words
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cant', 'wont', 'dont',
            'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        return Counter(filtered_words).most_common(top_n)

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings using basic word overlap."""
        words1 = set(re.sub(r'[^a-zA-Z\s]', '', content1.lower()).split())
        words2 = set(re.sub(r'[^a-zA-Z\s]', '', content2.lower()).split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0