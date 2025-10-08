from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Optional
from datetime import datetime
import json

class ConversationMemoryInput(BaseModel):
    """Input schema for ConversationMemoryManager Tool."""
    action: str = Field(..., description="Action to perform: 'store' or 'retrieve'")
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    user_id: str = Field(..., description="Identifier for the user")
    project_idea: str = Field(..., description="The project idea or topic of the conversation")
    conversation_data: str = Field(..., description="The actual conversation content/data")
    search_query: Optional[str] = Field(None, description="Optional search query for retrieve action")

class ConversationMemoryManagerTool(BaseTool):
    """Tool for managing conversation data in memory during the current session."""

    name: str = "conversation_memory_manager"
    description: str = (
        "Manages conversation data in memory for the current session. "
        "Can store conversations with structured format including timestamps, "
        "and retrieve conversations using search functionality by user_id, "
        "project_idea keywords, or conversation_id."
    )
    args_schema: Type[BaseModel] = ConversationMemoryInput
    
    # Class-level dictionary to store conversations across instances
    _conversations: Dict[str, Dict[str, Any]] = {}

    def _run(
        self, 
        action: str, 
        conversation_id: str, 
        user_id: str, 
        project_idea: str, 
        conversation_data: str, 
        search_query: Optional[str] = None
    ) -> str:
        try:
            if action.lower() == 'store':
                return self._store_conversation(
                    conversation_id, user_id, project_idea, conversation_data
                )
            elif action.lower() == 'retrieve':
                return self._retrieve_conversations(
                    conversation_id, user_id, project_idea, search_query
                )
            else:
                return f"Error: Invalid action '{action}'. Use 'store' or 'retrieve'."
                
        except Exception as e:
            return f"Error processing conversation memory request: {str(e)}"

    def _store_conversation(
        self, 
        conversation_id: str, 
        user_id: str, 
        project_idea: str, 
        conversation_data: str
    ) -> str:
        """Store a conversation in memory with structured format."""
        try:
            # Create structured conversation data
            conversation_entry = {
                'conversation_id': conversation_id,
                'user_id': user_id,
                'project_idea': project_idea,
                'conversation_data': conversation_data,
                'timestamp': datetime.now().isoformat(),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Store in class-level dictionary
            self._conversations[conversation_id] = conversation_entry
            
            return f"Successfully stored conversation '{conversation_id}' for user '{user_id}' " \
                   f"about project: '{project_idea}' at {conversation_entry['created_at']}"
                   
        except Exception as e:
            return f"Error storing conversation: {str(e)}"

    def _retrieve_conversations(
        self, 
        conversation_id: str, 
        user_id: str, 
        project_idea: str, 
        search_query: Optional[str] = None
    ) -> str:
        """Retrieve conversations using search functionality."""
        try:
            if not self._conversations:
                return "No conversations found in memory."
            
            matching_conversations = []
            
            # Search through all stored conversations
            for conv_id, conversation in self._conversations.items():
                match_found = False
                
                # Exact conversation_id match has highest priority
                if conv_id == conversation_id:
                    matching_conversations.insert(0, conversation)
                    continue
                
                # Search by user_id
                if conversation['user_id'].lower() == user_id.lower():
                    match_found = True
                
                # Search by project_idea keywords
                if project_idea.lower() in conversation['project_idea'].lower() or \
                   conversation['project_idea'].lower() in project_idea.lower():
                    match_found = True
                
                # Search by optional search_query
                if search_query:
                    search_terms = search_query.lower().split()
                    for term in search_terms:
                        if (term in conversation['conversation_data'].lower() or
                            term in conversation['project_idea'].lower() or
                            term in conversation['user_id'].lower()):
                            match_found = True
                            break
                
                if match_found:
                    matching_conversations.append(conversation)
            
            if not matching_conversations:
                return f"No conversations found matching the criteria. " \
                       f"Searched for: conversation_id='{conversation_id}', user_id='{user_id}', " \
                       f"project_idea='{project_idea}'" + \
                       (f", search_query='{search_query}'" if search_query else "")
            
            # Format the results
            result = f"Found {len(matching_conversations)} matching conversation(s):\n\n"
            
            for i, conv in enumerate(matching_conversations[:5], 1):  # Limit to 5 results
                result += f"--- Conversation {i} ---\n"
                result += f"ID: {conv['conversation_id']}\n"
                result += f"User: {conv['user_id']}\n"
                result += f"Project: {conv['project_idea']}\n"
                result += f"Created: {conv['created_at']}\n"
                result += f"Data: {conv['conversation_data'][:200]}{'...' if len(conv['conversation_data']) > 200 else ''}\n\n"
            
            if len(matching_conversations) > 5:
                result += f"... and {len(matching_conversations) - 5} more conversation(s)."
            
            return result
            
        except Exception as e:
            return f"Error retrieving conversations: {str(e)}"

    def get_all_conversation_ids(self) -> List[str]:
        """Utility method to get all stored conversation IDs."""
        return list(self._conversations.keys())
    
    def get_conversation_count(self) -> int:
        """Utility method to get total number of stored conversations."""
        return len(self._conversations)
    
    def clear_all_conversations(self) -> str:
        """Utility method to clear all stored conversations."""
        self._conversations.clear()
        return "All conversations cleared from memory."