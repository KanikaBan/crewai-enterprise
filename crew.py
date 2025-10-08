import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	SerplyWebSearchTool,
	SerplyScholarSearchTool,
	ScrapeWebsiteTool
)
from conversational_innovation_assistant_with_jira_integration.tools.conversation_contextualizer import ConversationContextualizerTool
from conversational_innovation_assistant_with_jira_integration.tools.conversation_memory_manager import ConversationMemoryManagerTool

from crewai_tools import CrewaiEnterpriseTools



@CrewBase
class ConversationalInnovationAssistantWithJiraIntegrationCrew:
    """ConversationalInnovationAssistantWithJiraIntegration crew"""

    
    @agent
    def research_agent(self) -> Agent:

        
        return Agent(
            config=self.agents_config["research_agent"],
            
            
            tools=[
				SerplyWebSearchTool(),
				SerplyScholarSearchTool(),
				ScrapeWebsiteTool(),
				ConversationContextualizerTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def validation_agent(self) -> Agent:

        
        return Agent(
            config=self.agents_config["validation_agent"],
            
            
            tools=[
				SerplyWebSearchTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def technical_planning_agent(self) -> Agent:

        
        return Agent(
            config=self.agents_config["technical_planning_agent"],
            
            
            tools=[
				SerplyWebSearchTool(),
				ScrapeWebsiteTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def business_strategy_agent(self) -> Agent:

        
        return Agent(
            config=self.agents_config["business_strategy_agent"],
            
            
            tools=[
				SerplyWebSearchTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def project_structuring_agent(self) -> Agent:

        
        return Agent(
            config=self.agents_config["project_structuring_agent"],
            
            
            tools=[

            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def intent_context_manager_agent(self) -> Agent:

        
        return Agent(
            config=self.agents_config["intent_context_manager_agent"],
            
            
            tools=[
				ConversationMemoryManagerTool(),
				ConversationContextualizerTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def jira_ticket_generator_agent(self) -> Agent:
        enterprise_actions_tool = CrewaiEnterpriseTools(
            actions_list=[
                
                "jira_create_issue",
                
                "jira_search_by_jql",
                
            ],
        )

        
        return Agent(
            config=self.agents_config["jira_ticket_generator_agent"],
            
            
            tools=[
				*enterprise_actions_tool
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    

    
    @task
    def detect_project_intent_context(self) -> Task:
        return Task(
            config=self.tasks_config["detect_project_intent_context"],
            markdown=False,
            
            
        )
    
    @task
    def quick_research_check(self) -> Task:
        return Task(
            config=self.tasks_config["quick_research_check"],
            markdown=False,
            
            
        )
    
    @task
    def gather_project_details(self) -> Task:
        return Task(
            config=self.tasks_config["gather_project_details"],
            markdown=False,
            
            
        )
    
    @task
    def create_technical_overview(self) -> Task:
        return Task(
            config=self.tasks_config["create_technical_overview"],
            markdown=False,
            
            
        )
    
    @task
    def create_business_summary(self) -> Task:
        return Task(
            config=self.tasks_config["create_business_summary"],
            markdown=False,
            
            
        )
    
    @task
    def create_conversational_project_summary(self) -> Task:
        return Task(
            config=self.tasks_config["create_conversational_project_summary"],
            markdown=False,
            
            
        )
    
    @task
    def generate_jira_ticket(self) -> Task:
        return Task(
            config=self.tasks_config["generate_jira_ticket"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the ConversationalInnovationAssistantWithJiraIntegration crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

    def _load_response_format(self, name):
        with open(os.path.join(self.base_directory, "config", f"{name}.json")) as f:
            json_schema = json.loads(f.read())

        return SchemaConverter.build(json_schema)
