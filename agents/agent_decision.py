"""
Agent Decision System for Multi-Agent Medical Chatbot using Local Ollama Models.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama  # CHANGED: Import local Ollama connector
from langgraph.graph import MessagesState, StateGraph, END
import os
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails
from langgraph.checkpoint.memory import MemorySaver
from config import Config

load_dotenv()
config = Config()
memory = MemorySaver()
thread_config = {"configurable": {"thread_id": "1"}}

class AgentConfig:
    """Configuration settings optimized for 8GB RAM using Phi-3 Mini."""
    
    # CHANGED: Use Local Ollama Phi-3 instead of GPT-4o
    DECISION_MODEL_NAME = "phi3:mini" 
    
    # Initialize the local reasoning model
    # ChatOllama manages memory efficiently for 8GB systems
    decision_llm = ChatOllama(
        model=DECISION_MODEL_NAME,
        temperature=0,
    )
    
    CONFIDENCE_THRESHOLD = 0.85
    
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system. 
    Analyze the user's request and determine the best specialized agent.
    
    Available agents:
    1. CONVERSATION_AGENT - General chat/greetings.
    2. RAG_AGENT - Specific medical knowledge questions from literature.
    3. WEB_SEARCH_PROCESSOR_AGENT - Recent developments/outbreaks.
    4. BRAIN_TUMOR_AGENT - Brain MRI analysis.
    5. CHEST_XRAY_AGENT - Chest X-ray analysis.
    6. SKIN_LESION_AGENT - Skin lesion analysis.

    JSON Format Required:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "reasoning steps",
    "confidence": 0.95
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)

class AgentState(MessagesState):
    agent_name: Optional[str]
    current_input: Optional[Union[str, Dict]]
    has_image: bool
    image_type: Optional[str]
    output: Optional[str]
    needs_human_validation: bool
    retrieval_confidence: float
    bypass_routing: bool
    insufficient_info: bool

class AgentDecision(TypedDict):
    agent: str
    reasoning: str
    confidence: float

def create_agent_graph():
    # Initialize guardrails using local config
    guardrails = LocalGuardrails(config.rag.llm)

    # Use the Local Ollama model for decisions
    decision_model = AgentConfig.decision_llm
    
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    decision_chain = decision_prompt | decision_model | json_parser
    
    def analyze_input(state: AgentState) -> AgentState:
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
        
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True
                }
        
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            # This calls moondream internally via ImageAnalysisAgent
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
        
        return {**state, "has_image": has_image, "image_type": image_type, "bypass_routing": False}
    
    def check_if_bypassing(state: AgentState) -> str:
        return "apply_guardrails" if state.get("bypass_routing", False) else "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")
        
        recent_context = ""
        for msg in messages[-6:]:
            prefix = "User: " if isinstance(msg, HumanMessage) else "Assistant: "
            recent_context += f"{prefix}{msg.content}\n"
        
        decision_input = f"User query: {input_text}\nContext: {recent_context}\nHas image: {has_image}\nType: {image_type}"
        
        decision = decision_chain.invoke({"input": decision_input})
        updated_state = {**state, "agent_name": decision["agent"]}
        
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    def run_conversation_agent(state: AgentState) -> AgentState:
        print(f"Selected agent: CONVERSATION_AGENT")
        input_text = state["current_input"] if isinstance(state["current_input"], str) else state["current_input"].get("text", "")
        
        # Use Phi-3 locally for conversation to save RAM
        response = AgentConfig.decision_llm.invoke(f"As a medical assistant, answer this: {input_text}")

        return {**state, "output": response, "agent_name": "CONVERSATION_AGENT"}
    
    def run_rag_agent(state: AgentState) -> AgentState:
        print(f"Selected agent: RAG_AGENT")
        rag_agent = MedicalRAG(config)
        query = state["current_input"]
        
        # RAG uses the configured LLM (Ensure config.rag.llm is also set to Ollama)
        response = rag_agent.process_query(query)
        retrieval_confidence = response.get("confidence", 0.0)
        
        return {
            **state,
            "output": AIMessage(content=response["response"]),
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": "don't have enough information" in response["response"].lower()
        }

    # ... [Remaining methods: run_web_search, run_chest_xray, etc. remain structurally similar]
    # Ensure they use state["output"] = AIMessage(content=...) for compatibility

    def run_chest_xray_agent(state: AgentState) -> AgentState:
        image_path = state["current_input"].get("image")
        # AgentConfig.image_analyzer should now be configured to use 'moondream' via Ollama
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)
        
        content = f"Analysis indicates: {predicted_class.upper()}"
        return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "CHEST_XRAY_AGENT"}

    # Define Workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    # ... [Add other nodes as per original code]
    
    workflow.set_entry_point("analyze_input")
    workflow.add_conditional_edges("analyze_input", check_if_bypassing, {"apply_guardrails": END, "route_to_agent": "route_to_agent"})
    
    # [Final Routing Edge]
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "needs_validation": "RAG_AGENT" 
        }
    )
    
    return workflow.compile(checkpointer=memory)

def process_query(query: Union[str, Dict]) -> str:
    graph = create_agent_graph()
    state = {"messages": [HumanMessage(content=str(query))], "current_input": query}
    result = graph.invoke(state, thread_config)
    return result