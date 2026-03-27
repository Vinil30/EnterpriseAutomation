# manager_agent.py — no imports from graph.py, no circular dependency

from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool


class ManagerAgent:
    def __init__(self, api_key: str, tools: List[BaseTool], model: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0
        ).bind_tools(tools)   # tools passed in, not imported

    def run(self, state: Dict[str, Any]) -> AIMessage:
        history = list(state.get("messages", []))

        prompt = f"""
You are an autonomous manager agent. Decide whether to act or do nothing.

RULES:
- If previous_action exists in state → call validation_agent_tool
- If no previous_action → call action_agent_tool  
- If metrics look healthy → do NOT call any tool, just say "System stable."

STATE SNAPSHOT:
- ba_output: {state.get('ba_output')}
- anomaly_output: {state.get('anomaly_output')}
- previous_action: {state.get('previous_action')}
- action_taken: {state.get('action_taken')}
"""
        messages = [
            SystemMessage(content="You are a tool-calling manager agent."),
            *history,
            HumanMessage(content=prompt),
        ]

        return self.llm.invoke(messages)