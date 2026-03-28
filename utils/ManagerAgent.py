# manager_agent.py - fixed version

from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
import json


class ManagerAgent:
    def __init__(self, api_key: str, tools: List[BaseTool], model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key  # Store the actual API key
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0
        ).bind_tools(tools)

    def run(self, state: Dict[str, Any]) -> AIMessage:
        """
        Autonomous manager that decides when to call tools based on state.
        """
        
        # Extract metrics
        ba_output = state.get("ba_output", {})
        anomaly_output = state.get("anomaly_output", {})
        
        metrics = ba_output.get("metrics", {})
        trends = ba_output.get("trends", {})
        
        anomaly_summary = anomaly_output.get("summary", {})
        
        # Build state snapshot with ACTUAL API key info
        state_snapshot = f"""
BUSINESS STATE:
Revenue: ${metrics.get('primary_metric_sum', 0):,.0f}
Margin: {metrics.get('profit_margin', 0):.1f}%
Trend: {trends.get('trend', 'N/A')} ({trends.get('trend_percentage', 0)}%)
Anomaly Loss: ${anomaly_summary.get('total_anomaly_loss', 0):,.0f}

FLAGS:
First Run: {state.get('is_first_run', True)}
Action Taken: {state.get('action_taken', False)}
Validation Done: {state.get('validation_done', False)}
Loop: {state.get('loop_counter', 0)}/3

IMPORTANT: When calling tools, use api_key = "{self.api_key}" (this is the actual key)
"""

        prompt = f"""
You are a manager agent. Current state:

{state_snapshot}

DECISION RULES:
- If loop_counter >= 2 → STOP (return without tools)
- If action_taken=True and validation_done=False → CALL validation_agent_tool
- If is_first_run=True and action_taken=False → CALL action_agent_tool  
- If profit_margin > 25% and trend is stable → NO ACTION NEEDED

TOOLS:
1. action_agent_tool - parameters: ba_output (dict), anomaly_output (dict), api_key (str)
2. validation_agent_tool - parameters: previous_action (dict), previous_ba (dict), previous_anomaly (dict), current_ba (dict), current_anomaly (dict), action_taken (bool), api_key (str)

Make decision now. Use the actual api_key shown above.
"""

        messages = [
            SystemMessage(content="You are a manager agent. Make decisions based on rules. Be concise. Use the actual api_key provided."),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        
        # Log decision
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\n🤖 Manager: Calling {response.tool_calls[0]['name']}")
        else:
            print(f"\n🤖 Manager: No action needed")
            
        return response