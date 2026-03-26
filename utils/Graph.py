from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage
from typing import TypedDict, Optional, Dict, Annotated, Sequence
import operator
from langsmith import traceable
from utils.ActionAgent import ActionAgent
from utils.ValidationAgent import ValidationAgent
from utils.ManagerAgent import ManagerAgent
class AgentState(TypedDict):
    api_key: str
    ba_output: dict
    anomaly_output: dict
    previous_action: Optional[dict]
    previous_ba: Optional[dict]
    previous_anomaly: Optional[dict]
    action_taken: Optional[bool]
    messages: Annotated[Sequence[BaseMessage], operator.add]


# -------- TOOLS (defined here, no tools.py needed) --------

@tool
def action_agent_tool(ba_output: Dict, anomaly_output: Dict, api_key: str) -> Dict:
    """Suggest an action based on business analysis and detected anomalies."""
    agent = ActionAgent(api_key=api_key)
    return agent.suggest_action(ba_output, anomaly_output)


@tool
def validation_agent_tool(
    previous_action: Dict,
    previous_ba: Dict,
    previous_anomaly: Dict,
    current_ba: Dict,
    current_anomaly: Dict,
    action_taken: bool,
    api_key: str,
) -> Dict:
    """Validate whether a previous action improved business metrics."""
    agent = ValidationAgent(api_key=api_key)
    return agent.validate(
        previous_action,
        previous_ba,
        previous_anomaly,
        current_ba,
        current_anomaly,
        action_taken,
    )


TOOLS = [action_agent_tool, validation_agent_tool]
tool_node = ToolNode(TOOLS)


# -------- ROUTING --------

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# -------- PIPELINE NODES --------

@traceable(name="DataTransformationAgent")
def data_transformer(state: AgentState) -> AgentState:
    return state


@traceable(name="DataAnalysisAgent")
def data_analysis(state: AgentState) -> AgentState:
    return state


@traceable(name="AnomalyDetectionAgent")
def anomaly_detection(state: AgentState) -> AgentState:
    return state


@traceable(name="ManagerAgent")
def manager_agent_node(state: AgentState) -> dict:
    manager = ManagerAgent(api_key=state["api_key"], tools=TOOLS)
    response = manager.run(state)
    return {"messages": [response]}


# -------- GRAPH --------

graph = StateGraph(AgentState)

graph.add_node("DataTransformer", data_transformer)
graph.add_node("DataAnalysis", data_analysis)
graph.add_node("AnomalyDetection", anomaly_detection)
graph.add_node("ManagerAgent", manager_agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "DataTransformer")
graph.add_edge("DataTransformer", "DataAnalysis")
graph.add_edge("DataAnalysis", "AnomalyDetection")
graph.add_edge("AnomalyDetection", "ManagerAgent")

graph.add_conditional_edges("ManagerAgent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "ManagerAgent")

app = graph.compile()