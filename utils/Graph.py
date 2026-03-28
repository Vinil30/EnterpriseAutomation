from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage
from typing import TypedDict, Optional, Dict, Annotated, Sequence
import operator
import os
import pandas as pd
from langsmith import traceable
from utils.ActionAgent import ActionAgent
from utils.ValidationAgent import ValidationAgent
from utils.ManagerAgent import ManagerAgent

class AgentState(TypedDict):
    api_key: str
    file_paths: list[str]                    # input CSV paths
    unified_dataset_path: Optional[str]      # set by data_transformer
    unified_dataframe: Optional[pd.DataFrame]  # store dataframe directly
    ba_output: dict
    anomaly_output: dict
    ba_schema: Optional[dict]                 # Store schema from BA for reuse
    previous_action: Optional[dict]
    previous_ba: Optional[dict]
    previous_anomaly: Optional[dict]
    action_taken: Optional[bool]
    validation_done: Optional[bool]           # NEW: Track if validation completed
    messages: Annotated[Sequence[BaseMessage], operator.add]
    is_first_run: Optional[bool]
    loop_counter: Optional[int]               # NEW: Safety counter to prevent infinite loops

# -------- TOOLS --------

@tool
def action_agent_tool(ba_output: Dict, anomaly_output: Dict, api_key: str) -> Dict:
    """Suggest an action based on business analysis and detected anomalies."""
    agent = ActionAgent()
    result = agent.suggest_action(ba_output, anomaly_output)
    
    # Return structured result with action taken flag
    return {
        "status": "action_suggested",
        "action": result,
        "message": "Action suggested successfully"
    }

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
    agent = ValidationAgent()
    result = agent.validate(
        previous_action,
        previous_ba,
        previous_anomaly,
        current_ba,
        current_anomaly,
        action_taken,
    )
    
    return {
        "status": "validation_complete",
        "validation_result": result,
        "message": "Validation completed"
    }

TOOLS = [action_agent_tool, validation_agent_tool]
tool_node = ToolNode(TOOLS)

# -------- ROUTING --------

def should_continue(state: AgentState) -> str:
    """Determine if we should continue to tools or end."""
    # Safety check: prevent infinite loops
    loop_counter = state.get("loop_counter", 0)
    if loop_counter >= 3:  # Max 3 iterations
        print(f"⚠️ Safety: Max iterations reached ({loop_counter}). Stopping.")
        return END
    
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

# -------- PIPELINE NODES --------

@traceable(name="DataTransformationAgent")
def data_transformer(state: AgentState) -> AgentState:
    from utils.DataUnifier import SchemaUnificationPipeline

    pipeline = SchemaUnificationPipeline(api_key=state["api_key"])
    result = pipeline.unify(state["file_paths"])

    # Load the unified dataset into a dataframe for direct use
    unified_df = None
    if result["output_file"] and os.path.exists(result["output_file"]):
        unified_df = pd.read_csv(result["output_file"])
    
    return {
        **state,
        "unified_dataset_path": result["output_file"],
        "unified_dataframe": unified_df,
        "is_first_run": True,
        "loop_counter": 0  # Initialize loop counter
    }


@traceable(name="DataAnalysisAgent")
def data_analysis(state: AgentState) -> AgentState:
    from utils.DatasetAnalyser import BusinessAnalystAgent
    
    agent = BusinessAnalystAgent(
        api_key=state["api_key"],
        verbose=True,
        use_cache=True  # Enable caching
    )
    
    if state.get("unified_dataframe") is not None:
        result = agent.analyze(state["unified_dataframe"])
    elif state.get("unified_dataset_path"):
        result = agent.analyze(state["unified_dataset_path"])
    else:
        result = {"status": "error", "error": "No data available for analysis"}
    
    new_state = {**state, "ba_output": result}
    
    # Store the detected schema for reuse by anomaly detection
    if result.get("status") == "success" and result.get("detected_schema"):
        new_state["ba_schema"] = result["detected_schema"]
        if state.get("verbose", False):
            print(f"✅ Schema stored for reuse: {list(result['detected_schema'].keys())}")
    
    return new_state

@traceable(name="AnomalyDetectionAgent")
def anomaly_detection(state: AgentState) -> AgentState:
    from utils.AnamolyDetection import AnomalyDetectionAgent
    
    agent = AnomalyDetectionAgent(
        api_key=state["api_key"],  # Still pass for root cause insights
        verbose=True
    )
    
    df = state.get("unified_dataframe")
    
    if df is not None:
        # Pass the pre-detected schema from BA (NO LLM call in anomaly detection!)
        result = agent.analyze(
            df=df,
            schema=state.get("ba_schema")  # ← KEY CHANGE: reuse schema
        )
    elif state.get("unified_dataset_path"):
        result = agent.analyze(
            df=state["unified_dataset_path"],
            schema=state.get("ba_schema")
        )
    else:
        result = {"status": "error", "error": "No data available for anomaly detection"}
    
    if state.get("verbose", False) and state.get("ba_schema"):
        print(f"✅ Anomaly detection using reused schema from Business Analyst")
    
    new_state = {**state, "anomaly_output": result}
    
    return new_state

# In graph.py - update manager_agent_node

@traceable(name="ManagerAgent")
def manager_agent_node(state: AgentState) -> dict:
    # Hard stop — return empty AIMessage with no tool calls
    if state.get("loop_counter", 0) >= 3:
        print(f"🛑 Hard stop: loop_counter={state.get('loop_counter')}. Ending.")
        return {"messages": [AIMessage(content="Analysis complete. Max iterations reached.")]}

    # Also stop if both action and validation are done
    if state.get("action_taken") and state.get("validation_done"):
        print("✅ Action + Validation complete. Ending.")
        return {"messages": [AIMessage(content="Action taken and validated. Pipeline complete.")]}

    manager = ManagerAgent(api_key=state["api_key"], tools=TOOLS)
    
    serializable_state = {
        "api_key": state["api_key"],
        "ba_output": state.get("ba_output", {}),
        "anomaly_output": state.get("anomaly_output", {}),
        "previous_ba": state.get("previous_ba"),
        "previous_anomaly": state.get("previous_anomaly"),
        "previous_action": state.get("previous_action"),
        "action_taken": state.get("action_taken", False),
        "validation_done": state.get("validation_done", False),
        "is_first_run": state.get("is_first_run", True),
        "loop_counter": state.get("loop_counter", 0)
    }
    
    response = manager.run(serializable_state)
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_call['args']['api_key'] = state["api_key"]
            if tool_call['name'] == 'action_agent_tool':
                tool_call['args']['ba_output'] = state.get("ba_output", {})
                tool_call['args']['anomaly_output'] = state.get("anomaly_output", {})
            elif tool_call['name'] == 'validation_agent_tool':
                tool_call['args']['previous_ba'] = state.get("previous_ba", {})
                tool_call['args']['previous_anomaly'] = state.get("previous_anomaly", {})
                tool_call['args']['current_ba'] = state.get("ba_output", {})
                tool_call['args']['current_anomaly'] = state.get("anomaly_output", {})
                tool_call['args']['previous_action'] = state.get("previous_action", {})
                tool_call['args']['action_taken'] = state.get("action_taken", False)
        print(f"✅ Injected full state into {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": [response]}

@traceable(name="ProcessToolResults")
def process_tool_results(state: AgentState) -> AgentState:
    new_state = {**state}
    
    # Always increment when we enter this node
    new_state["loop_counter"] = state.get("loop_counter", 0) + 1

    # Find the AIMessage with tool_calls (not the last ToolMessage)
    messages = state["messages"]
    ai_message = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)),
        None
    )

    if ai_message:
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call.get('name')
            if tool_name == 'action_agent_tool':
                new_state["action_taken"] = True
                new_state["previous_action"] = tool_call.get('args', {})
                new_state["validation_done"] = False
                new_state["is_first_run"] = False
                print(f"✅ Action recorded: {tool_name}")
            elif tool_name == 'validation_agent_tool':
                new_state["validation_done"] = True
                new_state["action_taken"] = False
                new_state["is_first_run"] = False
                print(f"✅ Validation recorded: {tool_name}")

    new_state["previous_ba"] = state.get("ba_output")
    new_state["previous_anomaly"] = state.get("anomaly_output")
    return new_state

# -------- GRAPH --------

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("DataTransformer", data_transformer)
graph.add_node("DataAnalysis", data_analysis)
graph.add_node("AnomalyDetection", anomaly_detection)
graph.add_node("ManagerAgent", manager_agent_node)
graph.add_node("tools", tool_node)
graph.add_node("process_tool_results", process_tool_results)  # New node to process results

# Add edges
graph.add_edge(START, "DataTransformer")
graph.add_edge("DataTransformer", "DataAnalysis")
graph.add_edge("DataAnalysis", "AnomalyDetection")
graph.add_edge("AnomalyDetection", "ManagerAgent")

# Conditional routing after ManagerAgent
graph.add_conditional_edges(
    "ManagerAgent", 
    should_continue, 
    {
        "tools": "tools",  # If tool calls, go to tools
        END: END           # Otherwise end
    }
)

# After tools, process results, then go back to ManagerAgent
graph.add_edge("tools", "process_tool_results")
graph.add_edge("process_tool_results", "ManagerAgent")

app = graph.compile()