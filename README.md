# RAGAML

# Agentic RAG for the example for AML
Agentic Retrieval-Augmented Generation (RAG) is a powerful method for enhancing anti-money laundering (AML) workflows. It improves on traditional RAG systems by incorporating autonomous AI agents that can break down complex tasks, use a variety of tools, and iteratively refine their approach

# The challenge: AML investigations
Traditional AML systems primarily use rule-based detection to flag suspicious activity, which often results in a high volume of false positives. Human investigators must then manually sift through these alerts, a resource-intensive process that can allow sophisticated threats to slip through undetected.

# Agentic RAG for AML: A detailed example
An agentic RAG system can act as a "Co-Investigator" to automate and improve the entire AML workflow, from alert review to suspicious activity report (SAR) generation. 

Hypothetical Scenario: An agentic AML system receives an alert flagging unusual transaction patterns from a new business account. 

# The multi-agent workflow
Instead of a single, static process, a team of specialized AI agents collaborates to investigate the alert: 

# Alert Triage Agent:
Action: Reviews the initial alert from the transaction monitoring system.

Tool Use: Accesses internal documents to find the rule that was triggered, such as "unusual transaction size or frequency".

Reasoning: Determines the alert's initial priority and delegates sub-tasks to other agents.

# Research and Analysis Agents:
Action: Gathers a comprehensive profile of the customer and their related entities.

# Tool Use:
Customer Database: Retrieves customer onboarding documents, beneficial ownership details, and Know Your Customer (KYC) records.

External APIs: Scans for adverse media—news reports, sanctions lists, and watchlists—related to the account's owners or directors.

Transaction Database: Analyzes the account's current and historical transaction data, as well as the behavior of peer groups, to identify behavioral anomalies.

Reasoning: Identifies a pattern of multiple small, round-number deposits followed by larger transfers to offshore accounts—a sign of "layering" in money laundering.

# Validation Agent:
Action: Verifies the accuracy and coherence of the information collected by other agents.

Tool Use: Cross-references data points, checks for conflicts, and flags any inconsistencies to the system.

Reasoning: Ensures that the gathered evidence is reliable and well-supported, preventing the propagation of errors or "hallucinations".

# Reporting Agent:
Action: Drafts a preliminary SAR for human review.

Tool Use: Uses the synthesized findings from the other agents to generate a clear, narrative-driven report.

Reasoning: Ensures the report is factually grounded, explains the rationale for the suspicion, and aligns with regulatory standards.

# Human Investigator-in-the-Loop:
Action: Reviews the AI-generated SAR.

Tool Use: An integrated interface allows the investigator to review the full audit trail of the agents' actions, conversations, and data sources.

Reasoning: Applies domain expertise to refine the draft, handle complex exceptions, and make the final decision to file the report. 

# Key advantages over traditional methods
Reduced False Positives: By analyzing behavior and context instead of relying on simple rules, agentic RAG drastically lowers the number of false alarms, allowing human analysts to focus on real threats.

Faster, More Accurate Investigations: Automated information gathering and analysis significantly cut down the time needed to prepare a case, from weeks to hours.

Enhanced Discovery: The multi-agent system can uncover sophisticated, non-obvious money laundering patterns that are difficult for humans to spot.

Superior Auditing and Explainability: The system's step-by-step reasoning provides a complete audit trail of how conclusions were reached, which is crucial for regulatory compliance.

Improved Adaptability: Unlike static rule-based systems, an agentic system can continuously learn from new fraud patterns and update its workflow, providing an evolving defense against financial crime. 

# Overview of the code architecture
The following code demonstrates a simplified, modular approach to an Agentic RAG for AML. The system will:

Receive an AML alert with a customer_id.

Use a ResearchAgent to retrieve and analyze KYC records and transaction history from simulated data sources.

Use a ReportingAgent to synthesize the findings and generate a draft SAR, assisted by an LLM.

Incorporate human-in-the-loop review, allowing a compliance officer to approve or edit the final report.

This code is an illustrative example. In a production environment, you would replace the mock data and LLM calls with secure, real-world connections to your organization's databases and a robust, compliant LLM API.

# 1. Setup and dependencies
First, ensure you have the required libraries installed. You will need langchain, a client for your LLM (e.g., openai), and pydantic for data models.
sh
pip install langchain langchain_openai pydantic

# 2. Define data models
The investigation will use structured data for KYC records, transactions, and the final SAR. pydantic models ensure data consistency.

python

from pydantic import BaseModel, Field

from typing import List, Dict, Optional

from datetime import datetime

# KYC Record Model
class KYCRecord(BaseModel):

    customer_id: str
    
    name: str
    
    date_of_birth: str
    
    address: str
    
    occupation: str
    
    source_of_wealth: str

# Transaction Model
class Transaction(BaseModel):
    
    transaction_id: str
    
    customer_id: str
    
    date: str
    
    amount: float
    
    type: str # e.g., 'deposit', 'withdrawal', 'wire_transfer'
    
    counterparty: str
    
    timestamp: datetime

# SAR Model (Generated Report)
class SAR(BaseModel):
    
    customer_id: str
    
    risk_summary: str
    
    transaction_details: List[str]
    
    kyc_details: str
    
    regulatory_justification: str
    
    status: str = "Draft"


# 3. Implement simulated data sources
For this example, we will use mock databases to simulate a financial institution's internal systems.

python

# Simulated Databases

mock_kyc_database = {

    "CUST101": KYCRecord(
    
        customer_id="CUST101",
        
        name="Jane Doe",
        
        date_of_birth="1985-05-20",
        
        address="123 Example Street, Anytown",
        
        occupation="Consultant",
        
        source_of_wealth="Savings and investments"
    
    ),
    
    # Add more mock customers if needed

}

mock_transaction_database = [

    Transaction(transaction_id="TX001", customer_id="CUST101", date="2025-09-01", timestamp=datetime(2025, 9, 1), amount=5000.0, type="deposit", counterparty="Business A"),
    
    Transaction(transaction_id="TX002", customer_id="CUST101", date="2025-09-05", timestamp=datetime(2025, 9, 5), amount=4900.0, type="deposit", counterparty="Business B"),
    
    Transaction(transaction_id="TX003", customer_id="CUST101", date="2025-09-10", timestamp=datetime(2025, 9, 10), amount=95000.0, type="wire_transfer", counterparty="Offshore Holdings Ltd."),
    
    Transaction(transaction_id="TX004", customer_id="CUST101", date="2025-09-15", timestamp=datetime(2025, 9, 15), amount=4500.0, type="deposit", counterparty="Business C"),
    
    Transaction(transaction_id="TX005", customer_id="CUST101", date="2025-09-18", timestamp=datetime(2025, 9, 18), amount=98000.0, type="wire_transfer", counterparty="Overseas Investments Corp."),
    
]

def retrieve_kyc_data(customer_id: str) -> Optional[KYCRecord]:

    return mock_kyc_database.get(customer_id)

def retrieve_transactions(customer_id: str, start_date: str = None, end_date: str = None) -> List[Transaction]:

    # In a real system, you would filter by date
    
    return [t for t in mock_transaction_database if t.customer_id == customer_id]


# 4. Create specialized tools for the agents
Agents in langchain use "tools" to interact with external systems. Here, we define tools for retrieving KYC data and transaction history.

python

from langchain_core.tools import tool

@tool

def get_kyc_record_tool(customer_id: str) -> str:

    """Fetches the KYC record for a given customer ID."""
    
    record = retrieve_kyc_data(customer_id)
    
    return record.model_dump_json(indent=2) if record else "KYC record not found."

@tool

def get_transaction_history_tool(customer_id: str) -> str:

    """Fetches transaction history for a customer ID. Includes filtering capabilities in a real-world scenario."""
    
    transactions = retrieve_transactions(customer_id)
    
    return [t.model_dump_json() for t in transactions]


# 5. Define and orchestrate the agents
We will use langchain_openai to create the agents and langgraph to define the flow of control between them.

python

from langchain_openai import ChatOpenAI

from langchain.agents import AgentExecutor

from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages

from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

# Define the LLM (replace with your chosen model)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Agent 1: Research Agent ---
research_prompt = ChatPromptTemplate.from_messages(

    [
        ("system", "You are an AML investigation research assistant. Your task is to gather all relevant KYC and transaction data for a customer flagged for suspicious activity."),
        
        MessagesPlaceholder(variable_name="messages"),
        
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        
    ]
    
)

research_agent = (

    research_prompt
    
    | llm.bind_tools([get_kyc_record_tool, get_transaction_history_tool])
    
    | OpenAIToolsAgentOutputParser()
    
)

research_executor = AgentExecutor(agent=research_agent, tools=[get_kyc_record_tool, get_transaction_history_tool])

# --- Agent 2: Reporting Agent ---
reporting_prompt = ChatPromptTemplate.from_messages(

    [
        ("system", "You are an expert AML compliance officer. Your task is to synthesize the KYC and transaction data provided by the research agent to generate a Suspicious Activity Report narrative. Highlight any red flags, such as structuring or large wire transfers. The final report should follow the SAR data model."),
        
        MessagesPlaceholder(variable_name="messages"),
        
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        
    ]
    
)

reporting_agent = (

    reporting_prompt
    
    | llm.bind_tools([])  # Reporting agent primarily generates text, may not need new tools
    
    | OpenAIToolsAgentOutputParser()
    
)

reporting_executor = AgentExecutor(agent=reporting_agent, tools=[])


# 6. Create the agentic workflow with LangGraph
langgraph provides the state management to connect the agents.

python

from langgraph.graph import StateGraph, END

# Define the state for the graph

class AgentState(BaseModel):

    customer_id: str
    
    messages: List[BaseMessage]
    
    kyc_data: Optional[KYCRecord] = None
    
    transaction_data: Optional[List[Transaction]] = None
    
    sar_report: Optional[SAR] = None

# Define the workflow nodes

def call_research_agent(state: AgentState):

    response = research_executor.invoke({"messages": state.messages})
    
    kyc_record = retrieve_kyc_data(state.customer_id)
    
    transactions = retrieve_transactions(state.customer_id)
    
    return {
    
        "messages": state.messages + [AIMessage(content="Research complete. Data gathered.")],
        
        "kyc_data": kyc_record,
        
        "transaction_data": transactions,
        
    }

def call_reporting_agent(state: AgentState):

    prompt_text = f"""
    
    Based on the following data, generate a SAR report.
    
    KYC Data: {state.kyc_data.model_dump_json()}
    
    Transaction Data: {state.transaction_data}
    
    """
    
    response = reporting_executor.invoke({"messages": [HumanMessage(content=prompt_text)]})
    
    # Parse the LLM output into a SAR model. This part needs refinement in practice.
    
    sar_draft = SAR(
    
        customer_id=state.customer_id,
        
        risk_summary=response["output"],
        
        transaction_details=[t.model_dump_json() for t in state.transaction_data],
        
        kyc_details=state.kyc_data.model_dump_json(),
        
        regulatory_justification="Potential Structuring and Offshore Transfers"
        
    )
    
    return {"sar_report": sar_draft}

# Build the LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("research", call_research_agent)
workflow.add_node("reporting", call_reporting_agent)
workflow.set_entry_point("research")
workflow.add_edge("research", "reporting")
workflow.add_edge("reporting", END)

# Compile the graph

app = workflow.compile()



# 7. Run the investigation and generate the report

The final step is to execute the agentic workflow.

python

# Example Usage
if __name__ == "__main__":

    customer_id_to_investigate = "CUST101"
    
    initial_state = AgentState(
    
        customer_id=customer_id_to_investigate,
        
        messages=[HumanMessage(content=f"Investigate suspicious activity for customer {customer_id_to_investigate}")],
        
    )
    
    final_state = app.invoke(initial_state)

    sar_report = final_state.sar_report
    print("--- GENERATED SAR REPORT DRAFT ---")
    print(sar_report.model_dump_json(indent=2))

    # --- Human-in-the-Loop ---
    print("\n--- HUMAN REVIEW ---")
    print("A human compliance officer can now review, edit, and approve the SAR.")
    human_approved_sar = sar_report.copy()
    human_approved_sar.status = "Approved"
    print("\n--- FINAL APPROVED SAR ---")
    print(human_approved_sar.model_dump_json(indent=2))
