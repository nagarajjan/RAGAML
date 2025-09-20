import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, field_validator, ValidationError

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables to store loaded CSV data
kyc_data_csv = []
transactions_data_csv = []
pep_list_csv = []
sanctioned_countries_csv = []
ubo_data_csv = []

# -------------------------------------------------------------
# STEP 1: DEFINE DATA MODELS
# -------------------------------------------------------------
class KYCRecord(BaseModel):
    customer_id: str
    name: str
    date_of_birth: str
    address: str
    occupation: str
    source_of_wealth: str
    country: str
    business_type: str

class Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float
    type: str
    counterparty: str
    
    @field_validator('timestamp', mode='before')
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

class UBO(BaseModel):
    name: str
    ownership_percentage: float

class SAR(BaseModel):
    customer_id: str
    risk_summary: str
    investigation_details: Dict
    risk_level: str
    timestamp: datetime
    status: str = "Draft"

# -------------------------------------------------------------
# STEP 2: GENERATE CSV SOURCE DATA FILES
# -------------------------------------------------------------
def generate_source_data_csvs():
    """Generates mock source data and saves it to separate CSV files."""
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    kyc_data = [
        {"customer_id": "CUST101", "name": "Jane Doe", "date_of_birth": "1985-05-20", "address": "123 Example Street, Anytown", "country": "Country A", "occupation": "Consultant", "source_of_wealth": "Savings and investments", "business_type": "Offshore Consulting"},
        {"customer_id": "CUST102", "name": "John Smith", "date_of_birth": "1990-01-15", "address": "456 Another Avenue, Metropolis", "country": "Country B", "occupation": "Business Owner", "source_of_wealth": "Inheritance and business profits", "business_type": "Local Retail"},
        {"customer_id": "CUST103", "name": "Alice Johnson", "date_of_birth": "1978-11-30", "address": "789 Main Boulevard, Smallville", "country": "Country B", "occupation": "Dentist", "source_of_wealth": "Business earnings", "business_type": "Dental Clinic"},
    ]
    transactions_data = [
        {"transaction_id": "TX001", "customer_id": "CUST101", "timestamp": "2025-09-01T10:00:00", "amount": 5000.0, "type": "deposit", "counterparty": "Business A"},
        {"transaction_id": "TX002", "customer_id": "CUST101", "timestamp": "2025-09-05T11:00:00", "amount": 4900.0, "type": "deposit", "counterparty": "Business B"},
        {"transaction_id": "TX003", "customer_id": "CUST101", "timestamp": "2025-09-10T15:30:00", "amount": 95000.0, "type": "wire_transfer", "counterparty": "Offshore Holdings Ltd."},
        {"transaction_id": "TX004", "customer_id": "CUST102", "timestamp": "2025-09-02T09:15:00", "amount": 15000.0, "type": "deposit", "counterparty": "Client Payment"},
        {"transaction_id": "TX005", "customer_id": "CUST102", "timestamp": "2025-09-08T14:00:00", "amount": 250000.0, "type": "wire_transfer", "counterparty": "Local Supplier"},
        {"transaction_id": "TX006", "customer_id": "CUST103", "timestamp": "2025-09-03T12:30:00", "amount": 12000.0, "type": "deposit", "counterparty": "Patient Billing"},
        {"transaction_id": "TX007", "customer_id": "CUST103", "timestamp": "2025-09-09T16:45:00", "amount": 11000.0, "type": "withdrawal", "counterparty": "ATM"},
    ]
    pep_data = [
        {"name": "Prominent Official", "role": "Senior Politician", "country": "Country A"},
    ]
    sanctioned_countries_data = [
        {"country": "Country A"},
        {"country": "Country C"},
    ]
    ubo_data = [
        {"customer_id": "CUST101", "name": "Jane Doe", "ownership_percentage": 50.0},
        {"customer_id": "CUST101", "name": "Prominent Official", "ownership_percentage": 50.0},
        {"customer_id": "CUST102", "name": "John Smith", "ownership_percentage": 100.0},
        {"customer_id": "CUST103", "name": "Alice Johnson", "ownership_percentage": 100.0},
    ]

    def write_to_csv(filepath, data):
        if not data: return
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(data[0].keys()) if data else []
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Generated {filepath}")

    write_to_csv(os.path.join(data_dir, 'kyc_data.csv'), kyc_data)
    write_to_csv(os.path.join(data_dir, 'transactions.csv'), transactions_data)
    write_to_csv(os.path.join(data_dir, 'pep_list.csv'), pep_data)
    write_to_csv(os.path.join(data_dir, 'sanctioned_countries.csv'), sanctioned_countries_data)
    write_to_csv(os.path.join(data_dir, 'ubo_data.csv'), ubo_data)


# -------------------------------------------------------------
# STEP 3: LOAD DATA FROM CSV FILES (Helper Function)
# -------------------------------------------------------------
def load_csv_data(filepath: str) -> List[Dict]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# -------------------------------------------------------------
# STEP 4: DEFINE HELPERS AND TOOLS (Moved outside initialize_system)
# -------------------------------------------------------------
def retrieve_kyc_data(customer_id: str) -> Optional[KYCRecord]:
    for row in kyc_data_csv:
        if row['customer_id'] == customer_id:
            try:
                return KYCRecord(**row)
            except ValidationError as e:
                print(f"Validation error for KYC record {customer_id}: {e}")
                return None
    return None

def retrieve_transactions(customer_id: str) -> List[Transaction]:
    transactions = []
    for t in transactions_data_csv:
        if t['customer_id'] == customer_id:
            try:
                transactions.append(Transaction(**t))
            except ValidationError as e:
                print(f"Validation error for transaction {t['transaction_id']}: {e}")
    return transactions

def get_ubo_data_from_csv(customer_id: str) -> List[UBO]:
    ubos = []
    for ubo in ubo_data_csv:
        if ubo['customer_id'] == customer_id:
            try:
                ubos.append(UBO(**ubo))
            except ValidationError as e:
                print(f"Validation error for UBO data: {e}")
    return ubos

@tool
def get_kyc_record_tool(customer_id: str) -> str:
    """Fetches the KYC record for a given customer ID."""
    record = retrieve_kyc_data(customer_id)
    return record.model_dump_json(indent=2) if record else "KYC record not found."

@tool
def get_transaction_history_tool(customer_id: str) -> str:
    """Fetches transaction history for a customer ID."""
    transactions = retrieve_transactions(customer_id)
    return json.dumps([t.model_dump(mode='json') for t in transactions])

@tool
def check_pep_status_tool(names: List[str]) -> str:
    """Checks if any of the provided names are on the Politically Exposed Persons (PEP) list."""
    pep_names_lower = {p['name'].lower() for p in pep_list_csv}
    results = []
    for name in names:
        if name.lower() in pep_names_lower:
            pep_info = next((p for p in pep_list_csv if p['name'].lower() == name.lower()), None)
            results.append(f"Match found for '{name}': {pep_info['role']} in {pep_info['country']}.")
    return "\n".join(results) if results else "No PEP matches found for the provided names."

@tool
def check_country_blacklist_tool(country: str) -> str:
    """Checks if a given country is on a sanctions or blacklist."""
    if any(row['country'] == country for row in sanctioned_countries_csv):
        return f"WARNING: The country '{country}' is on a sanctioned or blacklisted list."
    return f"The country '{country}' is not on a known blacklist."

@tool
def get_ubo_info_tool(customer_id: str) -> str:
    """Fetches Ultimate Beneficial Owner (UBO) information for a given customer ID."""
    ubos = get_ubo_data_from_csv(customer_id)
    if not ubos:
        return "No UBO information found."
    
    return json.dumps([ubo.model_dump() for ubo in ubos])

# -------------------------------------------------------------
# STEP 5: INITIALIZE AGENTS
# -------------------------------------------------------------
def initialize_agents() -> Tuple[AgentExecutor, AgentExecutor]:
    llm = ChatOllama(model="llama3", temperature=0)

    research_tools = [get_kyc_record_tool, get_transaction_history_tool, check_pep_status_tool, check_country_blacklist_tool, get_ubo_info_tool]
    research_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an AML investigation research assistant. Your task is to gather all relevant KYC, transaction, PEP, UBO, and country blacklist data for a customer. Use the provided tools."),
         MessagesPlaceholder(variable_name="messages"),
         MessagesPlaceholder(variable_name="agent_scratchpad")]
    )
    research_agent = (research_prompt | llm.bind_tools(research_tools) | OpenAIToolsAgentOutputParser())
    research_executor = AgentExecutor(agent=research_agent, tools=research_tools, verbose=False)

    reporting_prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an expert AML compliance officer. Analyze the provided investigation data, including KYC, transactions, PEP, UBOs, and country checks. Generate a SAR narrative, identify red flags, assign a risk level ('Low', 'Medium', 'High'), and format the final output as a JSON string matching the SAR pydantic model. Risk is determined by factors like large transactions, unusual patterns, sanctioned countries, and PEP exposure."),
         MessagesPlaceholder(variable_name="messages"),
         MessagesPlaceholder(variable_name="agent_scratchpad")]
    )
    reporting_agent = (reporting_prompt | llm.bind_tools([]) | OpenAIToolsAgentOutputParser())
    reporting_executor = AgentExecutor(agent=reporting_agent, tools=[], verbose=False)

    return research_executor, reporting_executor

# -------------------------------------------------------------
# STEP 6: BUILD THE LANGGRAPH WORKFLOW
# -------------------------------------------------------------
class AgentState(BaseModel):
    customer_id: str
    messages: List[BaseMessage]
    kyc_data: Optional[str] = None
    transaction_data: Optional[str] = None
    pep_check: Optional[str] = None
    country_check: Optional[str] = None
    ubo_info: Optional[str] = None
    sar_report: Optional[str] = None

def get_workflow_app(research_executor, reporting_executor):
    def call_research_agent(state: AgentState):
        ubo_names = [ubo['name'] for ubo in ubo_data_csv if ubo['customer_id'] == state.customer_id]
        kyc_record = next((d for d in kyc_data_csv if d['customer_id'] == state.customer_id), None)
        country_of_kyc = kyc_record['country'] if kyc_record else ""

        return {
            "kyc_data": get_kyc_record_tool(state.customer_id),
            "transaction_data": get_transaction_history_tool(state.customer_id),
            "pep_check": check_pep_status_tool(ubo_names),
            "country_check": check_country_blacklist_tool(country_of_kyc),
            "ubo_info": get_ubo_info_tool(state.customer_id),
            "messages": state.messages + [AIMessage(content=f"Research complete for {state.customer_id}. Data retrieved.")]
        }

    def call_reporting_agent(state: AgentState):
        prompt_text = f"""
        Based on the following data, generate a SAR report draft.
        KYC Data: {state.kyc_data}
        Transaction Data: {state.transaction_data}
        PEP Check Results: {state.pep_check}
        Country Blacklist Check: {state.country_check}
        UBO Information: {state.ubo_info}

        Instructions:
        1. Analyze all information and synthesize a risk summary, mentioning any specific red flags (e.g., PEP links, sanctioned countries, large wire transfers, structuring, offshore business).
        2. Assign a risk level: 'Low', 'Medium', or 'High'.
        3. The risk level should be based on a holistic assessment. For instance, a PEP link or sanctioned country is likely high risk, while a large transaction to a verified local supplier might be low risk.
        4. Format the final output as a JSON string matching the SAR pydantic model.
        """
        response = reporting_executor.invoke({"messages": [HumanMessage(content=prompt_text)]})
        return {"sar_report": response["output"]}

    workflow = StateGraph(AgentState)
    workflow.add_node("research", call_research_agent)
    workflow.add_node("reporting", call_reporting_agent)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "reporting")
    workflow.add_edge("reporting", END)

    return workflow.compile()


# -------------------------------------------------------------
# STEP 7: EXECUTE BATCH PROCESSING AND VISUALIZATION
# -------------------------------------------------------------
def visualize_sar_data(reports: List[SAR], transactions_data_csv):
    if not reports:
        print("No reports to visualize.")
        return

    print("\n--- Generating visualizations for SAR data ---")

    data = [{'customer_id': r.customer_id, 'risk_level': r.risk_level, 'timestamp': r.timestamp} for r in reports]
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))
    sns.countplot(x='risk_level', data=df, palette='viridis', order=['Low', 'Medium', 'High'])
    plt.title('SAR Reports by Assigned Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Number of Reports')
    plt.show()

    high_risk_customers = df[df['risk_level'] == 'High']['customer_id'].tolist()
    if high_risk_customers:
        transactions_df = pd.DataFrame(transactions_data_csv)
        high_risk_transactions = transactions_df[transactions_df['customer_id'].isin(high_risk_customers)]
        
        if not high_risk_transactions.empty:
            high_risk_transactions['timestamp'] = pd.to_datetime(high_risk_transactions['timestamp'])
            high_risk_transactions['amount'] = pd.to_numeric(high_risk_transactions['amount'])
            
            plt.figure(figsize=(12, 8))
            sns.lineplot(x='timestamp', y='amount', hue='customer_id', data=high_risk_transactions, marker='o')
            plt.title('Transaction Timeline for High-Risk Customers')
            plt.xlabel('Date')
            plt.ylabel('Amount (USD)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    
    ubo_data_df = pd.DataFrame(load_csv_data('data/ubo_data.csv'))
    ubo_connections = ubo_data_df[ubo_data_df['customer_id'].isin(high_risk_customers)]
    if not ubo_connections.empty:
        print("\n--- Network graph data for high-risk customers ---")
        print("Entity Connections (Customer to UBO):")
        print(ubo_connections[['customer_id', 'name']])


if __name__ == "__main__":
    print("Generating source data CSV files...")
    generate_source_data_csvs()

    # Load data from CSVs into global variables
    kyc_data_csv = load_csv_data('data/kyc_data.csv')
    transactions_data_csv = load_csv_data('data/transactions.csv')
    pep_list_csv = load_csv_data('data/pep_list.csv')
    sanctioned_countries_csv = load_csv_data('data/sanctioned_countries.csv')
    ubo_data_csv = load_csv_data('data/ubo_data.csv')

    research_executor, reporting_executor = initialize_agents()
    app = get_workflow_app(research_executor, reporting_executor)

    customers_to_process = list(set(d['customer_id'] for d in kyc_data_csv))
    generated_reports = {}

    for customer_id in customers_to_process:
        print(f"\n--- Starting automated investigation for customer {customer_id} ---")
        initial_state = AgentState(
            customer_id=customer_id,
            messages=[HumanMessage(content=f"Initial alert for customer {customer_id}")],
        )

        try:
            final_state = app.invoke(initial_state)
            sar_report_str = final_state.sar_report

            try:
                sar_report_str_clean = sar_report_str.strip().replace("```json\n", "").replace("\n```", "")
                sar_report_dict = json.loads(sar_report_str_clean)
                sar_report_dict['timestamp'] = datetime.now().isoformat()
                sar_report = SAR(**sar_report_dict)

                if sar_report.risk_level in ["High", "Medium"]:
                    generated_reports[customer_id] = sar_report
                    print(f"Generated a {sar_report.risk_level} risk SAR for {customer_id}.")
                else:
                    print(f"No significant suspicious activity detected for {customer_id} (Risk: {sar_report.risk_level}). No report generated.")
            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                print(f"Failed to parse LLM output for {customer_id}: {e}")
                print(f"Raw output: {sar_report_str}")

        except Exception as e:
            print(f"An error occurred while processing customer {customer_id}: {e}")

    print("\n--- BATCH PROCESSING COMPLETE ---")
    print(f"Generated {len(generated_reports)} SAR reports for review.")

    output_dir = 'output_reports'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cust_id, report in generated_reports.items():
        filename = os.path.join(output_dir, f"sar_report_{cust_id}_{report.risk_level}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(filename, "w", encoding='utf-8') as f:
            f.write(report.model_dump_json(indent=2))
        print(f"Saved report for {cust_id} to {filename}")

    all_reports = list(generated_reports.values())
    visualize_sar_data(all_reports, transactions_data_csv)
