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
from pydantic import BaseModel, field_validator, ValidationError, Field

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
        if not data:
            print(f"Skipping empty data for {filepath}")
            return
            
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(data[0].keys())
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
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# -------------------------------------------------------------
# STEP 4: DEFINE HELPERS AND TOOLS
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

class GetKycInput(BaseModel):
    """Input for getting KYC record."""
    customer_id: str = Field(description="The customer ID.")

@tool(args_schema=GetKycInput)
def get_kyc_record_tool(customer_id: str) -> str:
    """Fetches the KYC record for a given customer ID."""
    record = retrieve_kyc_data(customer_id)
    return record.model_dump_json(indent=2) if record else "KYC record not found."

class GetTransactionHistoryInput(BaseModel):
    """Input for getting transaction history."""
    customer_id: str = Field(description="The customer ID.")

@tool(args_schema=GetTransactionHistoryInput)
def get_transaction_history_tool(customer_id: str) -> str:
    """Fetches transaction history for a customer ID."""
    transactions = retrieve_transactions(customer_id)
    return json.dumps([t.model_dump(mode='json') for t in transactions], indent=2)

class CheckPepStatusInput(BaseModel):
    """Input for checking PEP status."""
    names: List[str] = Field(description="List of names to check against the PEP list.")

@tool(args_schema=CheckPepStatusInput)
def check_pep_status_tool(names: List[str]) -> str:
    """Checks if any of the provided names are on the Politically Exposed Persons (PEP) list."""
    pep_names_lower = {p['name'].lower() for p in pep_list_csv}
    results = []
    for name in names:
        if name.lower() in pep_names_lower:
            pep_info = next((p for p in pep_list_csv if p['name'].lower() == name.lower()), None)
            results.append({
                "name": name,
                "is_pep": True,
                "pep_info": pep_info
            })
        else:
            results.append({
                "name": name,
                "is_pep": False,
                "pep_info": None
            })
    return json.dumps(results, indent=2)

# -------------------------------------------------------------
# STEP 5: INITIALIZE AND RUN THE AGENTIC SYSTEM
# -------------------------------------------------------------
def initialize_system():
    global kyc_data_csv, transactions_data_csv, pep_list_csv, sanctioned_countries_csv, ubo_data_csv
    
    # Generate and load data
    generate_source_data_csvs()
    data_dir = 'data'
    kyc_data_csv = load_csv_data(os.path.join(data_dir, 'kyc_data.csv'))
    transactions_data_csv = load_csv_data(os.path.join(data_dir, 'transactions.csv'))
    pep_list_csv = load_csv_data(os.path.join(data_dir, 'pep_list.csv'))
    sanctioned_countries_csv = load_csv_data(os.path.join(data_dir, 'sanctioned_countries.csv'))
    ubo_data_csv = load_csv_data(os.path.join(data_dir, 'ubo_data.csv'))

    # Define tools
    tools = [
        get_kyc_record_tool,
        get_transaction_history_tool,
        check_pep_status_tool
    ]

    # Use a tool-enabled Ollama model.
    # The default `llama3` does not support tools.
    # Use `llama3.1`, which is known to support tool calling.
    # You must run `ollama pull llama3.1` in your terminal first.
    llm = ChatOllama(model="llama3.1", temperature=0)

    # Agent setup
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized financial compliance officer. Your task is to analyze customer information and transaction history to assess potential risks. You have access to several tools to retrieve relevant data."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm_with_tools = llm.bind_tools(tools)
    
    # Create the runnable agent
    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

# -------------------------------------------------------------
# STEP 6: ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the system, which also loads data
    agent_executor = initialize_system()

    # --- Start Automated Investigation ---
    print("--- Starting automated investigation for customer CUST103 ---")
    investigation_query = "What is the KYC information for Alice Johnson (CUST103), what is her transaction history, and what is her PEP status? Also check if any associated UBOs are on the PEP list and sanctioned countries"
    
    # Run the full investigation query through the agent
    final_response = agent_executor.invoke({"input": investigation_query, "chat_history": []})

    # --- Output to JSON file ---
    output_dir = 'output_report'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    output_filename = os.path.join(output_dir, 'report_CUST103.json')
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # The agent's output is already a dict, which can be directly dumped to JSON.
            json.dump(final_response, f, indent=4)
        print(f"Investigation report saved to {output_filename}")
    except IOError as e:
        print(f"Error saving file: {e}")
    
    print("\nFinal Agent Response:")
    print(final_response["output"])

