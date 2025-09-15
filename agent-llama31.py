from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# NOTE: We are no longer using the AgentExecutor
import requests
import uvicorn
import traceback
import json

app = FastAPI()

# --- CORS middleware ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- vLLM Configuration ---
print("--- Connecting to local vLLM OpenAI-compatible server... ---")
llm = ChatOpenAI(
    openai_api_base="http://127.0.0.1:8000/v1",
    api_key="NOT_USED",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    temperature=0,
)
print("--- Successfully configured to connect to vLLM ---")

# --- Tool Definitions (as simple functions) ---
MCP_SERVER_URL = "http://127.0.0.1:8002"

def get_user_profile(user_name: str) -> dict:
    """Gets the user ID and profile for a given user name."""
    print(f"Calling tool: get_user_profile with user_name='{user_name}'")
    response = requests.post(f"{MCP_SERVER_URL}/get_user_profile", json={"user_name": user_name})
    return response.json()

def get_order_history(user_id: str) -> dict:
    """Gets the order history for a given user ID."""
    print(f"Calling tool: get_order_history with user_id='{user_id}'")
    response = requests.post(f"{MCP_SERVER_URL}/get_order_history", json={"user_id": user_id})
    return response.json()

def get_product_details(product_id: str) -> dict:
    """Gets the status and details for a given product ID."""
    print(f"Calling tool: get_product_details with product_id='{product_id}'")
    response = requests.post(f"{MCP_SERVER_URL}/get_product_details", json={"product_id": product_id})
    return response.json()


# --- API Endpoint with ENFORCED SEQUENTIAL LOGIC ---
class AgentRequest(BaseModel):
    question: str

@app.post("/invoke_agent")
def invoke_agent(request: AgentRequest):
    """
    Invokes a hard-coded sequence of tool calls and uses the LLM for final summarization.
    This is a more robust pattern for measuring latency impact.
    """
    print("\n--- Starting new enforced sequential workflow ---")
    try:
        # Step 1: Always get the user profile first.
        # We assume the question is always about "Jane Doe" for this experiment.
        profile = get_user_profile("Jane Doe")
        user_id = profile.get("user_id")
        if not user_id:
            raise ValueError(f"Could not find user_id in profile: {profile}")

        # Step 2: Always get the order history next.
        history = get_order_history(user_id)
        # For simplicity, we'll just take the first product of the first order.
        product_id = history.get("orders", [{}])[0].get("product_id")
        if not product_id:
            raise ValueError(f"Could not find product_id in history: {history}")

        # Step 3: Always get the product details last.
        details = get_product_details(product_id)
        status = details.get("status")
        if not status:
             raise ValueError(f"Could not find status in details: {details}")

        # Step 4: Use the LLM for the final summarization step.
        print("--- All tool calls successful. Passing context to LLM for final answer. ---")
        
        # --- FIX: Define the prompt with placeholders ---
        summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Based on the provided context, formulate a final answer to the user's question. Be concise."),
            ("human", """
            Here is the context I have gathered from the tools:
            - User Profile: {profile_str}
            - Order History: {history_str}
            - Product Details: {details_str}

            Based on this, please answer the original question: '{question_str}'
            """)
        ])
        
        chain = summarization_prompt | llm
        
        # --- FIX: Pass the variables to the invoke method ---
        response = chain.invoke({
            "profile_str": json.dumps(profile),
            "history_str": json.dumps(history),
            "details_str": json.dumps(details),
            "question_str": request.question
        })
        final_answer = response.content

        print(f"--- Workflow complete.  {final_answer} ---")
        return {"answer": final_answer}

    except Exception as e:
        print("\n--- WORKFLOW ENCOUNTERED AN ERROR ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

