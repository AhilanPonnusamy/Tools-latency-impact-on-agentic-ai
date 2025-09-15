from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import uvicorn
import threading
# --- NEW: Import CORSMiddleware ---
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- NEW: Add CORS middleware to allow requests from the browser ---
# This will handle the "OPTIONS" preflight request and prevent the 405 error.
origins = ["*"] # Allows all origins for this local experiment

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DYNAMIC LATENCY CONFIGURATION ---
# This dictionary holds the latency settings for each profile.
# Latencies are distributed to total 0.5s, 2s, and 4s respectively.
LATENCY_CONFIG = {
    "ultra": {"profile": 0.25, "order": 0.15, "product": 0.10}, # Total: 0.5s
    "super": {"profile": 1.2, "order": 0.5, "product": 0.3},    # Total: 2.0s
    "standard": {"profile": 2.5, "order": 1.0, "product": 0.5}, # Total: 4.0s
}

# A thread-safe way to store the current setting
class AppState:
    def __init__(self):
        self.latency_profile = "super" # Default profile
        self.lock = threading.Lock()

    def get_profile(self):
        with self.lock:
            return self.latency_profile

    def set_profile(self, profile: str):
        with self.lock:
            if profile not in LATENCY_CONFIG:
                raise ValueError("Invalid latency profile")
            self.latency_profile = profile
            print(f"LATENCY PROFILE CHANGED TO: {profile.upper()}")

app_state = AppState()

def get_latency(tool_name: str) -> float:
    """Returns the simulated latency for a given tool based on the current profile."""
    profile = app_state.get_profile()
    return LATENCY_CONFIG[profile][tool_name]

# --- Endpoint to change latency profile ---
class LatencyProfile(BaseModel):
    profile: str

@app.post("/set_latency_profile")
def set_latency_profile(request: LatencyProfile):
    try:
        app_state.set_profile(request.profile)
        return {"status": "success", "new_profile": request.profile}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Tool Definitions (No changes here) ---
class UserProfileRequest(BaseModel):
    user_name: str = "jane doe"

@app.post("/get_user_profile")
def get_user_profile(request: UserProfileRequest):
    latency = get_latency("profile")
    print(f"TOOL: get_user_profile called. Simulating {latency}s latency...")
    time.sleep(latency)
    if request.user_name.lower() == "jane doe":
        return {"user_id": "user123", "name": "Jane Doe", "tier": "Gold"}
    return {"error": "User not found"}

class OrderHistoryRequest(BaseModel):
    user_id: str = "user123"

@app.post("/get_order_history")
def get_order_history(request: OrderHistoryRequest):
    latency = get_latency("order")
    print(f"TOOL: get_order_history called. Simulating {latency}s latency...")
    time.sleep(latency)
    if request.user_id == "user123":
        return {"orders": [{"order_id": "A987", "product_id": "CS-5000", "date": "2025-09-10"}]}
    return {"error": "No orders found"}

class ProductDetailsRequest(BaseModel):
    product_id: str = "CS-5000"

@app.post("/get_product_details")
def get_product_details(request: ProductDetailsRequest):
    latency = get_latency("product")
    print(f"TOOL: get_product_details called. Simulating {latency}s latency...")
    time.sleep(latency)
    if request.product_id == "CS-5000":
        return {"product_id": "CS-5000", "product_name": "Chrono-Synthesizer 5000", "status": "Shipped"}
    return {"error": "Product not found"}

if __name__ == "__main__":
    print(f"MCP Server starting with default LATENCY_PROFILE: '{app_state.get_profile()}'")
    uvicorn.run(app, host="0.0.0.0", port=8000)

