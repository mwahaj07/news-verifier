import os
import time
from typing import Literal, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tavily import TavilyClient
from google import genai
from google.genai import types

# 1. Load Environment Variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or TAVILY_API_KEY in environment.")

# Initialize SDK Clients
genai_client = genai.Client(api_key=GEMINI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# =====================================================================
# STEP 1: THE SPEC (Pydantic Data Contracts)
# =====================================================================

class VerifyRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The news headline or claim to verify.",
        examples=["NASA confirmed asteroid 2026-KX will hit the Atlantic Ocean tomorrow."]
    )

class VerifyResponse(BaseModel):
    verdict: Literal["TRUE", "FALSE", "MISLEADING", "UNVERIFIED"] = Field(
        ..., description="Final fact-check classification."
    )
    confidence: int = Field(
        ..., ge=0, le=100, description="Confidence score from 0 to 100."
    )
    search_query_used: str = Field(
        ..., description="The primary search query the agent executed."
    )
    explanation: str = Field(
        ..., description="Concise 2-3 sentence summary explaining the verdict based ONLY on search results."
    )
    sources: List[str] = Field(
        default_factory=list, description="Exact URLs from the search results that back the verdict."
    )

# =====================================================================
# STEP 2: THE AGENT'S TOOL (External Web Search)
# =====================================================================

def search_web(query: str) -> str:
    """Searches live news and web sources for a neutral query and returns snippets and URLs."""
    print(f"\n[AGENT ACTION] Calling search_web with query: '{query}'")
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=False
        )
        results = response.get("results", [])
        if not results:
            print("[AGENT OBSERVATION] 0 results found.")
            return "OBSERVATION: ZERO_RESULTS_FOUND. Stop searching and output your verdict."

        formatted = []
        for idx, item in enumerate(results, 1):
            formatted.append(
                f"[{idx}] Title: {item.get('title')}\n"
                f"    URL: {item.get('url')}\n"
                f"    Snippet: {item.get('content')}\n"
            )
        print(f"[AGENT OBSERVATION] Retrieved {len(results)} results from Tavily.")
        return "\n".join(formatted)
    except Exception as e:
        print(f"[AGENT ERROR] Search tool error: {str(e)}")
        return f"OBSERVATION: TOOL_ERROR ({str(e)}). Stop searching."

# =====================================================================
# STEP 3: THE SINGLE AGENT ORCHESTRATOR (WITH CIRCUIT BREAKER)
# =====================================================================

SYSTEM_PROMPT = """
You are a strict, fast News Verification Agent.
Follow these exact rules:
1. Call `search_web` AT MOST 1 or 2 times total. Do NOT keep searching repeatedly.
2. If your first search returns general articles that do NOT mention the specific claim or entity (e.g., a fake asteroid name or fake law), DO NOT search 5 more times.
3. Apply the Proportionality Rule immediately:
   - If the claim is a massive public event (like a NASA asteroid impact) and the top search results have zero mention of that specific event/asteroid, conclude it is FALSE (fabricated hoax) or UNVERIFIED immediately.
   - Output TRUE only if reputable sources directly confirm the claim.
   - Output MISLEADING if sources show the claim exaggerates or distorts a real event.
4. Rely 100% on the tool output and only cite URLs present in the observation.
"""

def run_verification_agent(claim: str) -> VerifyResponse:
    start_time = time.time()
    print(f"\n[NEW REQUEST] Verifying claim: '{claim}' using model '{GEMINI_MODEL}'")
    
    response = genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"Verify this news claim: {claim}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[search_web],
            # CIRCUIT BREAKER: Hard limit of maximum 2 tool calls per request!
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=2
            ),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=VerifyResponse,
        ),
    )
    
    elapsed = round(time.time() - start_time, 2)
    print(f"[COMPLETED] Verdict generated in {elapsed} seconds.")
    return VerifyResponse.model_validate_json(response.text)

# =====================================================================
# STEP 4: FASTAPI ENDPOINTS
# =====================================================================

app = FastAPI(
    title="News Verification Agent MVP",
    version="1.1.0",
    description="Spec-driven single-agent API with a 2-call circuit breaker."
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": GEMINI_MODEL}

@app.post("/verify", response_model=VerifyResponse)
def verify_news_endpoint(request: VerifyRequest):
    try:
        result = run_verification_agent(request.claim)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")