import os
import json
from dotenv import load_dotenv
from tavily import TavilyClient
from anthropic import Anthropic

# Import the local modules we just built
from medical_gate import MedicalGate
from xgb import UncertaintyScorer
from schemas import ClinicalSummary

# Load environment variables from the .env file
load_dotenv()

class CareCallerPipeline:
    """
    The master orchestrator. This ties together the ML scorer, the medical gate, 
    Tavily verification, and Claude safe-correction.
    """
    def __init__(self):
        # Initialize API Clients
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize our Local Intelligence Layer
        self.gate = MedicalGate()
        self.scorer = UncertaintyScorer()

    def verify_with_tavily(self, term: str) -> str | None:
        """
        Pings Tavily to find the canonical drug name. 
        Returns the search snippet if found, None if unverified.
        """
        print(f"🔍 Searching Tavily for: {term}")
        try:
            # We ask a highly specific medical question to guide the search
            query = f"What is the canonical medical name and correct spelling for the drug '{term}'?"
            
            # search_depth="basic" is faster. max_results=1 keeps it cheap.
            response = self.tavily_client.search(query=query, search_depth="basic", max_results=1)
            
            if response and len(response.get("results", [])) > 0:
                # Return the content snippet so Claude can read it to find the exact spelling
                return response["results"][0]["content"]
        except Exception as e:
            print(f"Tavily Error: {e}")
            
        return None

    def run_pipeline(self, raw_words: list[dict]) -> dict:
        """
        Runs the full 4-step intelligence pipeline on a raw transcript.
        """
        words_to_verify = []
        full_raw_transcript = ""

        for w_data in raw_words:
            word = w_data["word"]
            full_raw_transcript += word + " "
            
            # Pass the word to your XGBoost shell
            confidence, risk = self.scorer.score_word(w_data)
            
            # STRICT GATE: Only verify if it's a medical term AND confidence is LOW
            if confidence == "LOW" and self.gate.is_medical_term(word):
                words_to_verify.append(word)
                
        # Limit to 5 API calls per transcript to prevent rate limits / high costs
        words_to_verify = list(set(words_to_verify))[:5]

        # TAVILY LOOKUP
        tavily_context = {}
        for term in words_to_verify:
            tavily_result = self.verify_with_tavily(term)
            if tavily_result:
                tavily_context[term] = tavily_result

        # CLAUDE CORRECTION (Call 1)
        print("📝 Running Claude Safe-Correction...")
        correction_prompt = f"""
        You are a strict medical transcription corrector.
        
        RAW TRANSCRIPT: {full_raw_transcript.strip()}
        
        TAVILY VERIFICATIONS (Source of Truth): 
        {json.dumps(tavily_context)}
        
        STRICT RULES:
        1. ONLY correct words if the correction is explicitly supported by the TAVILY VERIFICATIONS.
        2. NEVER guess a correction. If a word sounds like a drug but Tavily didn't verify it, DO NOT fix it.
        3. If a medical word is clearly garbled and NOT in the Tavily verifications, replace it with [UNVERIFIED].
        4. Output ONLY the final corrected text string. No conversational filler.
        """

        correction_response = self.claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=0.0, # 0.0 means zero creativity = zero hallucinations
            messages=[{"role": "user", "content": correction_prompt}]
        )
        corrected_text = correction_response.content[0].text

        # --- STEP 4: CLAUDE EXTRACTION (Call 2) ---
        print("🏥 Extracting Structured JSON Summary...")
        
        # We use Anthropic's 'Tool Use' to force it to output our Pydantic schema perfectly
        extraction_tool = {
            "name": "extract_clinical_summary",
            "description": "Extracts medications, symptoms, and allergies from the corrected transcript.",
            "input_schema": ClinicalSummary.model_json_schema()
        }

        extraction_response = self.claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=0.0,
            tools=[extraction_tool],
            tool_choice={"type": "tool", "name": "extract_clinical_summary"},
            messages=[{"role": "user", "content": f"Extract structured data from this transcript: {corrected_text}"}]
        )

        final_json = {}
        for block in extraction_response.content:
            if block.type == "tool_use":
                final_json = block.input

        return {
            "raw_transcript": full_raw_transcript.strip(),
            "corrected_transcript": corrected_text,
            "clinical_summary": final_json
        }

# --- Quick Test ---
if __name__ == "__main__":
    pipeline = CareCallerPipeline()
    
    # Let's simulate a bad transcript coming from Person A
    mock_transcript = [
        {"word": "i", "start_ms": 0, "end_ms": 200},
        {"word": "take", "start_ms": 250, "end_ms": 400},
        {"word": "metoformin", "start_ms": 450, "end_ms": 500}, # Fast/Mumbled!
        {"word": "500mg", "start_ms": 550, "end_ms": 800},
        {"word": "daily", "start_ms": 850, "end_ms": 1100},
        {"word": "for", "start_ms": 1150, "end_ms": 1300},
        {"word": "my", "start_ms": 1350, "end_ms": 1500},
        {"word": "fever", "start_ms": 1550, "end_ms": 1800}
    ]
    
    result = pipeline.run_pipeline(mock_transcript)
    
    print("\n--- PIPELINE RESULTS ---")
    print(f"RAW:       {result['raw_transcript']}")
    print(f"CORRECTED: {result['corrected_transcript']}")
    print("JSON EXTRACT:")
    print(json.dumps(result['clinical_summary'], indent=2))