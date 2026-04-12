from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# Word-Level Schemas (The Pipeline Data)
class WordInput(BaseModel):
    # What Person A (STT) hands to you.
    word: str
    start_ms: int
    end_ms: int
    speaker: str 

class WordOutput(BaseModel):
    # What you hand back out to the frontend for the Corrected Transcript panel.
    word: str
    start_ms: int
    end_ms: int
    speaker: str
    confidence_level: ConfidenceLevel
    changed: bool = False             # Did Claude change this word? (Amber highlight)
    tavily_verified: bool = False     # Did Tavily confirm it? (Teal underline)
    unverified: bool = False          # Did Tavily fail to confirm it? (Gray [?] marker)

class MedicationEntity(BaseModel):
    # A single medication extracted by Claude.
    name: str = Field(description="The canonical name of the medication")
    dosage: Optional[str] = Field(None, description="Dosage amount, e.g., 500mg")
    frequency: Optional[str] = Field(None, description="How often, e.g., twice daily")
    route: Optional[str] = Field(None, description="How it is taken, e.g., oral, injection")
    tavily_verified: bool = Field(description="Must be True if the transcript had a teal underline for this drug")

class ClinicalSummary(BaseModel):
    # The final JSON output of the entire system.
    # We will pass this exact schema to Claude's API to force it to return structured JSON.
    medications: List[MedicationEntity]
    symptoms: List[str] = Field(default_factory=list, description="List of patient symptoms")
    allergies: List[str] = Field(default_factory=list, description="List of patient allergies")
    follow_up_actions: List[str] = Field(default_factory=list, description="Instructions for next steps")
    appointment_needed: bool = Field(description="True if the doctor or patient requested a follow-up visit")

if __name__ == "__main__":
    # This shows how strict and clean Pydantic is.
    sample_med = MedicationEntity(
        name="Metformin", 
        dosage="500mg", 
        tavily_verified=True
    )
    print("Valid Pydantic Model:")
    print(sample_med.model_dump_json(indent=2))