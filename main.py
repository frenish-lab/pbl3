import re
from io import BytesIO
from typing import List, Optional
from enum import Enum
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Humanitarian Relief Helper API",
    description="An API that preprocesses emergency aid requests and prioritizes help for people in need.",
    version="1.0.0"
)


# =====================================================================
# DATA MODELS (Pydantic Schemas)
# =====================================================================

class ReliefCategory(str, Enum):
    MEDICAL = "Medical Assistance"
    FOOD = "Food & Water"
    SHELTER = "Shelter & Clothing"
    RESCUE = "Emergency Rescue"
    GENERAL = "General Support"


class UnprocessedAidRequest(BaseModel):
    applicant_name: str = Field(..., example="  john DOE ")
    contact_number: str = Field(..., example="+1 (555)-019-2831")
    description: str = Field(..., example="URGENT!! We need drinking water and blankets near block 4. Please HELP!")
    requested_amount: Optional[float] = Field(default=0.0, example=250.00)


class ProcessedAidRequest(BaseModel):
    clean_name: str
    clean_phone: str
    category: ReliefCategory
    is_urgent: bool
    cleaned_description: str
    approved_priority_score: int  # 1 (Low) to 5 (Critical)


# =====================================================================
# PREPROCESSING HELPER FUNCTIONS
# =====================================================================

def clean_text(text: str) -> str:
    """Removes special characters, extra spaces, and normalizes casing."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Remove multiple spaces
    return text.strip().lower()

def extract_phone_digits(phone: str) -> str:
    """Extracts standardized numeric digits from phone numbers."""
    return re.sub(r'\D', '', phone)

def categorize_and_prioritize(description: str) -> tuple[ReliefCategory, bool, int]:
    """Analyzes text to extract emergency urgency and category."""
    desc_lower = description.lower()
    
    is_urgent = any(word in desc_lower for word in ["urgent", "emergency", "asap", "critical", "help"])
    
    # Simple NLP Keyword Categorization
    if any(w in desc_lower for w in ["hospital", "doctor", "medicine", "injured", "blood"]):
        category = ReliefCategory.MEDICAL
        priority = 5 if is_urgent else 4
    elif any(w in desc_lower for w in ["trapped", "flood", "rescue", "fire", "evacuate"]):
        category = ReliefCategory.RESCUE
        priority = 5
    elif any(w in desc_lower for w in ["food", "water", "ration", "hungry", "starving"]):
        category = ReliefCategory.FOOD
        priority = 4 if is_urgent else 3
    elif any(w in desc_lower for w in ["shelter", "blanket", "tent", "clothes", "house"]):
        category = ReliefCategory.SHELTER
        priority = 3
    else:
        category = ReliefCategory.GENERAL
        priority = 2 if is_urgent else 1

    return category, is_urgent, priority


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/")
def home():
    """Health check & welcoming endpoint."""
    return {
        "service": "Humanitarian Relief Helper API",
        "status": "Online",
        "documentation": "/docs"
    }


@app.post("/api/v1/preprocess-request", response_model=ProcessedAidRequest, status_code=status.HTTP_200_OK)
def process_single_request(request: UnprocessedAidRequest):
    """
    **Endpoint 1:** Preprocesses a single emergency request.
    Cleans phone numbers, formats applicant names, categorizes aid types, 
    and assigns priority levels.
    """
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Request description cannot be empty.")

    # 1. Preprocess inputs
    formatted_name = request.applicant_name.strip().title()
    clean_phone = extract_phone_digits(request.contact_number)
    category, is_urgent, priority_score = categorize_and_prioritize(request.description)
    cleaned_desc = clean_text(request.description)

    # 2. Return structured, actionable data
    return ProcessedAidRequest(
        clean_name=formatted_name,
        clean_phone=clean_phone,
        category=category,
        is_urgent=is_urgent,
        cleaned_description=cleaned_desc,
        approved_priority_score=priority_score
    )


@app.post("/api/v1/batch-preprocess-csv")
async def process_batch_csv(file: UploadFile = File(...)):
    """
    **Endpoint 2:** Accepts a CSV file containing multiple raw help requests, 
    cleans the data in bulk using Pandas, and outputs structured relief records.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    
    try:
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")

    # Ensure required columns exist
    required_cols = ["applicant_name", "contact_number", "description"]
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(
            status_code=400, 
            detail=f"CSV must contain columns: {', '.join(required_cols)}"
        )

    # Preprocessing Data Pipeline using Pandas
    df["applicant_name"] = df["applicant_name"].astype(str).str.strip().str.title()
    df["clean_phone"] = df["contact_number"].astype(str).apply(extract_phone_digits)
    
    # Process categories & priority
    processed_tuples = df["description"].astype(str).apply(categorize_and_prioritize)
    df["category"] = [t[0].value for t in processed_tuples]
    df["is_urgent"] = [t[1] for t in processed_tuples]
    df["priority_score"] = [t[2] for t in processed_tuples]
    
    # Sort by priority score (Critical first)
    df = df.sort_values(by="priority_score", ascending=False)

    return {
        "filename": file.filename,
        "total_requests_processed": len(df),
        "urgent_cases_count": int(df["is_urgent"].sum()),
        "processed_data": df.to_dict(orient="records")
    }