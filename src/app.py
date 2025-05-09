from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
import torch 
import pandas as pd
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Optional 
from .config import Config
from .rag_system import  policy_retriever 
from .data_preprocessing import Preprocessor
import uvicorn
import warnings
warnings.filterwarnings("ignore")

#intialize application
app = FastAPI(
    title = "Resume Analysis API",
    version = "2.0",
    description = "AI-powered resume screening with category-aware polycicies"
)

#load models an components 
class Models : 
    def __init__(self):
        Config.setup_dirs()
        self.tokenizer = BertTokenizer.from_pretrained(Config.TOKENIZER_DIR)
        self.model = BertForSequenceClassification.from_pretrained(Config.MODEL_DIR)
        self.scaler = joblib.load(Config.MODEL_DIR / "scaler.joblib")
        self.preprocessor = Preprocessor()
        self.numerical_cols = [
            'text_length', 'num_skills', 'experience_years',
            'features.tech_stack_count', 'features.certification_count',
            'features.finance_terms', 'features.achievement_count'
        ]

models = Models()
#request models 
class ResumeRequest(BaseModel):
    text:  str ="HR"
    check_policies: bool = True
    
class PolicyResponse(BaseModel):
    question: str
    category: Optional[str] = None

#API Endpoints
@app.post("/analyze")
async def analyze_resume(request: ResumeRequest):
    try:
        #1. Preprocess input 
        df = pd.DataFrame([
            {
                "Resume_str": request.text,
                "category": request.category
            }
        ])
        processed = models.preprocessor.process(df)
        
        #2. Prepare numerical features
        numerical_features = models.scaler0transform(
            processed[models.numerical_cols].fillna(0)
        )
        #3. tokenize text
        inputs = models.tokenizer(
            processed['clean_text'].iloc[0],
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        #4. Predict with both features
        with torch.no_grad():
            outputs = models.model(
                **inputs,
                numerical_features = torch.tensor(numerical_features, dtype=torch.float)
                
            )
        # 5. process results
        score= torch.sigmoid(outputs.logits[0][0]).item()
        is_fraud = torch.sigmoid(outputs.logits[0][1]).item() > 0.5
        
        #6. Check policies if requested
        policy_check = {}
        if request.chekc_policies:
            policy_check = policy_retriever.query(
                f"Verify compilance for {request.category} role",
                category = request.category
            )
        
        return {
            "score": round(score,2),
            "is_fraud": bool(is_fraud),
            "category": request.category,
            "policy_compliance": policy_check.get("answer", "Not checked"),
            "flagged_issues": self._detect_issues(processed.iloc[0], score)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/policy")
async def get_policy(request: PolicyRequest):
    try:
        return policy_retriever.query(request.question, request.category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper methods
def _detect_issues(self, row, score):
    issues = []
    if row['is_fraud']:
        issues.append("Potential resume fraud detected")
    if score < 0.5:
        issues.append("Low qualification score")
    if row['experience_years'] < 2:
        issues.append("Insufficient experience")
    return issues

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)       