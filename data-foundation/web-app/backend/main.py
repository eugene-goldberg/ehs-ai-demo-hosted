from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import data_management, analytics, instructions, langsmith_conversations, risk_assessment_transcript
import uvicorn

app = FastAPI(title="EHS Compliance Platform API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data_management.router, prefix="/api/data", tags=["data"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(instructions.router, prefix="/api", tags=["instructions"])
app.include_router(langsmith_conversations.router, prefix="/api/langsmith", tags=["langsmith-conversations"])
app.include_router(risk_assessment_transcript.router, prefix="/api/risk-assessment-transcript", tags=["risk-assessment"])

@app.get("/")
def read_root():
    return {"message": "EHS Compliance Platform API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
