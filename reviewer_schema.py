from pydantic import BaseModel, Field

# Define the exact JSON structure for the Agent's output
class ReviewOutput(BaseModel):
    """A detailed review of a resume by a Senior HR Analyst."""
    overall_score: int = Field(
        ..., 
        description="A final score from 1 to 10 (10 being best) based on the resume's strength."
    )
    is_keyword_optimized: bool = Field(
        ..., 
        description="True if the resume uses strong keywords relevant to a modern tech job, False otherwise."
    )
    summary_feedback: str = Field(
        ..., 
        description="A concise, professional summary (3-4 sentences) of the resume's main strengths and weaknesses."
    )
    top_recommendation: str = Field(
        ..., 
        description="The single most important, actionable recommendation for the user to improve the resume."
    )