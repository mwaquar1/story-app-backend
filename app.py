from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.openai_client import get_completion

# --- FastAPI app ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StoryRequest(BaseModel):
    genre: str
    characters: int
    paragraphs: int
    extraPrompt: str | None = None   # optional field

@app.post("/generate")
def generate_story(req: StoryRequest):
    # Build the prompt
    prompt = (
        f"Write a {req.genre} story with {req.characters} characters in {req.paragraphs} paragraphs. "
    )
    if req.extraPrompt:
        prompt += f"Additional instructions: {req.extraPrompt}"

    try:
        story = get_completion(prompt)
        return {"story": story}
    except Exception as e:
        return {"story": f"Error: {str(e)}"}
