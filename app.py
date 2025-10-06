from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.openai_client import get_completion, generate_image_base64

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
    extraPrompt: str | None = None
    generateImages: bool = False

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
        print("Generated Story")
        paragraphs = story.split("\n\n")
        images = []
        print("Image Generation: ", req.generateImages)
        if req.generateImages:
            for p in paragraphs:
                img_prompt = f"Illustration for: {p}"
                img_b64 = generate_image_base64(img_prompt)
                images.append(img_b64)

        return {"story": story, "images": images}
    except Exception as e:
        return {"story": f"Error: {str(e)}"}
