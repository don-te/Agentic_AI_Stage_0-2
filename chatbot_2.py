import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field, ValidationError # Import all pydantic tools
import sys
import pathlib
from langchain_community.document_loaders import PyPDFLoader # For PDF loading
from reviewer_schema import ReviewOutput # Import the Pydantic schema

# --- 2. Robust JSON Extraction Helper ---
def safe_json_load(text_content: str):
    """
    Safely extracts and loads a JSON object from text that may contain
    conversational filler or markdown code fences (```json).
    """
    # 1. Clean the string to find the JSON structure
    text_content = text_content.strip()
    
    # Common case 1: Model wraps the JSON in markdown fences
    if text_content.startswith('```'):
        # Find the first and last curly brace to isolate the JSON
        start = text_content.find('{')
        end = text_content.rfind('}') + 1
        
        if start != -1 and end != 0:
            json_str = text_content[start:end]
        else:
            # Fallback if braces are not found inside code fences
            json_str = text_content
            
    # Common case 2: Simple conversational text wraps the JSON
    else:
        json_str = text_content.strip()

    # 2. Attempt to load the cleaned JSON string
    return json.loads(json_str)


# --- 3. AGENT SETUP ---
load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

# Configure the LLM client (OpenRouter/DeepSeek)
client = OpenAI(
    base_url="google/gemma-3-27b-it:free",
    api_key=api_key,
)
model_name = "openai/gpt-oss-120b:free"

# Define the Agent's Persona (System Instruction)
system_instruction = """
You are a Senior HR Analyst specializing in modern tech and finance roles. 
Your goal is to meticulously review the provided resume.
Your entire response MUST be a single, raw JSON object that precisely
matches the ReviewOutput Pydantic schema. 
DO NOT include any conversation, greetings, extra text, or markdown code fences (```json) in the final output.
"""

# --- 4. Main Interaction Loop ---

print("--- HR Resume Reviewer Agent Initialized ---")
print("Enter the file path of the resume (e.g., resume.txt or resume.pdf).")
print("Type 'exit' or 'quit' to close the program.")

while True:
    file_path = input("\nFile Path: ").strip()

    if file_path.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    # --- File Handling and Text Extraction (FINAL ROBUST VERSION) ---
    resume_text = ""
    try:
        # Use pathlib to resolve the path reliably across OS
        resolved_path = pathlib.Path(file_path).resolve()
        
        if str(resolved_path).lower().endswith('.pdf'):
            print(f"Loading and extracting text from PDF: {resolved_path}")
            
            # Load the PDF content
            loader = PyPDFLoader(str(resolved_path)) 
            pages = loader.load()
            resume_text = "\n\n".join([page.page_content for page in pages])

        elif str(resolved_path).lower().endswith(('.txt', '.md')):
            # Handle plain text files
            with open(resolved_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()
                
        else:
            print("ERROR: Unsupported file type. Please provide a .pdf, .txt, or .md file.")
            continue
            
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'. Please check the path and try again.")
        continue
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        continue
    
    if not resume_text.strip():
        print("ERROR: The file is empty or no text could be extracted.")
        continue
        
    # --- Agent Execution ---
    print(f"Processing content (First 50 chars: {resume_text[:50].replace('\n', ' ')}...)")

    try:
        # 1. Construct the core prompt
        prompt = f"Critique the following resume text. Focus on technical roles and modern resume standards:\n\n{resume_text}"

        # 2. API call with Structured Output
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2 
        )

        # 3. Validate and print output
        # Use the SAFE EXTRACTION FUNCTION to get clean JSON
        raw_content = response.choices[0].message.content
        json_output = safe_json_load(raw_content)
        
        # Now, attempt to validate the extracted JSON
        validated_review = ReviewOutput(**json_output)

        # --- Print the Structured and Validated Output ---
        print("\n--- VALIDATED HR REVIEW ---")
        print("----------------------------")
        print(f"Final Score: {validated_review.overall_score}/10")
        print(f"Keyword Optimized: {'Yes' if validated_review.is_keyword_optimized else 'No'}")
        print(f"\nFeedback Summary: {validated_review.summary_feedback}")
        print(f"\nActionable Recommendation: {validated_review.top_recommendation}")
        print("----------------------------")
        
    except (json.JSONDecodeError, ValidationError) as e:
        # If the failure is here, the model returned completely unusable JSON.
        print("\n[ERROR] Model failed to generate perfect JSON.")
        print(f"Details: {e}")
        print(f"\nRaw Content: \n{raw_content}") # Show the raw content to see why it failed
    except Exception as e:
        print(f"\nAn unexpected API error occurred: {e}")