# models/llm.py
import os
import google.generativeai as genai
from langchain_core.language_models.llms import LLM


class GeminiLLM(LLM):

    def _call(self, prompt, stop=None):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @property
    def _identifying_params(self):
        return {"model": "gemini-1.5-flash"}

    @property
    def _llm_type(self):
        return "gemini"