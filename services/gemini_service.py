"""
Gemini Service for FoodieHelp AI
Interfaces with the official Google Gen AI SDK (google-genai) to generate grounded LLM responses.
Handles offline states, missing API keys, and error recovery gracefully.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.settings import settings


class GeminiService:
    """Manages LLM generation using the official Google Gen AI Python SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self._client: Optional[genai.Client] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Google Gen AI client if API key is present."""
        if self.api_key and settings.is_gemini_configured():
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiService] Warning: Failed to initialize genai.Client: {e}")
                self._client = None
        else:
            self._client = None

    def is_available(self) -> bool:
        """Check if Gemini client is ready and configured."""
        return self._client is not None and settings.is_gemini_configured()

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Generate text content using the configured Gemini model.
        Returns a dictionary with 'content', 'success', 'model', and 'error' (if any).
        """
        if not self.is_available():
            return {
                "success": False,
                "content": (
                    "⚠️ **Gemini API Key Required**\n\n"
                    "Google Gemini API key is not yet configured or is using placeholder credentials.\n\n"
                    "**How to configure:**\n"
                    "1. Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).\n"
                    "2. Open the `.env` file in the project directory and set `GEMINI_API_KEY=your_actual_key`.\n"
                    "3. Alternatively, enter your key in the web app settings modal."
                ),
                "model": self.model,
                "error": "API_KEY_MISSING"
            }

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction if system_instruction else None,
            )

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            text = response.text or ""
            return {
                "success": True,
                "content": text.strip(),
                "model": self.model,
                "error": None
            }

        except APIError as e:
            # Fallback to gemini-3.5-flash-lite if primary model is unavailable or rate-limited
            if self._client and self.model != "gemini-3.5-flash-lite":
                try:
                    response = self._client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt,
                        config=config,
                    )
                    text = response.text or ""
                    return {
                        "success": True,
                        "content": text.strip(),
                        "model": "gemini-3.5-flash-lite",
                        "error": None
                    }
                except Exception:
                    pass

            return {
                "success": False,
                "content": f"⚠️ **Gemini API Error**: {str(e)}",
                "model": self.model,
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "content": f"⚠️ **Unexpected Error generating response**: {str(e)}",
                "model": self.model,
                "error": str(e)
            }


# Global singleton instance
gemini_service = GeminiService()

