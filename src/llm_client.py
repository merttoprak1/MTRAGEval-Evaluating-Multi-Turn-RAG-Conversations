from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(provider: str, api_key: str = None, base_url: str = None, model_name: str = "gpt-3.5-turbo"):
    """
    Factory function to return the appropriate LLM client.
    
    Args:
        provider: "OpenAI" or "Local"
        api_key: API Key for OpenAI or Local server (if required)
        base_url: Base URL for Local server (e.g., "http://localhost:11434/v1")
        model_name: Model name to use
    """
    if provider == "OpenAI":
        if not api_key:
            raise ValueError("API Key is required for OpenAI provider.")
        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0
        )
    
    elif provider == "Gemini":
        if not api_key:
            raise ValueError("API Key is required for Gemini provider.")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=0
        )

    elif provider == "Local":
        if not base_url:
            raise ValueError("Base URL is required for Local provider.")
        
        # Local LLMs usually follow OpenAI API format (e.g. Ollama, LM Studio)
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "not-needed", # Some local servers might need a dummy key
            model=model_name,
            temperature=0
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
