"""
Query Rewrite Module for RAG Pipeline.

This module provides query rewriting capabilities to improve retrieval quality.
Supports LLM-based (Contextual & Simple), rule-based, and hybrid rewriting methods.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import re
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ==================== Prompts ====================

# Default prompt: Used when len(history) == 0
DEFAULT_REWRITE_PROMPT = """You are a query rewriting assistant. Your task is to polish the user query to be effective for a semantic search engine.

Guidelines:
- **Expand Acronyms:** If a clear acronym is used (e.g., "US"), expand it (e.g., "United States").
- **Fix Ambiguity:** Ensure the query is grammatically complete.
- **Do NOT Answer:** Do not generate an answer, just rewrite the query.
- **Be Concise:** Do not add unnecessary synonyms or fluff.

Current Query: {query}

Respond ONLY with the rewritten query, nothing else."""

# Contextual prompt: Used when len(history) > 0
CONTEXTUAL_REWRITE_PROMPT = """You are an expert Query Rewriter for a Retrieval-Augmented Generation system. 
Your task is to rewrite the "Current User Query" into a **Standalone Query** that is fully descriptive and can be understood without any conversation history.

**CRITICAL INSTRUCTIONS:**
1. **Resolve Pronouns:** Replace words like "it", "he", "they", "that", "this" with the specific entity names from the History (User or Agent text).
2. **Carry Over Predicates:** If the user asks "What about X?", apply the *action* or *question* from the previous turn to entity X.
3. **Resolve Agent References:** If the user refers to an entity introduced by the Agent (e.g., "that last one"), identify the specific entity from the Agent's response.
4. **Maintain Constraints:** Keep specific dates, seasons, or locations mentioned in previous turns if they limit the scope (e.g., "2017 season").
5. **Handle Corrections:** If the user says "No, I meant X", rewrite the query to focus entirely on X, ignoring the mistake.

**EXAMPLES:**
History:
User: Who is the CEO of Apple?
Agent: Tim Cook.
Query: What about Microsoft?
Rewrite: Who is the CEO of Microsoft?

History:
User: Tell me about the 2017 Arizona Cardinals season.
Agent: [Details about 2017 games...]
Query: Did they play in London?
Rewrite: Did the Arizona Cardinals play in London during the 2017 season?

History:
User: What films is Doctor Strange in?
Agent: He appears in Doctor Strange, Thor: Ragnarok, and Avengers: Infinity War.
Query: When was that last one released?
Rewrite: When was the film Avengers: Infinity War released?

History:
User: Price of the Pixel 7.
Query: No, I meant the Pixel 8.
Rewrite: Price of the Pixel 8.

**YOUR TASK:**
Conversation History:
{history}

Current Query: {query}

Respond ONLY with the rewritten standalone query, nothing else."""


# ==================== Helper Functions ====================

def format_history(history: List[Dict[str, str]], limit: int = 3) -> str:
    """
    Format a list of message dictionaries into a string for the prompt.
    Limits to the last N turns to fit context window.
    
    Args:
        history: List of dicts like [{"speaker": "user", "text": "..."}, ...]
        limit: Number of recent turns to include.
    
    Returns:
        Formatted string: "User: ... \nAgent: ..."
    """
    if not history:
        return ""
        
    # Take the last N turns
    relevant_turns = history[-limit:]
    formatted_lines = []
    
    for turn in relevant_turns:
        role = turn.get("speaker", "user").capitalize() # "user" -> "User", "agent" -> "Agent"
        text = turn.get("text", "").strip()
        formatted_lines.append(f"{role}: {text}")
        
    return "\n".join(formatted_lines)


def _build_rewrite_prompt(prompt_text: str) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate from a string."""
    return ChatPromptTemplate.from_template(prompt_text)


# ==================== LLM-based Rewrite ====================

def rewrite_query_llm(
    query: str, 
    llm, 
    custom_prompt: str = None,
    history: List[Dict[str, str]] = None
) -> str:
    """
    Rewrite query using LLM for better retrieval, optionally using history.
    
    Args:
        query: Original user query
        llm: LangChain LLM instance
        custom_prompt: Optional custom system prompt
        history: Optional list of conversation turns
    
    Returns:
        Rewritten query string
    """
    try:
        # Determine if we have history to work with
        has_history = history is not None and len(history) > 0
        
        # Select Prompt Strategy
        if custom_prompt:
            prompt_text = custom_prompt
            # Detect if custom prompt expects history variable
            uses_history = "{history}" in custom_prompt
        elif has_history:
            prompt_text = CONTEXTUAL_REWRITE_PROMPT
            uses_history = True
        else:
            prompt_text = DEFAULT_REWRITE_PROMPT
            uses_history = False
            
        prompt = _build_rewrite_prompt(prompt_text)
        chain = prompt | llm | StrOutputParser()
        
        # Prepare inputs
        inputs = {"query": query}
        if uses_history:
            # Format history list into string
            history_str = format_history(history) if has_history else "No history."
            inputs["history"] = history_str
            
        rewritten = chain.invoke(inputs)
        
        # Log result slightly differently based on context usage
        log_prefix = "LLM Contextual Rewrite" if uses_history else "LLM Simple Rewrite"
        logger.info(f"{log_prefix}: '{query}' -> '{rewritten.strip()}'")
        
        return rewritten.strip()
        
    except Exception as e:
        logger.error(f"LLM rewrite failed: {e}")
        return query  # Fallback to original


# ==================== Rule-based Rewrite ====================

# Common expansions and corrections
EXPANSIONS = {
    "rag": "retrieval augmented generation RAG",
    "llm": "large language model LLM",
    "ml": "machine learning ML",
    "ai": "artificial intelligence AI",
    "nlp": "natural language processing NLP",
    "api": "application programming interface API",
    "db": "database",
    "docs": "documents documentation",
    "info": "information",
}

STOPWORDS_TO_REMOVE = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}


def rewrite_query_rules(query: str) -> str:
    """
    Rewrite query using rule-based transformations.
    Note: Rules operate only on the current query, ignoring history.
    
    Args:
        query: Original user query
    
    Returns:
        Rewritten query string
    """
    original = query
    query_lower = query.lower()
    
    # Expand known abbreviations
    words = query_lower.split()
    expanded_words = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in EXPANSIONS:
            expanded_words.append(EXPANSIONS[clean_word])
        else:
            expanded_words.append(word)
    
    # Remove some stopwords for cleaner search
    filtered_words = [w for w in expanded_words if w.lower() not in STOPWORDS_TO_REMOVE]
    
    rewritten = " ".join(filtered_words) if filtered_words else query
    
    logger.info(f"Rule Rewrite: '{original}' -> '{rewritten}'")
    return rewritten


# ==================== Hybrid Rewrite ====================

def rewrite_query_hybrid(
    query: str, 
    llm, 
    custom_prompt: str = None,
    history: List[Dict[str, str]] = None
) -> str:
    """
    Hybrid approach: Apply rules first, then LLM refinement.
    
    Args:
        query: Original user query
        llm: LangChain LLM instance
        custom_prompt: Optional custom system prompt for LLM step
        history: Optional list of conversation turns
    
    Returns:
        Rewritten query string
    """
    # Step 1: Rule-based preprocessing
    # We always apply rules to the current query string regardless of history
    rule_rewritten = rewrite_query_rules(query)
    
    # Step 2: LLM refinement (Contextual or Simple)
    final = rewrite_query_llm(rule_rewritten, llm, custom_prompt, history)
    
    logger.info(f"Hybrid Rewrite: '{query}' -> '{final}'")
    return final


# ==================== Main Rewrite Interface ====================

def rewrite_query(
    query: str, 
    method: str = "LLM-based", 
    llm=None, 
    enabled: bool = True,
    custom_prompt: str = None,
    history: List[Dict[str, str]] = None
) -> dict:
    """
    Main interface for query rewriting.
    
    Args:
        query: Original user query
        method: Rewriting method - "LLM-based", "Rule-based", or "Hybrid"
        llm: LangChain LLM instance (required for LLM-based and Hybrid)
        enabled: Whether rewriting is enabled
        custom_prompt: Optional custom system prompt
        history: Optional list of conversation turns [{"speaker": "user", "text": "..."}]
    
    Returns:
        Dict with 'original', 'rewritten', 'method', 'enabled', and 'prompt_used' keys
    """
    prompt_used = "custom" if custom_prompt else ("contextual" if history else "default")
    
    result = {
        "original": query,
        "rewritten": query,
        "method": method,
        "enabled": enabled,
        "prompt_used": prompt_used
    }
    
    if not enabled:
        logger.info("Query rewrite disabled, returning original query")
        return result
    
    try:
        if method == "LLM-based":
            if llm is None:
                logger.warning("LLM not provided for LLM-based rewrite, falling back to original")
                return result
            result["rewritten"] = rewrite_query_llm(query, llm, custom_prompt, history)
        
        elif method == "Rule-based":
            result["rewritten"] = rewrite_query_rules(query)
        
        elif method == "Hybrid":
            if llm is None:
                logger.warning("LLM not provided for Hybrid rewrite, using Rule-based only")
                result["rewritten"] = rewrite_query_rules(query)
                result["method"] = "Rule-based (fallback)"
            else:
                result["rewritten"] = rewrite_query_hybrid(query, llm, custom_prompt, history)
        
        else:
            logger.warning(f"Unknown rewrite method: {method}")
    
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
    
    return result