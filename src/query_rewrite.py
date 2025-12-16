"""
Query Rewrite Module for RAG Pipeline.

This module provides query rewriting capabilities to improve retrieval quality.
Supports LLM-based, rule-based, and hybrid rewriting methods.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import re

logger = logging.getLogger(__name__)


# ==================== LLM-based Rewrite ====================

# Default rewrite prompt text (exported for UI display)
DEFAULT_REWRITE_PROMPT = """You are a query rewriting assistant. Your task is to rewrite user queries to be more effective for semantic search in a document retrieval system.

Guidelines:
- Expand abbreviations and acronyms
- Add relevant synonyms or related terms
- Make implicit context explicit
- Fix typos and grammatical issues
- Keep the original intent intact

Respond ONLY with the rewritten query, nothing else."""


def _build_rewrite_prompt(system_prompt: str) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate from a system prompt string."""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])


def rewrite_query_llm(query: str, llm, custom_prompt: str = None) -> str:
    """
    Rewrite query using LLM for better retrieval.
    
    Args:
        query: Original user query
        llm: LangChain LLM instance
        custom_prompt: Optional custom system prompt (uses DEFAULT_REWRITE_PROMPT if None)
    
    Returns:
        Rewritten query string
    """
    try:
        prompt_text = custom_prompt if custom_prompt else DEFAULT_REWRITE_PROMPT
        prompt = _build_rewrite_prompt(prompt_text)
        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({"query": query})
        logger.info(f"LLM Rewrite: '{query}' -> '{rewritten}'")
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

def rewrite_query_hybrid(query: str, llm, custom_prompt: str = None) -> str:
    """
    Hybrid approach: Apply rules first, then LLM refinement.
    
    Args:
        query: Original user query
        llm: LangChain LLM instance
        custom_prompt: Optional custom system prompt for LLM step
    
    Returns:
        Rewritten query string
    """
    # Step 1: Rule-based preprocessing
    rule_rewritten = rewrite_query_rules(query)
    
    # Step 2: LLM refinement
    final = rewrite_query_llm(rule_rewritten, llm, custom_prompt)
    
    logger.info(f"Hybrid Rewrite: '{query}' -> '{final}'")
    return final


# ==================== Main Rewrite Interface ====================

def rewrite_query(
    query: str, 
    method: str = "LLM-based", 
    llm=None, 
    enabled: bool = True,
    custom_prompt: str = None
) -> dict:
    """
    Main interface for query rewriting.
    
    Args:
        query: Original user query
        method: Rewriting method - "LLM-based", "Rule-based", or "Hybrid"
        llm: LangChain LLM instance (required for LLM-based and Hybrid)
        enabled: Whether rewriting is enabled
        custom_prompt: Optional custom system prompt (uses DEFAULT if None)
    
    Returns:
        Dict with 'original', 'rewritten', 'method', 'enabled', and 'prompt_used' keys
    """
    prompt_used = "custom" if custom_prompt else "default"
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
            result["rewritten"] = rewrite_query_llm(query, llm, custom_prompt)
        
        elif method == "Rule-based":
            result["rewritten"] = rewrite_query_rules(query)
        
        elif method == "Hybrid":
            if llm is None:
                logger.warning("LLM not provided for Hybrid rewrite, using Rule-based only")
                result["rewritten"] = rewrite_query_rules(query)
                result["method"] = "Rule-based (fallback)"
            else:
                result["rewritten"] = rewrite_query_hybrid(query, llm, custom_prompt)
        
        else:
            logger.warning(f"Unknown rewrite method: {method}")
    
    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
    
    return result
