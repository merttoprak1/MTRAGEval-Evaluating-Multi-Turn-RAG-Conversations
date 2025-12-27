from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(llm, retriever):
    """
    Composes the RAG chain using LCEL.
    Returns a dict with 'answer' and 'context'.
    """
    logger.info("Initializing RAG chain components")
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Chain to generate answer from context and input
    question_answer_chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    
    # RAG Chain that retrieves context, then generates answer
    # We use RunnablePassthrough.assign to add 'context' to the existing input (which contains 'input')
    # Note: retrieval needs to receive just the query string, not the whole dict
    dag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever
        )
        .assign(answer=lambda x: question_answer_chain.invoke({
            "context": format_docs(x["context"]),
            "input": x["input"]
        }))
    )
    
    return dag_chain


# ==================== Multi-Turn History Support ====================

def convert_mtrag_history_to_messages(input_list: List[Dict[str, str]]) -> tuple:
    """
    Convert MTRAG input format to LangChain messages.
    
    MTRAG format (input field in reference.jsonl):
    [
        {"speaker": "user", "text": "First question"},
        {"speaker": "agent", "text": "First answer"},
        {"speaker": "user", "text": "Follow-up question"}  # Current question
    ]
    
    Returns:
        Tuple of (history_messages, current_question)
        - history_messages: List of HumanMessage/AIMessage for previous turns
        - current_question: The last user message (current question to answer)
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    if not input_list:
        return [], ""
    
    history_messages = []
    current_question = ""
    
    # All but last message go to history
    for i, turn in enumerate(input_list):
        speaker = turn.get("speaker", "user")
        text = turn.get("text", "")
        
        if i == len(input_list) - 1:
            # Last message is the current question
            current_question = text
        else:
            # Previous messages go to history
            if speaker == "user":
                history_messages.append(HumanMessage(content=text))
            else:  # agent/assistant
                history_messages.append(AIMessage(content=text))
    
    return history_messages, current_question


def create_rag_chain_with_history(llm, retriever):
    """
    Create RAG chain with multi-turn conversation history support.
    
    This chain accepts:
    - history: List of previous HumanMessage/AIMessage
    - input: Current user question
    
    Returns a dict with 'answer' and 'context'.
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    logger.info("Initializing RAG chain with history support")
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),  # Previous conversation turns
            ("human", "{input}"),  # Current question
        ]
    )
    
    # Chain to generate answer from context, history, and input
    question_answer_chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    
    # RAG Chain with history support
    dag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever
        )
        .assign(answer=lambda x: question_answer_chain.invoke({
            "context": format_docs(x["context"]),
            "history": x.get("history", []),
            "input": x["input"]
        }))
    )
    
    return dag_chain


def run_rag_with_mtrag_input(
    llm, 
    retriever, 
    mtrag_input: List[Dict[str, str]],
    use_history: bool = True
) -> Dict[str, Any]:
    """
    Run RAG chain with MTRAG format input.
    
    Args:
        llm: Language model
        retriever: Vector store retriever
        mtrag_input: MTRAG input list from reference.jsonl
        use_history: Whether to use conversation history
        
    Returns:
        Dictionary with 'answer', 'context', and 'history_used'
    """
    from langchain_core.documents import Document
    
    # Convert MTRAG format to messages
    history_messages, current_question = convert_mtrag_history_to_messages(mtrag_input)
    
    if use_history and history_messages:
        # Use history-aware chain
        chain = create_rag_chain_with_history(llm, retriever)
        result = chain.invoke({
            "input": current_question,
            "history": history_messages
        })
    else:
        # Use simple chain without history
        chain = create_rag_chain(llm, retriever)
        result = chain.invoke({"input": current_question})
    
    return {
        "answer": result.get("answer", ""),
        "context": result.get("context", []),
        "history_used": use_history and len(history_messages) > 0,
        "history_turns": len(history_messages)
    }

