from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import logging

logger = logging.getLogger(__name__)

def create_rag_chain(llm, retriever):
    """
    Composes the RAG chain.
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

    logger.debug("Creating stuff_documents_chain")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    logger.debug("Creating retrieval_chain")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
