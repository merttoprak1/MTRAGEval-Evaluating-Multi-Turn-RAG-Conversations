from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
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
