from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.components.vector_store import load_vector_store
from src.components.llm_setup import load_llm
from src.config.prompts import MEDICAL_QA_PROMPT

def create_qa_chain():
    llm = load_llm()
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 1. The Condensation Step: Formulates a standalone question from history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # This retriever looks at history to improve the search
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. The Answering Step: Answers based on context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", MEDICAL_QA_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Combine them
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)