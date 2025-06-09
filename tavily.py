import os
import getpass
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_tavily import TavilySearch
import json
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END
from langgraph.graph import StateGraph
from IPython.display import Image, display

__package__ = "tavily-langchain"

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
_set_env("TAVILY_API_KEY")
_set_env("LANGCHAIN_API_KEY")

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = "darksouls-chatbot"

web_search_tool = TavilySearch(k=1)

local_llm = ChatOllama(model="llama3.2:3b-instruct-fp16")
llm = ChatOllama(model="llama3.2:3b-instruct-fp16", temperature=0)
llm_json_mode = ChatOllama(model="llama3.2:3b-instruct-fp16", format="json")

urls = [
    "https://docs.google.com/document/d/11dP6pi7OAFWrxG-_DBMQ_qkMblwxn4iIK-ghjDgjd9M/edit?tab=t.0",
    "https://docs.google.com/document/d/11dP6pi7OAFWrxG-_DBMQ_qkMblwxn4iIK-ghjDgjd9M/edit?tab=t.0",
    "https://docs.google.com/document/d/11dP6pi7OAFWrxG-_DBMQ_qkMblwxn4iIK-ghjDgjd9M/edit?tab=t.0",
    "https://docs.google.com/document/d/11dP6pi7OAFWrxG-_DBMQ_qkMblwxn4iIK-ghjDgjd9M/edit?tab=t.0",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [doc for sublist in docs for doc in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)

retriever = vectorstore.as_retriever(k=1)

retriever.invoke("agent memory")

router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
the vectorstore contains documents about the game Dark Souls.
Use the vectorstore to answer questions about the game.
Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

question =  [HumanMessage(content="What is the lore of Dark Souls?")]
test_vector_store = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + question)
json.loads(test_vector_store.content)

doc_grader_instructions = """You are an expert at grading documents based on their relevance to a user question.
if the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

doc_grader_prompt = """Here is the retrieved document: \n\n{document}\n\n
Here is the user question: \n\n{question}.
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

rag_prompt = """You are an assistant for question-answering tasks.
Here is the context to use to answer the question: {context}
Think carefully about the above context.
Now,review the user question:{question}
Provide an answer to this question using only the above context.
Use three sentences maximum and keep the answer concise.
Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

hallucination_grader_instructions = """You are a strict grader of factual accuracy.

You will receive a FACTS section and a STUDENT ANSWER.
Return ONLY a JSON in the following format:

{"binary_score": "yes"}    ← if the student's answer is fully grounded in the facts
{"binary_score": "no"}     ← if the answer contains hallucinated or unsupported content

DO NOT include any explanation. DO NOT add any text outside the JSON.
Only respond with the JSON object.
"""

hallucination_grader_prompt = """FACTS: \n\n {docs} \n\n STUDENT ANSWER: {generated_answer}"""

class GraphState (TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question: str # User question
    generated_answer: str # Generated answer
    web_search: str # Binary decision to run web search
    max_retries: int # Max number of retries for answer generation
    answers: int # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str] # List of retrived documents

def retrieve(state):
    """Retrieve documents from vectorstore
    
    Args:
        state (dict): The current graph state.
    
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    """
    Generate answer using RAG on retrieved documents
    
    Args:
        state (dict): The current graph state.
        
    Returns:
        state (dict): New key added to state, generated_answer, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generated_answer = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generated_answer": generated_answer, "loop_step": loop_step + 1}

def grade_documents(state):
    """Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search.
    Args:
        state (dict): The current graph state.
    
    Returns:
        state (dict): Filtered out irrelevant documents and update web_search state.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = llm_json_mode.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grader_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state):
    """Web search based on the question
    
    Args:
        state (dict): The current graph state.
    
    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    
    # Mantém os documentos anteriores (se houver)
    previous_docs = state.get("documents", [])

    # Faz a busca
    new_results = web_search_tool.invoke({"query": question})

    # Garante que seja uma lista de documentos
    if isinstance(new_results, dict):
        new_docs = [Document(page_content=new_results.get("content", str(new_results)))]
    elif isinstance(new_results, list):
        new_docs = [
            Document(page_content=doc.get("content", str(doc))) if isinstance(doc, dict) else
            doc if isinstance(doc, Document) else
            Document(page_content=str(doc))
            for doc in new_results
        ]
    else:
        new_docs = [Document(page_content=str(new_results))]

    concatenated = "\n".join(doc.page_content for doc in new_docs)
    summary_doc = Document(page_content=concatenated)

    return {
        "documents": previous_docs + new_docs + [summary_doc]
    }

def route_question(state):
    """
    Route question to web search or RAG
    
    Args:
        state (dict): The current graph state.
        
    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke([SystemMessage(content=router_instructions)] + [HumanMessage(content=state["question"])])
    source = json.loads(route_question.content)["datasource"]
    if source == 'websearch':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    docs = state["documents"]
    generated_answer = state["generated_answer"]
    max_retries = state.get("max_retries", 3)

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        docs=format_docs(docs), generated_answer=generated_answer.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATED_ANSWER IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATED_ANSWER vs QUESTION---")
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generated_answer=generated_answer.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATED_ANSWER ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATED_ANSWER DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATED_ANSWER IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"

workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

graph = workflow.compile()
display(Image(graph.get_graph().draw_mermaid_png()))