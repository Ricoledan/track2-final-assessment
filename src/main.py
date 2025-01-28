import init_creds as creds
from chromadb import HttpClient, Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from typing import TypedDict

# Azure OpenAI Credentials
AZURE_OPENAI_KEY = creds.get_api_key()
AZURE_OPENAI_ENDPOINT = creds.get_endpoint()

if not AZURE_OPENAI_KEY:
    raise ValueError("No AZURE_OPENAI_KEY set for Azure OpenAI API")
if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("No AZURE_OPENAI_ENDPOINT set for Azure OpenAI API")

# Initialize LLM and embeddings
llm = AzureChatOpenAI(
    model="gpt-4o-mini",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-07-01-preview"
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-07-01-preview"
)

# Initialize ChromaDB client
client = HttpClient(
    host="localhost",
    port=8000,
    ssl=False,
    headers=None,
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Function to ingest PDF into ChromaDB
def ingest_pdf_to_chromadb(pdf_path):
    """
    Ingest a PDF file into ChromaDB.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        ChromaDB: The ChromaDB instance containing the embeddings.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    for doc in docs:
        chroma_client.add(doc, embeddings.embed_text(doc.text))
    return chroma_client

# Ingest PDF into ChromaDB
pdf_path = '../assets/bitcoin.pdf'
chroma_client = ingest_pdf_to_chromadb(pdf_path)

# Prompt template for rewriting questions
re_write_prompt = PromptTemplate(
    template="""You are an expert prompt engineer that rephrases a user question to make it optimal \n
     for retrieval augmented generation. Rewrite the user question with detailed system instruction,
     clear question and chain of thought: \n\n {question}. """,
    input_variables=["question"]
)

# Function to rewrite a question
def rewrite_question(question):
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter.invoke({"question": question})

# Define Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question (str): Original question.
        updated_question (str): Rewritten question.
        response (dict): Response from the model.
        score (str): Grading score.
    """
    question: str
    updated_question: str
    response: dict
    score: str

# Generate a response from the graph state
def generate_response(state):
    """
    Generate response from the graph state.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        dict: The response generated from the graph state.
    """
    question = state["updated_question"] if state.get("updated_question") else state["question"]
    response = question | llm | StrOutputParser()
    return {'response': response}

# Rewrite the prompt based on the graph state
def prompt_rewrite(state):
    """
    Rewrite the prompt.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated graph state with rewritten question.
    """
    question = state.get("updated_question", state["question"])
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    updated_question = question_rewriter.invoke({"question": question})
    state["updated_question"] = updated_question
    return state

# Grade the response's relevance
def answer_grader(state):
    """
    Determines whether the generated response is relevant to the question.

    Args:
        state (GraphState): The current graph state.

    Returns:
        GraphState: Updated graph state with grading score.
    """
    question = state.get("question")
    answer = state.get("response")
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve a question. \n
        Here is the answer:
        \n ------- \n
        {answer}
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question and no preamble or explanation.""",
        input_variables=["answer", "question"]
    )
    answer_grader = prompt | llm | StrOutputParser()
    score = answer_grader.invoke({"question": question, "answer": answer})
    state["score"] = score
    return state