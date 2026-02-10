from groq import Groq
import os
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

print(os.getenv("LANGCHAIN_API_KEY"))

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# --------------------
# Embeddings
# --------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5"
)

# --------------------
# Vector Store
# --------------------
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="learing_rag"
)

def process_query(query:str):
    print("printing_query",query)
    search_results = vectorstore.similarity_search(
    query=query,
    k=3)

    context = "\n\n---\n\n".join(
    f"Page Content:\n{r.page_content}\n"
    f"Page Number: {r.metadata.get('page_label', 'N/A')}\n"
    f"File Location: {r.metadata.get('source', 'N/A')}"
    for r in search_results)

    SYSTEM_PROMPT = f"""
        You are a helpful assistant.

        Answer the user's question using ONLY the information provided in the context below. 
        and page number as well to more knowledge.

        Rules:
        - Do NOT use outside knowledge
        - Do NOT add explanations beyond the context
        - do not add file locations just give page numbers
        - If the answer is not present, say:
        "I don't have enough information in the provided documents."

        Context:
        {context}
        """
    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    temperature=0,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ])
    print(f"\n🤖 {response.choices[0].message.content}\n")
    return response.choices[0].message.content