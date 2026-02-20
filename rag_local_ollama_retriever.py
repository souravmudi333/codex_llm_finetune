import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

DB_DIR = "chroma_db"
COLLECTION_NAME = "local_rag_demo"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:0.5b"


def get_retriever():
    print("[get_retriever] Start")
    print(f"[get_retriever] Initializing embeddings: {EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print(f"[get_retriever] Connecting to Chroma: db={DB_DIR}, collection={COLLECTION_NAME}")
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    total_indexes = db._collection.count()
    print(f"[get_retriever] Total indexes in vector DB: {total_indexes}")
    print("[get_retriever] Retriever ready")
    return db.as_retriever(search_kwargs={"k": 3})


def answer_question(question: str):
    print("[answer_question] Start")
    print(f"[answer_question] User question: {question}")
    retriever = get_retriever()
    print(f"[answer_question] Initializing LLM: {LLM_MODEL}")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    print("[answer_question] Building prompt template")
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant.\n"
        "Answer only from the context below.\n"
        "If the answer is not in the context, say: I don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    print("[answer_question] Retrieving relevant chunks")
    docs = retriever.invoke(question)
    print(f"[answer_question] Retrieved chunks: {len(docs)}")
    print("[answer_question] Building context from retrieved chunks")
    context = "\n\n".join([d.page_content for d in docs])
    print("[answer_question] Calling LLM with retrieved context")
    response = (prompt | llm).invoke({"context": context, "question": question})

    print("\nAnswer:")
    print(response.content)
    print("\nRetrieved sources:")
    for i, d in enumerate(docs, start=1):
        print(f"{i}. {d.metadata}")

    print("\nRetrieved chunks:")
    for i, d in enumerate(docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata: {d.metadata}")
        print(d.page_content)
    print("[answer_question] Completed")


def interactive_loop():
    print("[interactive_loop] Start")
    print("Type your question. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            print("[interactive_loop] Empty input, waiting for next question")
            continue
        if q.lower() in {"exit", "quit"}:
            print("[interactive_loop] Exit command received")
            break
        answer_question(q)
    print("[interactive_loop] Completed")


if __name__ == "__main__":
    print("[main] Start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Ask a single question.")
    args = parser.parse_args()

    if args.question:
        print("[main] Single-question mode")
        answer_question(args.question)
    else:
        print("[main] Interactive mode")
        interactive_loop()
    print("[main] Completed")
