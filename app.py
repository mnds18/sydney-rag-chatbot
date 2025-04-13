# --- Imports ---
from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests
import os

# --- Configuration ---
FAISS_FOLDER = "D:/vs_code/llm_personalized_chatbot/faiss_db_folder"
WEBSITE_TEXT_FILE = "D:/vs_code/llm_personalized_chatbot/website_text.txt"
TARGET_URL = "https://en.wikipedia.org/wiki/Sydney"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Step 1: Scrape website and save text ---
def scrape_and_save_website():
    print(f"🌐 Scraping website: {TARGET_URL}")
    response = requests.get(TARGET_URL)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ""

        for paragraph in soup.find_all('p'):
            text += paragraph.get_text()

        os.makedirs(os.path.dirname(WEBSITE_TEXT_FILE), exist_ok=True)
        with open(WEBSITE_TEXT_FILE, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        print("✅ Website text extracted and saved successfully!")
    else:
        raise Exception(f"❌ Failed to retrieve website content. Status code: {response.status_code}")

# --- Step 2: Load or Create FAISS vectorstore ---
def load_or_create_vectorstore():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(FAISS_FOLDER):
        print("✅ FAISS vectorstore found. Loading from disk...")
        vectorstore = FAISS.load_local(
            FAISS_FOLDER,
            embeddings=embedding_function,
            allow_dangerous_deserialization=True
        )
    else:
        print("⚡ No FAISS vectorstore found. Scraping and creating new one...")

        scrape_and_save_website()

        with open(WEBSITE_TEXT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            prompt_text = f.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len,
        )

        chunks = splitter.split_text(prompt_text)
        print(f"✅ Website text split into {len(chunks)} chunks.")

        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_function
        )

        os.makedirs(FAISS_FOLDER, exist_ok=True)
        vectorstore.save_local(FAISS_FOLDER)
        print("✅ New FAISS vectorstore saved to disk.")

    return vectorstore

# --- Step 3: Initialize vectorstore ---
vectorstore = load_or_create_vectorstore()

# --- Step 4: Build the Prompt Template ---
hotel_assistant_template = """
You are the tourist guide for Sydney, named "Mr. Mrig Sydney".
You only provide information related to Sydney.
If a question is not about Sydney, respond with: "I can't assist you with that, sorry!"

Use the following context to help you answer:

{context}

Question: {question}

Answer:
"""

hotel_assistant_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=hotel_assistant_template
)

# --- Step 5: Initialize LLM ---
llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0
)

llm_chain = hotel_assistant_prompt_template | llm

# --- Step 6: Retrieval and Query Functions ---
def get_top_chunks(user_question, top_k=3):
    docs_and_scores = vectorstore.similarity_search_with_score(user_question, k=top_k)
    top_chunks = [doc.page_content for doc, _ in docs_and_scores]
    return top_chunks

def query_llm(question):
    top_chunks = get_top_chunks(question)
    context_text = "\n\n".join(top_chunks)

    result = llm_chain.invoke({
        'context': context_text,
        'question': question
    })

    return result.content

# --- Step 7: Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")  # Make sure templates/index.html exists

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    response_text = query_llm(question)
    return jsonify({"response": response_text})

# --- Step 8: Run the app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Support cloud deployment
    print("✅ Mr. Mrig Sydney Assistant ready! (Launching web server...)")
    app.run(host="0.0.0.0", port=port, debug=True)
