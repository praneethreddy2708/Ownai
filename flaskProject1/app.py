from flask import Flask, request, render_template, jsonify
import openai
import os
from transformers import AutoTokenizer, AutoModel
import torch
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = Flask(__name__)


# Placeholder function for splitting documents into sections
def split_document(document_content):
    # Here, you would implement logic to split the document into manageable parts
    # For example, by paragraphs or sections
    loader = TextLoader(' ./state-of-the-unton-23.txt')
    documents = loader.load()

    sections = document_content.split('\n\n')
    return ["Section 1 content", "Section 2 content", ...]


# Placeholder function for embedding document sections
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")


def embed_sections(sections):
    # Here, you would implement logic to convert sections into embeddings
    # This step is crucial if you're planning on doing similarity searches or other operations with embeddings
    embeddings = []
    for section in sections:
        inputs = tokenizer(section, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return embeddings
    # return ["Embedding for Section 1", "Embedding for Section 2", ...]


# Placeholder function to save data to Chroma DB
def save_to_db(document_id, sections, embeddings, chroma_db_client=None):
    # Implement your database saving logic here
    # This might involve saving each section along with its embedding to Chroma DB
    for i, (section, embedding) in enumerate(zip(sections, embeddings)):
        section_id = f"{document_id}_section_{i}"
        data = {"text": section, "embedding": embedding.tolist()}  # Convert numpy array to list for storage
        chroma_db_client.save(section_id, data)
    pass


def get_document_content():
    # Placeholder: Fetch the document content from Chroma DB
    return "This is the document content that you fetched from Chroma DB."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['document']
    if file:
        document_content = file.read().decode('utf-8')  # Assuming text document
        sections = split_document(document_content)
        embeddings = embed_sections(sections)
        document_id = "UniqueDocumentID"  # Generate or define your unique document ID
        save_to_db(document_id, sections, embeddings)
        return jsonify({"message": "Document processed and saved successfully"}), 200
    else:
        return jsonify({"message": "No document uploaded"}), 400


@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    # Here, you'll need to fetch the document from Chroma DB and use LangChain/OpenAI to answer
    # This is a placeholder for the logic
    document_content = get_document_content()  # Fetch the document's content

    openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure your OpenAI API key is securely stored and accessed

    response = openai.Completion.create(
        engine="text-davinci-003",  # Check OpenAI documentation for the latest and most suitable model
        prompt=f"{document_content}\n\nQ: {question}\nA:",
        temperature=0.5,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    answer = "This is where the answer will be generated."
    return render_template('index.html', answer=answer)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    app.run(debug=True)
