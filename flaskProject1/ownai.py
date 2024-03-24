from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from chromadb import Chroma
# from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

app = Flask(__name__)
app.secret_key = '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'  # Replace 'your_secret_key' with a real secret key

load_dotenv()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key='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')
    client = Chroma()
    vectorstore = client.create_vectorstore("embeddings")
    vectorstore.insert(embeddings.encode(chunks))
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # if 'pdf_docs' not in request.files:
        #     return redirect(request.url)
        pdf_docs = request.files.getlist('pdf_docs')
        if pdf_docs and all(file.filename.endswith('.pdf') for file in pdf_docs):
            pdf_paths = []
            for file in pdf_docs:
                print("h")
                filename = secure_filename(file.filename)
                filepath = os.path.join('/tmp', filename)
                file.save(filepath)
                pdf_paths.append(filepath)
                print("i")
            raw_text = get_pdf_text(pdf_paths)
            chunks = get_text_chunks(raw_text)
            print("j")
            vectorstore = get_vectorstore(chunks)
            print("k")
            conversation_chain = get_conversation_chain(vectorstore)
            session['conversation'] = conversation_chain
            # return redirect(url_for('success_page'))
            return redirect(url_for('chat'))
    return render_template('upload.html')


@app.route('/success')
def success_page():
    # Perform actions or simply return a success message
    return "Files uploaded successfully!"


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'conversation' not in session:
        return redirect(url_for('upload_files'))

    conversation_chain = session['conversation']
    if request.method == 'POST':
        user_question = request.form['question']
        response = conversation_chain({'question': user_question})
        session['chat_history'] = response['chat_history']
        return render_template('chat.html', chat_history=session['chat_history'])

    return render_template('chat.html', chat_history=[])


if __name__ == '__main__':
    app.run(debug=True)
