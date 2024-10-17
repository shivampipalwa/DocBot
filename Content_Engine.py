import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import tempfile
import os, time
from langchain.text_splitter import RecursiveCharacterTextSplitter

model_dir = 'weights'
model_name = "deepset/roberta-base-squad2"
tokenizer = None
model = None

def load_models():
    global tokenizer, model
    if tokenizer is None or model is None:
        if os.path.exists(model_dir):
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

def combine_pdf_files(input_files, output_file):
    with open(output_file, 'wb') as output_pdf:
        for input_file in input_files:
            output_pdf.write(input_file.getbuffer())
            output_pdf.write(b"\n\n")  # Optional separator between files

def handle_upload(uploaded_files):
    if uploaded_files:
        combined_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        combine_pdf_files(uploaded_files, combined_file)
        uploaded_filename = combined_file

        data = load_document(uploaded_filename)
        chunks = chunk_data(data, chunk_size=1024)  # Adjust chunk size as needed

        vector_store = create_embeddings(chunks)
        st.write('File uploaded, chunked, and embedded successfully.')

        st.session_state.vector_store = vector_store
        st.session_state.uploaded_files = uploaded_files  # Store uploaded files

def handle_question(question):
    if question and 'vector_store' in st.session_state:
        answer = ask_and_get_answer(st.session_state.vector_store, question)

        st.session_state.history.append({"question": question, "answer": answer})
        return answer

def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        st.error('Document format is not supported!')
        return None

    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=1):
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    documents = retriever.get_relevant_documents(q)

    context = " ".join([doc.page_content for doc in documents])

    load_models()

    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    result = qa_pipeline(question=q, context=context)

    return result['answer']

def response_generator(answer):
    response = answer
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if __name__ == "__main__":
    st.title('Document Processing and QA Chatbot')

    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    st.header('Upload Documents')
    upload = st.file_uploader('Upload PDF documents', type=['pdf'], accept_multiple_files=True)
    if upload and not st.session_state.uploaded_files:
        handle_upload(upload)
    elif not upload and st.session_state.uploaded_files:
        st.session_state.uploaded_files = []  # Reset uploaded files if none are currently uploaded

    st.header('Chat with us')
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(handle_question(prompt)))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
