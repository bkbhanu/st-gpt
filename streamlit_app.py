import streamlit as st
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PC_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]

# Set up Pinecone
pc = PineconeClient(api_key=pinecone_api_key)
index_name = "agbot"

index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

# Function to load and process documents
def load_and_process_docs(directory):
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to add documents to Pinecone
def add_docs_to_pinecone(docs):
    vectorstore = PineconeVectorStore(index=index_name, embedding=embeddings)
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        docs, embeddings, index_name=index_name, pinecone_client=pc
    )
    return vectorstore_from_docs

# Set up the QA chain
llm = ChatOpenAI(
    model_name="gpt-4-0125-preview",
    temperature=0,
    openai_api_key=openai_api_key
)

vectorstore = PineconeVectorStore(index=index_name, embedding=embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

st.set_page_config(layout="wide")

# Custom CSS and HTML structure
st.markdown("""
<style>
    body {
        font-family: Arial, sans-serif;
    }
    .container {
        display: flex;
        justify-content: space-between;
    }
    .column {
        padding: 10px;
    }
    .left-column {
        width: 20%;
    }
    .middle-column {
        width: 50%;
    }
    .right-column {
        width: 25%;
    }
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        height: 400px;
        overflow-y: auto;
    }
    .chat-message {
        margin-bottom: 10px;
        padding: 5px;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .ai-message {
        background-color: #f0f0f0;
    }
    .insight-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>

<div class="container">
    <div class="column left-column">
        <h3>Configuration</h3>
        <div id="file-upload"></div>
        <div id="file-stats"></div>
        <div id="temperature-slider"></div>
        <div id="max-tokens-input"></div>
    </div>
    <div class="column middle-column">
        <h3>Interactive Chat</h3>
        <div class="chat-container" id="chat-container"></div>
        <div id="chat-input"></div>
    </div>
    <div class="column right-column">
        <h3>Quick Insights</h3>
        <div id="insights"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Streamlit components
st.title('🌱 Welcome to FarmBox')

# Left column components
with st.container():
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'], key="file-upload")
    if uploaded_files:
        st.markdown(f"<div id='file-stats'>Files processed: {len(uploaded_files)}</div>", unsafe_allow_html=True)
        
        # Process and add uploaded files to Pinecone
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join(temp_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        
        docs = load_and_process_docs(temp_dir)
        vectorstore = add_docs_to_pinecone(docs)
        st.success(f"Added {len(docs)} documents to Pinecone")
        
        # Clean up temporary files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, key="temperature-slider")
    max_tokens = st.number_input("Max Tokens", 50, 500, 100, key="max-tokens-input")

# Middle column components
chat_input = st.empty()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
chat_html = "<div id='chat-container'>"
for message in st.session_state.messages:
    class_name = "user-message" if message["role"] == "user" else "ai-message"
    chat_html += f"<div class='chat-message {class_name}'>{message['content']}</div>"
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("What would you like to know about farming?", key="chat-input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response from the QA chain
    response = qa.invoke(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Right column components
insights_html = """
<div id='insights'>
    <div class='insight-box'>Crop yield might increase by 15% with current settings.</div>
    <div class='insight-box'>Water usage is higher than average for this season.</div>
    <div class='insight-box'>Soil nutrient levels are optimal for planting.</div>
</div>
"""
st.markdown(insights_html, unsafe_allow_html=True)

st.write('A place where Agri Knowledge and Insights are exchanged!')