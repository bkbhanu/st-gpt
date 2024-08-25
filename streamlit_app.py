import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Access secrets
pinecone_api_key = st.secrets["PC_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(layout="wide")

# Sidebar with collapsible configuration
with st.sidebar:
    st.title('Configure your chat')
    
    # Collapsible configuration section
    with st.expander("Configuration", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        max_tokens = st.number_input("Max Tokens", 250, 350, 500, 1500)
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"], index=0)  # Set default to gpt-4o-mini

# Main area
st.title('Farmer Friend - For the farmer, by the farmer')
st.write("An Agri Knowledge and Insights Hub")

# System message to provide context and rules
system_message = """
You are an AI assistant specialized in agricultural knowledge. Your primary function is to provide accurate and helpful information about farming practices, crop management, and agricultural technologies. Please adhere to the following rules:

1. Base your responses on the information retrieved from the knowledge base. If the information is not available, clearly state that you don't have that specific information.
2. Avoid making assumptions or providing speculative information. If you're unsure, express that uncertainty.
3. If asked about specific numerical data or statistics, only provide them if they are explicitly mentioned in the retrieved information.
4. When discussing farming practices or technologies, emphasize the importance of consulting with local agricultural experts or extension services for advice tailored to specific regions or conditions.
5. If a question is outside the scope of agriculture and farming, politely redirect the conversation back to relevant topics.
6. Prioritize sustainable and environmentally friendly farming practices in your recommendations when appropriate.
7. If asked about controversial topics in agriculture (e.g., GMOs, pesticide use), provide balanced information based on scientific consensus and encourage users to consult authoritative sources.

Remember, your goal is to provide helpful, accurate, and responsible information about agriculture and farming.
"""

# Initialize Pinecone and set up the chat model
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "agbot"
    index = pc.Index(index_name)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    chat_model = ChatOpenAI(
        temperature=temperature,
        openai_api_key=openai_api_key,
        streaming=True,
        model_name=model,
        max_tokens=max_tokens
    )

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=f"{system_message}\n\nChat History: {{chat_history}}\nHuman: {{question}}\nContext: {{context}}\nAI Assistant: "
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    # Create two columns for chat and insights
    chat_column, insights_column = st.columns([3, 1])

    # Chat Interface
    with chat_column:
        st.subheader("Discuss with your agri expert")
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat messages container
        chat_container = st.container()

        # Chat input at the bottom
        prompt = st.chat_input("What would you like to know about farming?")

        # Display chat messages
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Process chat input
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in qa_chain.stream({"question": prompt}):
                        if isinstance(chunk, dict) and "answer" in chunk:
                            full_response += chunk["answer"]
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Generate insights based on the AI's answer
            insights_prompt = f"Based on the following agricultural information, provide 2-3 short, actionable insights or tips in 2-3 sentences total:\n\n{full_response}"
            insights = chat_model.invoke(insights_prompt)

            # Display insights in the insights column
            with insights_column:
                st.subheader("Insights")
                st.info(insights.content)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")