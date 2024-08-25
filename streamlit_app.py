import streamlit as st

st.set_page_config(layout="wide")

st.title('🌱 Welcome to FarmBox')
st.write('A place where Agri Knowledge and Insights are exchanged!')

# Create three columns for the layout
left_column, middle_column, right_column = st.columns([1, 2, 1])

# Left sidebar
with left_column:
    st.sidebar.header("Configuration")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    
    if uploaded_files:
        st.sidebar.write(f"Files processed: {len(uploaded_files)}")
    
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
    max_tokens = st.sidebar.number_input("Max Tokens", 50, 500, 100)

# Middle column - Chat interface
with middle_column:
    st.header("Interactive Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about farming?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Here you would typically call your AI model to generate a response
        # For now, we'll just echo the user's input
        response = f"AI response to: {prompt}"
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Right column - Quick insights
with right_column:
    st.header("Quick Insights")
    
    # Placeholder for insights
    insight1 = st.empty()
    insight2 = st.empty()
    insight3 = st.empty()

    # Simulated insights (replace with actual logic later)
    insight1.info("Crop yield might increase by 15% with current settings.")
    insight2.warning("Water usage is higher than average for this season.")
    insight3.success("Soil nutrient levels are optimal for planting.")
