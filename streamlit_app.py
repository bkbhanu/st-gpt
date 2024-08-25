import streamlit as st
from pinecone import Pinecone

# Access secrets
pinecone_api_key = st.secrets["PC_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]

st.title('Pinecone Connection Test')

# Initialize Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    st.success("Successfully initialized Pinecone client")
    
    # List indexes
    indexes = pc.list_indexes()
    st.write("Available Pinecone indexes:")
    for index in indexes:
        st.write(f"- {index.name}")
    
    # Try to connect to a specific index
    index_name = "agbot"  # You can change this to match your index name
    try:
        index = pc.Index(index_name)
        st.success(f"Successfully connected to index: {index_name}")
        # Print some basic info about the index
        st.write(f"Index dimension: {index.describe_index_stats().dimension}")
        st.write(f"Total vector count: {index.describe_index_stats().total_vector_count}")
    except Exception as e:
        st.error(f"Failed to connect to index {index_name}: {str(e)}")

except Exception as e:
    st.error(f"Failed to initialize Pinecone client: {str(e)}")

# Display the Pinecone environment being used
st.write(f"Pinecone Environment: {pinecone_environment}")