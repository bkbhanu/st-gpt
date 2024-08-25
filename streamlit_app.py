import streamlit as st
from pinecone import Pinecone
import numpy as np

# Access secrets
pinecone_api_key = st.secrets["PC_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]

st.title('Pinecone Interaction Test')

# Initialize Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key)
    st.success("Successfully initialized Pinecone client")
    
    # Connect to a specific index
    index_name = "agbot"  # You can change this to match your index name
    index = pc.Index(index_name)
    st.success(f"Successfully connected to index: {index_name}")
    
    # Display index info
    stats = index.describe_index_stats()
    st.write(f"Index dimension: {stats.dimension}")
    st.write(f"Total vector count: {stats.total_vector_count}")

    # Query the index
    st.subheader("Query the Index")
    query_vector = np.random.rand(stats.dimension).tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    st.write("Query Results:")
    for result in results.matches:
        st.write(f"ID: {result.id}, Score: {result.score}")

    # Upsert data
    st.subheader("Upsert Data")
    if st.button("Upsert Sample Data"):
        upsert_data = [
            ("id1", np.random.rand(stats.dimension).tolist(), {"text": "Sample text 1"}),
            ("id2", np.random.rand(stats.dimension).tolist(), {"text": "Sample text 2"}),
        ]
        index.upsert(vectors=upsert_data)
        st.success("Sample data upserted successfully")

    # Query again after upsert
    st.subheader("Query After Upsert")
    if st.button("Query Again"):
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        st.write("New Query Results:")
        for result in results.matches:
            st.write(f"ID: {result.id}, Score: {result.score}, Metadata: {result.metadata}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Display the Pinecone environment being used
st.write(f"Pinecone Environment: {pinecone_environment}")