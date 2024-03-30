import streamlit as st
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores.neo4j_vector import Neo4jVector

# tag::importllm[]
from llm import llm, embeddings
# end::importllm[]

# Initialize Neo4jVector for Game of Thrones database
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                               # (1)
    url=st.secrets["NEO4J_URI"],              # (2)
    username=st.secrets["NEO4J_USERNAME"],    # (3)
    password=st.secrets["NEO4J_PASSWORD"],    # (4)
    index_name="gotPlots",                    # (5) Adjust index name for Game of Thrones
    node_label="Episode",                     # (6) Adjust node label for Game of Thrones
    text_node_property="plot",                # (7) Adjust text property for Game of Thrones plot
    embedding_node_property="fastrf_embedding",  # (8) Adjust embedding property for Game of Thrones plot
    retrieval_query="""
MATCH (episode:Episode)
WHERE exists(episode.plot)  # Ensure the episode has a plot
RETURN
    episode.plot AS text,
    score,
    {
        title: episode.title,
        directors: [ (director)-[:DIRECTED]->(episode) | director.name ],
        actors: [ (actor)-[r:APPEARED_IN]->(episode) | [actor.name, r.role] ],
        season: episode.season,
        episode_number: episode.episodeNumber,
        source: 'https://demo.neo4jlabs.com:7473/browser/?dbms=neo4j://gameofthrones@demo.neo4jlabs.com&db=gameofthrones' + replace(episode.title, ' ', '-')
    } AS metadata
"""
)

# tag::retriever[]
retriever = neo4jvector.as_retriever()
# end::retriever[]

# tag::qa[]
kg_qa = RetrievalQA.from_chain_type(
    llm,                  # <1>
    chain_type="stuff",   # <2>
    retriever=retriever,  # <3>
)
# end::qa[]

# tag::generate-response[]
def generate_response(prompt):
    """
    Use the Neo4j Vector Search Index
    to augment the response from the LLM
    """

    # Handle the response
    response = kg_qa({"question": prompt})

    return response['answer']
# end::generate-response[]
