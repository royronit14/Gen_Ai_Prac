from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Initialize Gemini Embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=10
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about bumrah"

# Generate document embeddings
doc_embeddings = embedding.embed_documents(
    documents,
    task_type="RETRIEVAL_DOCUMENT"
)

# Generate query embedding
query_embedding = embedding.embed_query(
    query,
    task_type="RETRIEVAL_QUERY"
)


'''prnt(query_embedding)  #its a 1d list
print(doc_embeddings)   # its a 2d list

Calculate cosine similarity: Calculates similarity between the query and each document.
cosine similarity mai dono 2Dimensional list honi chahiye
[0]: Extracts the first row of the resulting similarity matrix.
'''
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Find the most relevant document
# Matlab enumerate = index + value ko pair karna.
# Normally max() ko values compare karni hoti hain.
# but yahan key batata hai kis basis par comparison karna hai.
index, score = max(
    enumerate(scores),
    key=lambda x: x[1]
)

print(query)
print(documents[index])
print("Similarity score is:", score)