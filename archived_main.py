import ollama
import psycopg2

from settings import settings 

EMBEDDING_MODEL = settings.model_embedding
LANGUAGE_MODEL = settings.model_language

dataset = []

def create_and_load_chunks(cursor):
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(cursor, chunk)
        print(f'Added chunk {i + 1}/{len(dataset)} to the database')

def add_chunk_to_database(cursor, chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
  embedding = embedding['embeddings'][0]
  cursor.execute(
      "INSERT INTO rag_demo (chunk, embedding) VALUES (%s, %s)",
      (chunk, embedding)
  )

def retrieve(query, cursor, top_n=3):
    query_embedding = get_vector_embeddings(query)
    
    cursor.execute(
        """
         SELECT chunk, 1 - (embedding <=> %s::vector) AS similarity
         FROM rag_demo
         ORDER BY similarity DESC
        """,
        (query_embedding,)
    )
    similarities = cursor.fetchall()
    return similarities[:top_n]

def chat_with_rag(cursor):
    input_query = input("Ask me a question: ")
    retrieved_knowledge = retrieve(input_query, cursor)

    print("Retrieved Knowledge:")
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity}) {chunk}')

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role':'system', 'content': instruction_prompt},
            {'role':'user', 'content': input_query}
        ],
        stream=True
    )

    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def load_dataset():
    with open('assets/cat-facts.txt', 'r') as file:
        global dataset
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')

def get_vector_embeddings(query_string):
    return ollama.embed(model=EMBEDDING_MODEL, input=query_string)['embeddings'][0]

def main():
    conn = psycopg2.connect(settings.database_url)
    cur = conn.cursor()
    
    load_dataset()
    create_and_load_chunks(cur)
    chat_with_rag(cur)
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()