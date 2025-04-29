import streamlit as st
import pandas as pd
import io
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from google import genai
from google.genai import types
import re

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    def __init__(self):
        pass  # Empty init for now

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=embedding_task),
        )
        return [e.values for e in response.embeddings]
    
# --- Streamlit App ---
st.title("Your Friendly Neighborhood Macro-Man 🕸️🛒 (But for Groceries)")

# Get input key from the user
api_key = st.text_input("Enter your Google API Key:", type="password")

if api_key:
    # Initialize Google Gemini client
    client = genai.Client(api_key=api_key)

    # Load data
    df = pd.read_csv("food_items_list_cleaned.csv")
    set_items = set(df['item'])
    product_options = list(set_items)

    # Query & selection
    query = st.text_input("Describe your grocery needs:")
    selected_items = st.multiselect("Select products to include:", sorted(product_options))

    if st.button("Generate Grocery List"):
        if not query or not selected_items:
            st.warning("Please enter a query and select products.")
        else:
            with st.spinner("Generating your optimized grocery list... 🛒"):
                # Filter dataframe
                filtered_df = df[df['item'].str.lower().apply(
                lambda name: any(re.search(rf'\b{re.escape(item)}\b', name) for item in selected_items)
                )].copy()

                # Embedding
                embed_fn = GeminiEmbeddingFunction()

                # Create ChromaDB in memory
                chroma_client = chromadb.Client()
                DB_NAME = "googlecardb"

                embed_fn.document_mode = True  # document embedding
                db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

                # Embed and add to DB
                filtered_df['embedding'] = embed_fn(filtered_df['text'].tolist())  # 'text' column must exist
                db.add(
                    documents=filtered_df["text"].tolist(),
                    ids=[str(i) for i in range(len(filtered_df))]
                )
                
                # Embed query
                embed_fn.document_mode = False  # query embedding
                query_embedding = embed_fn([query])[0]

                results = db.query(
                    query_embeddings=[query_embedding],
                    n_results=len(filtered_df)
                )

                # Prepare context
                top_matches = results['documents'][0]
                context = "\n".join(top_matches)

                # Build Prompt
                prompt = f"""
                You are an expert grocery assistant. Based on the user's query and the product catalog below, create a structured grocery list.
                
                Item Number: [number]
                Item Name: [name]
                Price: $[price]
                Protein: [x]g
                Carbs: [y]g
                Fats: [z]g 
                Calories: [calories] kcal
                Explanation: [2-line reason why it's chosen]
                
                Format **each item** using the exact 7-line block above, with an empty line between blocks.
                
                Then at the end:
                
                - Total Cost for the day: $[value]
                - Total Protein for the day: [g]
                - Total Carbs for the day: [g]
                - Total Fats for the day: [g] 
                - Total Calories for the day: [kcal]
                
                - Total Cost for the week: $[value]
                - Total Protein for the week: [g]
                - Total Carbs for the week: [g]
                - Total Fats for the week: [g] 
                - Total Calories for the week: [kcal]
                
                RULES:
                1. Do NOT exceed the given budget.
                2. Do NOT exceed or fall below the target calories by more than 50.
                3. Do NOT include any one type of item more than twice.
                4. Start the Item Number from 1 and increment sequentially for each item.
                5. Use only items listed in the Product Info.
                6. Do NOT add any extra explanation, bullet points, or formatting outside what is shown above.
                
                Product Info:
                {context}
                
                User Query:
                "{query}"
                """
                # Call Gemini
                response = client.models.generate_content_stream(
                    model='gemini-2.0-flash-thinking-exp',
                    contents=prompt
                )

                text_output = st.empty()
                buf = io.StringIO()

                for chunk in response:
                    buf.write(chunk.text)
                    text_output.text(buf.getvalue())

else:
    st.info("Please enter your Google API Key to continue.")