# Context-Aware Meal Planning System using RAG and Gemini with Macro Tracking

## Data Collection
 Scrape superstore for the dataset.

## Data Cleaning
- Remove null values.
- Split and normalize item names, nutritional macros, and price details.
- Rename columns for clarity and consistency.

## Data Transformation
- Extract relevant fields such as item name, price, calories, protein, carbs, and fats.
- Export cleaned versions:
  - `food_items_list_cleaned.csv`: Core cleaned dataset.
  - `food_items_list_with_macros.csv`: Items with full macro details.
  - `food_items_list_without_macros.csv`: Items missing macros for tracking.

## Embedding and Retrieval with Gemini + ChromaDB
- Use Google Gemini (`text-embedding-004`) to embed both user queries and grocery product descriptions.
- Store embeddings in an in-memory ChromaDB collection for fast semantic retrieval.
- Rank product relevance based on cosine similarity to the query embedding.

## Application Overview: Macro-Man 🕸️🛒
A Streamlit web app where users can:
- Input a grocery-related query (e.g., "high protein meal plan under $15").
- Select desired products from a searchable list.
- Get a structured grocery list using Gemini 2.0 Flash, including:
  - Per-item macros, cost, and justification.
  - Daily and weekly nutritional and budget summaries.
  - Enforced constraints for calorie range, item repetition, and budget limits.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Launch the app

3. Provide your Google Generative AI API key in the UI.

## File Descriptions

`food_items_list_cleaned.csv`: Final cleaned product list with macros.

`macros.py`: File for building the app and deploying on Streamlit.

`Cleaning_Dataset.ipynb`: Notebook for initial data prep.

`Superstore_data_cleaning_transformation.ipynb`: More advanced transformations.

`requirements.txt`: All necessary packages.

`README.md`: You're reading it 

## Future Improvements

- Adding essential vitamins and minerals.
- Integrate weekly planning with more number of food items.
