Make sure you install all the required packages from requirements.txt
                                Or
Use the following command to install all the required packages
    pip install unstructured[all-docs] pydantic
    pip install poppler-utils tesseract-ocr
    pip install -qU langchain-groq
    pip install langchain-text-splitters
    pip install pinecone-client


Make sure you have your .env file with the necessary API keys in the same directory.
    1)HF_TOKEN
    2)PINECONE_API_KEY
    3)GROQ_API_KEY


Run the Streamlit app using the following command:
    streamlit run app.py


