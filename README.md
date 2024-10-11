# Project Name

## Overview
This project is designed to integrate multiple AI and LLM-driven technologies to create a dynamic application for handling PDF documents, video transcripts, and AI-based responses. The project leverages powerful tools such as Streamlit, Google Generative AI, and LangChain to deliver a seamless user experience.

## Features
- **Streamlit**: A fast, interactive, and web-based UI for showcasing the AI application.
- **Google Generative AI**: For generating responses and insights using Google's advanced AI models.
- **LangChain**: To build language model workflows and manage document querying.
- **PDF Handling**: Processes PDF documents to extract, convert, and analyze content.
- **YouTube Transcript API**: Fetches video transcripts for analysis.
- **ChromaDB & FAISS**: Efficient storage and retrieval of vector embeddings for semantic search.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8 or above installed. You can use `pyenv` to manage your Python version.

### 2. Install Dependencies
Run the following command to install the required libraries:

```bash
pip install streamlit google-generativeai python-dotenv langchain PyPDF2 chromadb faiss-cpu langchain_google_genai pdf2image youtube_transcript_api


GOOGLE_API_KEY=your_google_api_key_here

streamlit run app.py
├── app.py                 # Main Streamlit app file
├── README.md              # Project documentation
├── .env                   # Environment variables
├── requirements.txt       # Required libraries
└── src/
    ├── langchain_utils.py # LangChain functions
    ├── pdf_processor.py   # PDF processing module
    └── youtube_api.py     # YouTube transcript fetching module

Key Libraries Used
Streamlit: Interactive web app framework.
Google Generative AI: API for text generation and LLM tasks.
LangChain: Framework for building LLM-based applications.
PyPDF2: PDF file reading and parsing.
pdf2image: Converts PDF pages to images for further processing.
YouTube Transcript API: Retrieves transcript data from YouTube videos.
ChromaDB & FAISS: For efficient vector search and embedding storage.
Contributing
Feel free to fork this project and submit pull requests. All contributions are welcome!

