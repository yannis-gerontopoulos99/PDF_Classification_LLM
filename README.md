# YannisGerontopoulos_MLE_Assignment

PDF Classification and Information Extraction
Description:

Overview

This project processes scientific papers in PDF format using a Retrieval-Augmented Generation (RAG) approach. The script reads, splits, embeds, and classifies the content of each paper based on categories relevant to document analysis. It uses Langchain for document loading, FAISS for indexing, Hugging Face models for embeddings and language models, and outputs a JSON file with the results.

Key Features:

1)    PDF Processing: Extracts text content from PDFs.
2)    Text Splitting: Breaks down documents into manageable chunks for embedding and retrieval.
3)    Document Retrieval: Uses FAISS for vector-based document retrieval.
4)    Metadata Extraction: Extracts and categorizes the title, authors, and main focus of the paper.
5)    Classification: Classifies papers into predefined categories (Tables, Classification, Key Information Extraction, etc.).
6)    Output: Stores results in a structured JSON format, optionally allowing for file download.

Script Flow

1)    File Upload & Extraction:
        The script first uploads a ZIP file containing scientific PDFs via Google Colab’s file upload interface.
        It extracts the PDFs into a working folder for further processing.

2)    PDF Loading & Cleaning:
        Each PDF is loaded using the PyPDFLoader, which reads and cleans the content by removing excessive whitespace and newlines.

3)    Text Splitting:
        To avoid large chunks of data that could be difficult to process, the content is split using RecursiveCharacterTextSplitter, with adjustable chunk sizes and overlap parameters.

4)    Embeddings:
        The text chunks are embedded using the HuggingFaceEmbeddings model from the Hugging Face Hub, allowing for vector-based retrieval.

5)    LLM-Based Question Answering:
        The script uses a pretrained Hugging Face model (Hermes-3-Llama-3.1-8B) to process and extract the title, authors, and categorize the content.
        This process is done by prompting the model with specific instructions to extract and categorize information.

6)    Result Extraction:
        Extracts the title, authors, and category of each paper using regular expressions.
        Results are normalized to avoid duplicates and stored in a structured JSON format.

7)    Final Output:
        A JSON file containing categorized paper metadata is saved and can be downloaded directly from the Colab environment.

JSON Output Format

{
  "tables": [{"originalFileName": "paper_A.pdf", "title": "recognized title", "authors": ["author_A", "author_B"]},
            {"originalFileName": "paper_B.pdf", "title": "recognized title", "authors": ["author_C", "author_D"]},
            ...],
  "classification": [...],
  "keyInformationExtraction": [...],
  "opticalCharacterRecognition": [...],
  "datasets": [...],
  "layoutUnderstanding": [...],
  "others": [...]
}

Customizable Parameters

    Embedding Model: Change the embedding model used by modifying the model_name parameter in the HuggingFaceEmbeddings.

    LLM: Change the language model for question answering by updating the repo_id in the HuggingFaceHub initialization.

    Text Split Sizes: Adjust chunk size and overlap by modifying the chunk_size and chunk_overlap parameters in text_splits() function.
