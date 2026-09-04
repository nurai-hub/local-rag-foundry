# Local RAG Application with Foundry Local

A Retrieval-Augmented Generation (RAG) course assistant that runs entirely on-device using Microsoft Foundry Local. It answers questions about Branch Prediction by retrieving relevant sections from a PDF document and generating grounded answers with a local LLM — no cloud, no API keys, no internet required after setup.

## How it works

1. A PDF document is loaded and split into overlapping text chunks
2. Each chunk is converted into a vector embedding using qwen3-embedding-0.6b
3. When a question is asked, it is embedded the same way and compared to all chunks using cosine similarity
4. The most relevant chunks are retrieved and passed as context to a local chat model (qwen2.5-1.5b)
5. The model generates an answer grounded in the retrieved context

## Tech stack

- Foundry Local - on-device model runtime (OpenAI-compatible API)
- Python - openai SDK, pypdf, numpy

## Setup

1. Install Foundry Local: winget install Microsoft.FoundryLocal
2. Create a virtual environment and install dependencies: pip install openai pypdf numpy
3. Start the Foundry Local server: foundry server start
4. Load the required models: foundry model load qwen2.5-1.5b and foundry model load qwen3-embedding-0.6b
5. Place your PDF in the project folder and update PDF_FILE in rag.py
6. Run: python rag.py

## Notes

This project was built as part of the Microsoft Internship Program, exploring Foundry Local for building on-device RAG applications. The system performs best with English questions, since it retrieves context from an English-language source document.
