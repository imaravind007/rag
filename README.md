# Renewable Energy Copilot 🌿

Renewable Energy Copilot is a Streamlit-based application that leverages OpenAI and Pinecone to provide AI-powered insights and document analysis for renewable energy projects. It allows users to upload documents, embed them into a vector database, and query them using natural language.

---

## Features

- **Document Upload**: Supports PDF, DOCX, TXT, and XLSX formats.
- **AI-Powered Chat**: Ask questions about uploaded documents and get accurate, context-aware answers.
- **Namespace Management**: Organize and manage document embeddings using Pinecone namespaces.
- **Customizable UI**: Includes dark mode and a responsive design.
- **Integration with OpenAI and Pinecone**: Uses OpenAI for embeddings and chat models, and Pinecone for vector storage and search.

---

## Technical Details

- **Embedding Model**: Uses OpenAI's `text-embedding-3-large` model for generating vector embeddings of documents.
- **Chat Model**: Powered by OpenAI's `gpt-4` for natural language understanding and responses.
- **Vector Database**: Serverless Pinecone is used for storing and querying vector embeddings efficiently.
- **Framework**: Built with [Streamlit](https://streamlit.io/) for a fast and interactive user interface.
- **Document Parsing**: Utilizes libraries like `Docling` for handling various data types such as  PDF, DOCX, TXT, and XLSX.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag

2. Create a virtual environment and activate it
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install Dependencies:
   ```bash
   pip install -r requirements.txt

5. Create a .env file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   PINE_API_KEY=your_pinecone_api_key

6. Usage:
   ```bash
   streamlit run main.py


## Dependencies
   - Streamlit
   - OpenAI
   - Pinecone
   - LangChain
   - PyMuPDF
   - python-docx

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License 
This project is licensed under the MIT License. See the LICENSE file for details.