
# DOCMATE Chatbot

A local chatbot application built with Python and Streamlit that allows you to upload PDF documents and ask questions about their content. The chatbot uses semantic search to find relevant information and will respond with "Not found in provided documents." if the answer isn't available in the uploaded PDFs.

## Features

- 📚 **PDF Upload**: Upload one or more PDF files
- 🔍 **Semantic Search**: Uses sentence transformers for intelligent text search
- 💬 **Interactive Chat**: Ask questions and get answers based on document content
- 🎨 **Modern UI**: Clean, responsive interface with custom styling
- 📝 **Chat History**: View your conversation history
- 🗂️ **Document Management**: See uploaded documents and clear them when needed

## Installation

1. **Clone or download this repository**

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## How to Use

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Process Documents**: Click "Process PDFs" to extract and index the text
3. **Ask Questions**: Type your questions in the text area and click "Ask Question"
4. **View Results**: The chatbot will search through your documents and provide relevant answers

## Technical Details

- **PDF Processing**: Uses PyPDF2 for text extraction
- **Text Chunking**: Splits documents into overlapping chunks for better search
- **Semantic Search**: Uses the `all-MiniLM-L6-v2` model for sentence embeddings
- **Similarity Scoring**: Cosine similarity to find the most relevant text chunks
- **Answer Generation**: Extracts relevant sentences or provides context from the best matching chunk

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- sentence-transformers
- torch
- numpy

## Notes

- The application runs entirely locally - no data is sent to external servers
- The first run may take a few minutes to download the sentence transformer model
- Large PDF files may take longer to process
- The chatbot will respond with "Not found in provided documents." if it cannot find relevant information

## Troubleshooting

- **PDF text extraction issues**: Some PDFs may have security restrictions or be image-based
- **Memory issues**: Very large PDFs may require more RAM
- **Model download**: Ensure you have an internet connection for the first run to download the model

## License

This project is open source and available under the MIT License.


