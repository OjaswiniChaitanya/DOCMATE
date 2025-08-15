import streamlit as st
import PyPDF2
import io
import os
import tempfile
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Dict, Tuple
import re

# Page configuration
st.set_page_config(
    page_title="DOCMATE",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)



class PDFChatbot:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.text_chunks = []
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from a PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start + chunk_size // 2:  # Only break if we find a reasonable sentence boundary
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_documents(self, pdf_files: List) -> None:
        """Add PDF documents to the chatbot"""
        all_text = ""
        
        for pdf_file in pdf_files:
            with st.spinner(f"Processing {pdf_file.name}..."):
                text = self.extract_text_from_pdf(pdf_file)
                if text:
                    all_text += text + "\n\n"
                    self.documents.append(pdf_file.name)
        
        if all_text.strip():
            # Chunk the combined text
            self.text_chunks = self.chunk_text(all_text)
            
            # Create embeddings
            with st.spinner("Creating embeddings..."):
                self.embeddings = self.model.encode(self.text_chunks, convert_to_tensor=True)
            
            st.success(f"Successfully processed {len(pdf_files)} PDF(s) with {len(self.text_chunks)} text chunks!")
        else:
            st.error("No text could be extracted from the uploaded PDFs.")
    
    def search_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant document chunks based on the query"""
        if self.embeddings is None or not self.text_chunks:
            return []
        
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top-k results
        top_results = torch.topk(cos_scores, min(top_k, len(cos_scores)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append((self.text_chunks[idx], score.item()))
        
        return results
    
    def answer_question(self, question: str) -> str:
        """Answer a question based on the uploaded documents"""
        if not self.text_chunks:
            return "No documents have been uploaded yet. Please upload some PDFs first."
        
        # Search for relevant chunks
        relevant_chunks = self.search_documents(question, top_k=8)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information in the provided documents to answer your question."
        
        # Check if any chunk has a reasonable similarity score
        max_score = max(score for _, score in relevant_chunks)
        if max_score < 0.3:  # Threshold for relevance
            return "I couldn't find any relevant information in the provided documents to answer your question."
        
        # Filter chunks with good similarity scores
        good_chunks = [chunk for chunk, score in relevant_chunks if score > 0.3]
        
        if not good_chunks:
            return "I couldn't find any relevant information in the provided documents to answer your question."
        
        # Combine relevant chunks for a more comprehensive answer
        combined_text = "\n\n".join(good_chunks)
        
        # Try to extract relevant sentences that contain keywords from the question
        question_words = [word.lower() for word in re.findall(r'\b\w+\b', question) if len(word) > 2]
        
        sentences = re.split(r'[.!?]+', combined_text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # Consider more sentences
                # Check if sentence contains question keywords or is part of the context
                if any(word in sentence.lower() for word in question_words) or len(relevant_sentences) < 10:
                    relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Include all relevant sentences for a complete answer
            answer = ". ".join(relevant_sentences) + "."
        else:
            # If no specific sentences match, return the full best chunk
            best_chunk = good_chunks[0]
            # Return the complete chunk without truncation
            answer = best_chunk
        
        return answer

def main():
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploader_counter' not in st.session_state:
        st.session_state.uploader_counter = 0
    
    # Header
    st.title("📚 DOCMATE Chatbot")
    st.write("Upload PDFs and ask questions about their content")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        # Use a unique key that changes when we want to clear
        uploader_key = f"file_uploader_{st.session_state.get('uploader_counter', 0)}"
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze",
            key=uploader_key
        )
        
        if uploaded_files:
            if st.button("Process PDFs", type="primary"):
                st.session_state.chatbot.add_documents(uploaded_files)
                st.session_state.chat_history = []  # Clear chat history when new docs are added
        
        # Display uploaded documents
        if st.session_state.chatbot.documents:
            st.header("📋 Uploaded Documents")
            for doc in st.session_state.chatbot.documents:
                st.write(f"• {doc}")
            
            if st.button("Clear Documents"):
                st.session_state.chatbot = PDFChatbot()
                st.session_state.chat_history = []
                st.session_state.uploader_counter += 1  # This will reset the file uploader
                st.rerun()
    
    # Main chat interface - use more space for the chat
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    
    with col2:
        # Chat history
        if st.session_state.chat_history:
            st.header("💬 Chat History")
            
            # Create a container for chat messages with scrollable area
            chat_container = st.container()
            
            with chat_container:
                # Display each message using Streamlit components
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        # User message with blue background
                        st.markdown(f"**You:** {message['content']}")
                        st.markdown("---")
                    else:
                        # Bot message with purple background
                        st.markdown(f"**Bot:** {message['content']}")
                        st.markdown("---")
        
        # Question input
        st.header("❓ Ask a Question")
        question = st.text_area(
            "Enter your question about the uploaded PDFs:",
            height=100,
            placeholder="e.g., What is the main topic discussed in the documents?"
        )
        
        if st.button("Ask Question", type="primary", disabled=not question.strip()):
            if not st.session_state.chatbot.text_chunks:
                st.error("Please upload some PDFs first!")
            else:
                with st.spinner("Searching for answer..."):
                    answer = st.session_state.chatbot.answer_question(question)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                st.rerun()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
