import openai
from django.conf import settings
import numpy as np
import re
import PyPDF2
from io import BytesIO
from .models import Document, PlagiarismReport, SimilarityMatch



import os
import magic
from django.core.files.uploadedfile import UploadedFile

# Initialize OpenAI
openai.api_key = settings.OPENAI_API_KEY
def validate_audio_file(file: UploadedFile) -> str:
    """Validate the uploaded audio file"""
    if not isinstance(file, UploadedFile):
        return "Invalid file upload"
    
    # Check file size (max 5MB)
    if file.size > 5 * 1024 * 1024:
        return "File too large (max 5MB)"
    
    # Check file type
    allowed_types = ['audio/mpeg', 'audio/wav', 'audio/ogg']
    file_type = magic.from_buffer(file.read(1024), mime=True)
    file.seek(0)
    
    if file_type not in allowed_types:
        return f"Unsupported file type: {file_type}"
    
    return None

def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
class PDFTextExtractor:
    """Handles PDF text extraction with improved reliability"""
    
    @staticmethod
    def extract_text_from_pdf(file):
        """Extract text from PDF file with error handling"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text.append(page_text)
            return '\n'.join(text)
        except PyPDF2.PdfReadError as e:
            raise ValueError(f"PDF reading error: {str(e)}")
        except Exception as e:
            raise ValueError(f"PDF processing error: {str(e)}")

class PlagiarismDetector:
    """Plagiarism detection using OpenAI embeddings"""
    
    def __init__(self):
        self.similarity_threshold = 0.75  # Adjust based on your needs
        self.embedding_model = "text-embedding-3-small"  # Cost-effective model
        self.chunk_size = 1000  # Characters per chunk for processing
    
    def preprocess_text(self, text):
        """Clean and normalize text for comparison"""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        return text.lower()  # Case-insensitive comparison
    
    def chunk_text(self, text):
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_embeddings(self, texts):
        """Get embeddings from OpenAI API with retry logic"""
        if not texts:
            return []
            
        try:
            response = openai.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            return [np.array(data.embedding) for data in response.data]
        except openai.RateLimitError:
            raise Exception("OpenAI API rate limit exceeded")
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    

        """Main plagiarism detection workflow"""
        try:
            # Extract and preprocess text
            if target_document.file.name.endswith('.pdf'):
                with target_document.file.open('rb') as f:
                    target_text = PDFTextExtractor.extract_text_from_pdf(f)
            else:
                target_text = target_document.content
            
            target_text = self.preprocess_text(target_text)
            target_chunks = self.chunk_text(target_text)
            
            if not target_chunks:
                raise ValueError("No valid text extracted from document")
            
            # Get embeddings for target document
            target_embeddings = self.get_embeddings(target_chunks)
            
            # Create report
            report = PlagiarismReport.objects.create(
                document=target_document,
                overall_similarity=0.0,
                total_matches=0
            )
            
            total_similarity = 0
            total_matches = 0
            
            # Compare against other documents
            comparison_docs = Document.objects.exclude(id=target_document.id)
            
            for doc in comparison_docs:
                try:
                    if doc.file.name.endswith('.pdf'):
                        with doc.file.open('rb') as f:
                            doc_text = PDFTextExtractor.extract_text_from_pdf(f)
                    else:
                        doc_text = doc.content
                    
                    doc_text = self.preprocess_text(doc_text)
                    doc_chunks = self.chunk_text(doc_text)
                    
                    if not doc_chunks:
                        continue
                    
                    doc_embeddings = self.get_embeddings(doc_chunks)
                    
                    # Compare chunks
                    for i, (target_chunk, target_emb) in enumerate(zip(target_chunks, target_embeddings)):
                        for j, (doc_chunk, doc_emb) in enumerate(zip(doc_chunks, doc_embeddings)):
                            similarity = self.calculate_similarity(target_emb, doc_emb)
                            
                            if similarity > self.similarity_threshold:
                                # Verify with GPT for better accuracy
                                verified_score = self.verify_with_gpt(target_chunk, doc_chunk)
                                combined_score = (similarity + verified_score) / 2
                                
                                if combined_score > self.similarity_threshold:
                                    # Record the match
                                    start_pos = target_text.find(target_chunk)
                                    end_pos = start_pos + len(target_chunk)
                                    
                                    SimilarityMatch.objects.create(
                                        report=report,
                                        source_document=doc,
                                        similarity_score=combined_score,
                                        matched_text=target_chunk,
                                        source_text=doc_chunk,
                                        start_position=start_pos,
                                        end_position=end_pos
                                    )
                                    
                                    total_similarity += combined_score
                                    total_matches += 1
                
                except Exception as e:
                    print(f"Error comparing with document {doc.id}: {str(e)}")
                    continue
            
            # Update report statistics
            if total_matches > 0:
                report.overall_similarity = total_similarity / total_matches
                report.total_matches = total_matches
                report.save()
            
            return report
        
        except Exception as e:
            print(f"Plagiarism detection failed: {str(e)}")
            raise