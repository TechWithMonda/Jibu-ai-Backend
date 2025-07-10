# checker/utils.py
import openai
from django.conf import settings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from .models import Document, PlagiarismReport, SimilarityMatch

openai.api_key = settings.OPENAI_API_KEY

class PlagiarismDetector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.7
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:]', '', text)
        return text
    
    def split_into_sentences(self, text):
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_embeddings(self, texts):
        """Get embeddings for texts using sentence transformer"""
        return self.model.encode(texts)
    
    def check_with_openai(self, text1, text2):
        """Use OpenAI to check similarity between two texts"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a plagiarism detection expert. Analyze the similarity between two texts and provide a similarity score from 0 to 1."},
                    {"role": "user", "content": f"Compare these two texts for similarity:\n\nText 1: {text1}\n\nText 2: {text2}\n\nProvide only a similarity score between 0 and 1."}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            # Extract numerical score
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                return float(score_match.group(1))
            return 0.0
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return 0.0
    
    def detect_plagiarism(self, target_document):
        """Main plagiarism detection function"""
        target_text = self.preprocess_text(target_document.content)
        target_sentences = self.split_into_sentences(target_text)
        
        # Get all other documents for comparison
        comparison_docs = Document.objects.exclude(id=target_document.id)
        
        # Create plagiarism report
        report = PlagiarismReport.objects.create(
            document=target_document,
            overall_similarity=0.0,
            total_matches=0
        )
        
        total_similarity = 0
        total_matches = 0
        
        for doc in comparison_docs:
            doc_text = self.preprocess_text(doc.content)
            doc_sentences = self.split_into_sentences(doc_text)
            
            # Get embeddings
            target_embeddings = self.get_embeddings(target_sentences)
            doc_embeddings = self.get_embeddings(doc_sentences)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(target_embeddings, doc_embeddings)
            
            # Find high similarity matches
            for i, target_sentence in enumerate(target_sentences):
                for j, doc_sentence in enumerate(doc_sentences):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity > self.similarity_threshold:
                        # Double-check with OpenAI
                        openai_score = self.check_with_openai(target_sentence, doc_sentence)
                        final_score = (similarity + openai_score) / 2
                        
                        if final_score > self.similarity_threshold:
                            # Find position in original text
                            start_pos = target_text.find(target_sentence)
                            end_pos = start_pos + len(target_sentence)
                            
                            SimilarityMatch.objects.create(
                                report=report,
                                source_document=doc,
                                similarity_score=final_score,
                                matched_text=target_sentence,
                                source_text=doc_sentence,
                                start_position=start_pos,
                                end_position=end_pos
                            )
                            
                            total_similarity += final_score
                            total_matches += 1
        
        # Update report with final statistics
        overall_similarity = (total_similarity / total_matches) if total_matches > 0 else 0
        report.overall_similarity = overall_similarity
        report.total_matches = total_matches
        report.save()
        
        return report

def extract_text_from_file(file):
    """Extract text from uploaded files"""
    if file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    elif file.name.endswith('.docx'):
        import docx
        doc = docx.Document(file)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif file.name.endswith('.pdf'):
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        raise ValueError("Unsupported file format")