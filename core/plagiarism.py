# # plagiarism.py

# from openai import OpenAI
# import os

# client = OpenAI()

# def check_plagiarism_with_embeddings(uploaded_text, existing_texts):
#     """
#     Compares uploaded_text to existing_texts using OpenAI embeddings.
#     Returns a list of possible matches with similarity scores.
#     """
#     def get_embedding(text):
#         response = client.embeddings.create(
#             model="text-embedding-3-small",  # or text-embedding-ada-002
#             input=text,
#         )
#         return response.data[0].embedding

#     from scipy.spatial.distance import cosine

#     uploaded_emb = get_embedding(uploaded_text)
    
#     results = []

#     for other_text in existing_texts:
#         other_emb = get_embedding(other_text)
#         similarity = 1 - cosine(uploaded_emb, other_emb)
#         results.append({
#             "text": other_text[:100],  # just show a snippet
#             "similarity": round(similarity, 4),
#             "plagiarized": similarity > 0.8  # tweak threshold if needed
#         })

#     return results
