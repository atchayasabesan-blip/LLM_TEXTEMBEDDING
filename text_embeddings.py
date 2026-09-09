import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# STEP 1: COLLECT 20 SENTENCES


sentences = [
    "Artificial intelligence is transforming many industries around the world.",
    "Machine learning allows computers to learn patterns from data.",
    "Researchers are developing new technologies for renewable energy.",
    "Solar power is becoming an important source of clean electricity.",
    "The government announced new measures to improve public education.",
    "Online learning provides students with access to educational resources.",
    "Scientists discovered new evidence about the formation of distant galaxies.",
    "Space telescopes help astronomers study objects in the universe.",
    "The football team won the championship after a close final match.",
    "Athletes require regular training to improve their performance.",
    "Climate change is causing significant changes in global weather patterns.",
    "Rising temperatures are affecting ecosystems around the world.",
    "Doctors are using artificial intelligence to assist with medical diagnosis.",
    "Modern hospitals use digital systems to manage patient information.",
    "Electric vehicles are becoming more popular because they produce fewer emissions.",
    "Battery technology is important for the future of electric transportation.",
    "Cybersecurity protects computer systems from unauthorized access.",
    "Strong passwords can help users protect their online accounts.",
    "Social media platforms allow people to communicate and share information.",
    "Digital communication has changed the way people interact with each other."
]

print("=" * 60)
print("TEXT EMBEDDINGS AND SEMANTIC SIMILARITY")
print("=" * 60)

print("\nNumber of sentences:", len(sentences))



# STEP 2: LOAD EMBEDDING MODEL

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")



# STEP 3: GENERATE EMBEDDINGS


print("\nGenerating embeddings...")

embeddings = model.encode(sentences)

print("Embeddings generated successfully!")
print("Embedding shape:", embeddings.shape)



# STEP 4: DISPLAY FIRST EMBEDDING


print("\nFirst sentence:")
print(sentences[0])

print("\nFirst sentence embedding:")
print(embeddings[0])



# STEP 5: SAVE EMBEDDINGS AS CSV


embedding_df = pd.DataFrame(embeddings)

embedding_df.insert(
    0,
    "Sentence_ID",
    range(1, 21)
)

embedding_df.to_csv(
    "sentence_embeddings.csv",
    index=False
)

print("\nSentence embeddings saved as:")
print("sentence_embeddings.csv")



# STEP 6: SAVE EMBEDDINGS AS NUMPY


np.save(
    "sentence_embeddings.npy",
    embeddings
)

print("Sentence embeddings saved as:")
print("sentence_embeddings.npy")



# STEP 7: CREATE 10 SENTENCE PAIRS


pairs = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20)
]

print("\nNumber of sentence pairs:", len(pairs))



# STEP 8: CALCULATE COSINE SIMILARITY


results = []

for sentence1_id, sentence2_id in pairs:

    embedding1 = embeddings[sentence1_id - 1]
    embedding2 = embeddings[sentence2_id - 1]

    similarity = cosine_similarity(
        [embedding1],
        [embedding2]
    )[0][0]

    results.append({
        "Sentence_1_ID": sentence1_id,
        "Sentence_2_ID": sentence2_id,
        "Sentence_1": sentences[sentence1_id - 1],
        "Sentence_2": sentences[sentence2_id - 1],
        "Cosine_Similarity": round(float(similarity), 4)
    })


similarity_df = pd.DataFrame(results)



# STEP 9: DISPLAY ALL RESULTS


print("\n" + "=" * 60)
print("COSINE SIMILARITY RESULTS")
print("=" * 60)

print(similarity_df[
    [
        "Sentence_1_ID",
        "Sentence_2_ID",
        "Cosine_Similarity"
    ]
].to_string(index=False))



# STEP 10: SAVE SIMILARITY RESULTS

similarity_df.to_csv(
    "cosine_similarity_scores.csv",
    index=False
)

print("\nSimilarity scores saved as:")
print("cosine_similarity_scores.csv")



# STEP 11: FIND TOP 5 MOST SIMILAR PAIRS


top_5 = similarity_df.sort_values(
    by="Cosine_Similarity",
    ascending=False
).head(5)


print("\n" + "=" * 60)
print("TOP 5 MOST SIMILAR SENTENCE PAIRS")
print("=" * 60)

for index, row in top_5.iterrows():

    print("\nSimilarity:", row["Cosine_Similarity"])

    print("Sentence 1:")
    print(row["Sentence_1"])

    print("Sentence 2:")
    print(row["Sentence_2"])



# STEP 12: FINAL MESSAGE


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
print("1. sentence_embeddings.csv")
print("2. sentence_embeddings.npy")
print("3. cosine_similarity_scores.csv")