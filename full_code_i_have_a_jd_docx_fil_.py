# -*- coding: utf-8 -*-
"""
**1. Install Required Libraries:**

Ensure you have the following libraries installed:

```bash
pip install PyPDF2 docx2txt nltk sklearn
```

**2. Import Necessary Libraries:**
"""

!pip install PyPDF2 docx2txt nltk pandas
import PyPDF2
import docx2txt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

"""**3. Define Functions:**

* **Extract Text:**
"""

def extract_text(file_path):
      if file_path.endswith(".pdf"):
          with open(file_path, 'rb') as pdf_file:
              pdf_reader = PyPDF2.PdfReader(pdf_file)
              text = ""
              for page in pdf_reader.pages:
                  text += page.extract_text()
              return text
      elif file_path.endswith(".docx"):
          return docx2txt.process(file_path)
      else:
          return "Unsupported file format"

"""* **Preprocess Text:**"""

def preprocess_text(text):
      # Tokenization, stemming, stop word removal
      words = nltk.word_tokenize(text)
      words = [word.lower() for word in words if word.isalpha()]
      words = [word for word in words if word not in stopwords.words('english')]
      return ' '.join(words)

"""* **Calculate Similarity:**"""

def calculate_similarity(cv_text, jd_text):
      vectorizer = TfidfVectorizer()
      tfidf_matrix = vectorizer.fit_transform([cv_text, jd_text])
      similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
      return similarity_score

"""**4. Main Script:**"""

# Extract JD text
jd_text = preprocess_text(extract_text('/content/sample_data/job_description.docx'))

# Process CVs and calculate similarity scores
cv_folder_path = '/content/sample_data'
cv_scores = []
for cv_file in os.listdir(cv_folder_path):
    cv_path = os.path.join(cv_folder_path, cv_file)
    cv_text = preprocess_text(extract_text(cv_path))
    similarity_score = calculate_similarity(cv_text, jd_text)
    cv_scores.append((cv_file, similarity_score))

# Rank CVs by similarity score
cv_scores.sort(key=lambda x: x[1], reverse=True)

# Print top N CVs
top_n = 5
for cv_file, score in cv_scores[:top_n]:
    print(f"{cv_file}: {score}")



"""**Explanation:**

1. **Extract JD and CV Texts:** Extract text from the JD and each CV using the `extract_text` function.
2. **Preprocess Text:** Clean the text by tokenizing, stemming, and removing stop words.
3. **Calculate Similarity:** Use TF-IDF vectorization and cosine similarity to measure the similarity between each CV and the JD.
4. **Rank CVs:** Sort the CVs by their similarity scores to the JD.
5. **Print Top N:** Print the top N CVs with their similarity scores.

**Additional Considerations:**

- **Advanced Techniques:** Consider using more sophisticated techniques like word embeddings or BERT for better semantic understanding.
- **Domain-Specific Knowledge:** Incorporate domain-specific knowledge and rules to improve the accuracy of the comparison.
- **Human Review:** While automated tools can provide a good initial screening, human review is essential for making final decisions.

By following these steps and customizing the code to your specific needs, you can effectively use Python to compare CVs against a JD and identify the most suitable candidates.

<div class="md-recitation">
"""