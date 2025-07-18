from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import re
from fuzzywuzzy import fuzz

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load bias keywords dataset
df_keywords = pd.read_csv("final_bias_keywords_dataset.csv")
df_keywords.dropna(subset=["word", "category", "suggestion"], inplace=True)
df_keywords["word"] = df_keywords["word"].astype(str).str.lower().str.strip()



def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()


def match_fuzzy_keywords(text, bias_df, hide_categories=[]):
    text_clean = clean_text(text)
    results = []
    seen = set()

    for _, row in bias_df.iterrows():
        category = str(row['category']).lower()
        if category in [c.lower() for c in hide_categories]:
            continue

        word = str(row['word']).lower()
        if fuzz.partial_ratio(word, text_clean) >= 85 and word not in seen:
            seen.add(word)
            results.append({
                "matched_word": word,
                "original_word": row["word"],
                "category": row["category"],
                "suggestion": row["suggestion"],
                "severity": "medium"  # Optional
            })
    return results


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        text = data.get("text", "")
        hide_categories = data.get("hide_categories", [])  # optional

        if not text:
            return jsonify({"error": "Text is required"}), 400

        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        keywords = match_fuzzy_keywords(text, df_keywords, hide_categories)

        score = max(0, 100 - len(keywords) * 10)

        return jsonify({
            "bias": prediction == "biased",
            "suggestions": keywords,
            "score": score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/add_keyword", methods=["POST"])
def add_keyword():
    global df_keywords

    try:
        data = request.get_json()
        category = data.get("category", "").strip()
        word = data.get("word", "").strip().lower()
        suggestion = data.get("suggestion", "").strip()

        if not word or not category or not suggestion:
            return jsonify({"error": "All fields are required"}), 400

        if word in df_keywords["word"].tolist():
            return jsonify({"message": "Keyword already exists"}), 200

        new_row = {
            "category": category,
             "word": word,
            "suggestion": suggestion
        }

        # Add to in-memory DataFrame
        df_keywords = pd.concat([df_keywords, pd.DataFrame([new_row])], ignore_index=True)

        # Append to CSV file
        pd.DataFrame([{
            "category": category,
             "word": word,
            "suggestion": suggestion
        }]).to_csv("biased_job_descriptions_with_category.csv", mode="a", header=False, index=False)

        return jsonify({"message": "Keyword added successfully."}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
