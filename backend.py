# backend_flask.py
from flask import Flask, request, jsonify

app = Flask(__name__)

# Dummy drug database
drug_db = {
    "paracetamol": {"interactions": ["ibuprofen"], "alternatives": ["acetaminophen"]},
    "ibuprofen": {"interactions": ["paracetamol"], "alternatives": ["naproxen"]},
    "amoxicillin": {"interactions": [], "alternatives": ["azithromycin"]},
}

@app.route("/verify", methods=["POST"])
def verify_prescription():
    data = request.get_json()
    text = data.get("text", "").lower()

    drugs = [drug for drug in drug_db if drug in text]

    # Interactions
    interactions = []
    for drug in drugs:
        for i in drug_db[drug]["interactions"]:
            if i in drugs:
                interactions.append(f"{drug} interacts with {i}")

    # Alternatives
    alternatives = {drug: drug_db[drug]["alternatives"] for drug in drugs}

    return jsonify({
        "drugs": drugs,
        "interactions": interactions,
        "alternatives": alternatives
    })

if __name__ == "__main__":
    app.run(debug=True, port=8000)
