from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import pandas as pd
import os
import json
from predict_dominattraits import predict_personality  # Ensure this module exists

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define survey questions
questions = [
    "I am the life of the party", "I don't talk a lot", "I feel comfortable around people", "I keep in the background",
    "I start conversations", "I have little to say", "I talk to a lot of different people at parties",
    "I don't like to draw attention to myself", "I don't mind being the center of attention", "I am quiet around strangers",
    "I feel little concern for others", "I am interested in people", "I insult people", "I sympathize with others' feelings",
    "I am not interested in other people's problems", "I have a soft heart", "I am not really interested in others",
    "I take time out for others", "I feel others' emotions", "I make people feel at ease",
    "I am always prepared", "I leave my belongings around", "I pay attention to details", "I make a mess of things",
    "I get chores done right away", "I often forget to put things back in their proper place", "I like order",
    "I shirk my duties", "I follow a schedule", "I am exacting in my work",
    "I get stressed out easily", "I am relaxed most of the time", "I worry about things", "I seldom feel blue",
    "I am easily disturbed", "I get upset easily", "I change my mood a lot", "I have frequent mood swings",
    "I get irritated easily", "I often feel blue",
    "I have a rich vocabulary", "I have difficulty understanding abstract ideas", "I have a vivid imagination",
    "I am not interested in abstract ideas", "I have excellent ideas", "I do not have a good imagination",
    "I am quick to understand things", "I use difficult words", "I spend time reflecting on things", "I am full of ideas",
]

# Number of questions per page
QUESTIONS_PER_PAGE = 5

@app.route("/", methods=["GET", "POST"])
def username():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if not username:
            return redirect(url_for("username"))  # Prevent empty usernames
        session.clear()
        session["username"] = username
        session["responses"] = {}
        session.modified = True
        return redirect(url_for("survey", page=1))
    return render_template("username.html")

@app.route("/survey/<int:page>", methods=["GET", "POST"])
def survey(page):
    if "username" not in session:
        return redirect(url_for("username"))

    start = (page - 1) * QUESTIONS_PER_PAGE
    end = min(start + QUESTIONS_PER_PAGE, len(questions))
    questions_subset = questions[start:end]

    if "responses" not in session:
        session["responses"] = {}

    if request.method == "POST":
        new_responses = dict(session["responses"])

        for key, value in request.form.items():
            if key.startswith("q"):  
                try:
                    new_responses[key] = float(value)
                except ValueError:
                    new_responses[key] = 3.0 

        session["responses"] = new_responses
        session.modified = True

        if end < len(questions):
            return redirect(url_for("survey", page=page + 1))
        return redirect(url_for("results"))

    previous_responses = {
        f"q{i}": session["responses"].get(f"q{i}", "3") for i in range(start, end)
    }

    return render_template(
        "survey.html",
        questions=questions_subset,
        page=page,
        total_pages=(len(questions) + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE,
        previous_responses=previous_responses,
        QUESTIONS_PER_PAGE=QUESTIONS_PER_PAGE
    )

@app.route("/results")
def results():
    if "username" not in session or "responses" not in session:
        return redirect(url_for("username"))

    username = session["username"].replace(" ", "_")
    file_name = f"survey_results_{username}.json"
    file_path = os.path.join("static", file_name)

    responses = {}
    for i, q in enumerate(questions):
        key = f"q{i}"
        try:
            responses[q] = float(session["responses"].get(key, "3"))
        except ValueError:
            responses[q] = 3.0 

    df = pd.DataFrame([responses])

    try:
        trait_scores, dominant_traits, _ = predict_personality(df)
        trait_scores_json = trait_scores.to_dict(orient="records")[0]
    except Exception as e:
        trait_scores_json = {}
        dominant_traits = []

    result_data = {
        "username": username,
        "trait_scores": trait_scores_json,
        "dominant_traits": dominant_traits[:2]
    }

    with open(file_path, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    return render_template(
        "results.html",
        username=username,
        trait_scores=trait_scores_json,
        dominant_traits=result_data["dominant_traits"],
        file_path=url_for("download_results", filename=file_name)
    )

@app.route("/download/<filename>")
def download_results(filename):
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
