from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define questions for the personality survey
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
        session["username"] = request.form["username"]
        return redirect(url_for("survey", page=1))
    return render_template("username.html")

@app.route("/survey/<int:page>", methods=["GET", "POST"])
def survey(page):
    if "username" not in session:
        return redirect(url_for("username"))

    start = (page - 1) * QUESTIONS_PER_PAGE
    end = min(start + QUESTIONS_PER_PAGE, len(questions))
    questions_subset = questions[start:end]

    if request.method == "POST":
        for i in range(start, end):
            session[f"q{i}"] = request.form.get(f"q{i}", "3")

        if end < len(questions):
            return redirect(url_for("survey", page=page + 1))
        return redirect(url_for("results"))

    return render_template("survey.html", questions=questions_subset, page=page, total_pages=(len(questions) + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE)

@app.route("/results")
def results():
    if "username" not in session:
        return redirect(url_for("username"))

    # Ensure the static directory exists
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Save survey results properly formatted in an Excel sheet
    username = session["username"].replace(" ", "_")  # Sanitize filename
    file_name = f"survey_results_{username}.xlsx"
    file_path = os.path.join(static_dir, file_name)

    data = {q: [session.get(f"q{i}", "3")] for i, q in enumerate(questions)}
    df = pd.DataFrame(data)

    # Save the DataFrame properly formatted
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Survey Results")
        workbook = writer.book
        worksheet = writer.sheets["Survey Results"]
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value)  # Ensure headers are properly written

    return render_template("results.html", file_path=url_for('download_results', filename=file_name))

@app.route("/download/<filename>")
def download_results(filename):
    file_path = os.path.join("static", filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
