from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import math
import os

app = Flask(__name__)

# Define the questions for the personality questionnaire
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

# Pagination: 5 questions per page
QUESTIONS_PER_PAGE = 5

@app.route('/')
def get_username():
    """
    Home route to collect the username before starting the survey.
    """
    return render_template('username.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """
    Main route for displaying and handling the survey pages.
    """
    if request.method == 'POST':
        # Handle POST request: Save current responses and redirect to the next page
        username = request.form.get('username')
        page = int(request.form.get('page', 1))  # Default to page 1 if missing
        responses = {key: request.form[key] for key in request.form if key.startswith("question_")}
        
        # Save responses to a file named after the username
        filename = f"{username}_responses.xlsx"
        if os.path.exists(filename):
            existing_df = pd.read_excel(filename)
            new_df = pd.DataFrame([responses])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_excel(filename, index=False)
        else:
            pd.DataFrame([responses]).to_excel(filename, index=False)

        # Redirect to the next page or submit
        if page < math.ceil(len(questions) / QUESTIONS_PER_PAGE):
            return redirect(url_for('survey', username=username, page=page + 1))
        else:
            return redirect(url_for('submit', username=username))

    # Handle GET request: Display the current page
    username = request.args.get('username')
    page = int(request.args.get('page', 1))  # Default to page 1 if missing

    # Calculate the range of questions to display
    start_idx = (page - 1) * QUESTIONS_PER_PAGE
    end_idx = start_idx + QUESTIONS_PER_PAGE
    paginated_questions = questions[start_idx:end_idx]

    total_pages = math.ceil(len(questions) / QUESTIONS_PER_PAGE)
    return render_template(
        'survey.html',
        questions=paginated_questions,
        username=username,
        current_page=page,
        total_pages=total_pages,
    )

@app.route('/submit', methods=['GET'])
def submit():
    """
    Final route after completing the survey.
    """
    username = request.args.get('username')
    return f"Thank you, {username}, for completing the survey! Your responses have been saved."

if __name__ == '__main__':
    app.run(debug=True, port=5000)
