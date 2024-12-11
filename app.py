from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

# Create Flask App
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

@app.route('/')
def get_username():
    # Render the username input page
    return render_template('username.html')

@app.route('/survey', methods=['POST'])
def survey():
    # Get the username from the form and store it in a session
    username = request.form.get('username')
    if not username or username.strip() == "":
        return "Error: Username is required.", 400  # Return error for missing username
    
    return render_template('survey.html', questions=questions, username=username)

@app.route('/submit', methods=['POST'])
def submit():
    # Collect the username and ratings from the form
    username = request.form.get('username')
    ratings = {question: request.form.get(question) for question in questions}
    
    # Convert ratings to a DataFrame
    ratings_df = pd.DataFrame([ratings])
    
    # Save the results to a user-specific Excel file
    filename = f"{username.strip()}.xlsx"
    ratings_df.to_excel(filename, index=False)
    
    return f"Thank you, {username}, for submitting your ratings!"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)

