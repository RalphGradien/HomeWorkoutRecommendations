import json
import re
from flask import Flask, render_template, request, redirect, url_for, session, logging
from flask_pymongo import PyMongo
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'exerciseapp'

# Process and modify the JSON data
with open('data/exercises.json', 'r', encoding='utf-8') as file:
    exercises = json.load(file)

# Modify exercise data to remove folder names in image filenames
for exercise in exercises:
    images = exercise["images"]
    exercise["images"] = [image.split('/')[-1] for image in images]

# Convert the modified exercise data to a pandas DataFrame
dataframe = pd.DataFrame(exercises)

# Save the DataFrame to a CSV file with a comma delimiter
dataframe.to_csv('data/exercises.csv', index=False, sep=',')

# Load the cleaned data from the CSV file
df = pd.read_csv('data/exercises_cleaned.csv')

# Convert the 'images' field from a string to a list and strip single quotes
df['images'] = df['images'].apply(lambda x: [image.strip(" '") for image in x.strip("[]").split(", ")])

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["exercisesdb"]
collection = db["exercises"]

# Drop the collection
collection.drop()

# Insert the CSV data into MongoDB
df_dict = df.to_dict(orient='records')
collection.insert_many(df_dict)

# Define the priority for user input fields
priority_fields = ['level', 'equipment', 'secondaryMuscles', 'force', 'mechanic', 'category']

# Define priority weights
priority_weights = [15, 10, 5, 3, 2, 1]

# Concatenate the relevant columns to create content for recommendations
df['content'] = df[priority_fields].apply(
    lambda row: (
        ' '.join([str(val) * weight for val, weight in zip(row, priority_weights)])
    ),
    axis=1
)

# Create a TF-IDF vectorizer to convert the content into numerical form
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

# Calculate the cosine similarity between exercises
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Define a global variable to store the selected primary muscle
selected_primary_muscle = ""

@app.route('/')
def home():
    return redirect(url_for('primary_muscle'))

@app.route('/primary-muscle', methods=['GET', 'POST']) 
def primary_muscle():
    if request.method == 'POST':
        primary_muscle = request.form.get('primaryMuscle')
        if primary_muscle:
            session['selected_primary_muscle'] = primary_muscle
            return redirect(url_for('user_input'))

    primary_muscles_left = ["Neck", "Shoulders", "Chest", "Biceps", "Forearms", "Abdominals", "Quadriceps", "Adductors", "Calves"]
    primary_muscles_right = ["Traps", "Triceps", "Lats", "Middle Back", "Lower Back", "Abductors", "Glutes", "Hamstrings", "Calves"]

    return render_template('primary_muscle.html', primary_muscles=primary_muscles_left + primary_muscles_right)

@app.route('/user-input')
def user_input():
    selected_primary_muscle = session.get('selected_primary_muscle', None)
    return render_template('user_input.html', primaryMuscle=selected_primary_muscle)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_exercises():
    selected_primary_muscle = session.get('selected_primary_muscle', None)

    if request.method == 'POST':
        # Get the user's input from the form fields
        user_input = {field: request.form.get(field) for field in priority_fields}
        user_input['primaryMuscle'] = selected_primary_muscle  # Set the selected primary muscle

        # Ensure that other values are not None before using them
        for field in priority_fields:
            if user_input[field] is None:
                user_input[field] = ""  # Set to an empty string or a default value

        # Extract and process the secondary muscles
        secondary_muscles = request.form.getlist('secondaryMuscles[]')
        secondary_muscles_str = ' '.join(secondary_muscles)

        user_content = (
            selected_primary_muscle * 20 + ' ' +
            ''.join(map(str, user_input['level'])) * priority_weights[0] + ' ' +
            ''.join(map(str, user_input['equipment'])) * priority_weights[1] + ' ' +
            secondary_muscles_str * priority_weights[2] + ' ' +
            ''.join(map(str, user_input['force'])) * priority_weights[3] + ' ' +
            ''.join(map(str, user_input['mechanic'])) * priority_weights[4] + ' ' +
            ''.join(map(str, user_input['category'])) * priority_weights[5]
        )

        
        # Convert user content into TF-IDF vector for recommendation
        user_tfidf_matrix = tfidf_vectorizer.transform([user_content])
        user_cosine_sim = linear_kernel(user_tfidf_matrix, tfidf_matrix)
        sim_scores = user_cosine_sim[0]
        exercise_indices = sim_scores.argsort()[::-1][:5]  # Select top 5 recommendations

        # Retrieve exercise data for the recommended exercises
        exercise_data = []

        # Convert exercise_indices to a list of exercise IDs
        exercise_ids = [str(df.iloc[index]["id"]) for index in exercise_indices]

        for exercise_id in exercise_ids:
            exercise_doc = collection.find_one({"id": exercise_id}) 
            if exercise_doc:
                if 'instructions' in exercise_doc:
                    # Replace "\n" with "<br>" to add line breaks in the instructions
                    exercise_doc['instructions'] = exercise_doc['instructions'].replace('.,', '<br>')
                exercise_data.append(exercise_doc)
        # Render the recommendations template with the results
        return render_template('recommendations.html', recommendations=exercise_data, user_input=user_input, primaryMuscle=selected_primary_muscle)

if __name__ == '__main__':
    app.run(debug=True)
