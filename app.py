import json
from flask import Flask, render_template, request
from flask_pymongo import PyMongo
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__, static_url_path='/static')
app.config["MONGO_URI"] = "mongodb://localhost:27017/exercisesdb"
mongo = PyMongo(app)

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

# # Drop the collection
# db["exercises"].drop()

# Insert the CSV data into MongoDB
df_dict = df.to_dict(orient='records')
collection.insert_many(df_dict)

# Define the priority for user input fields
priority_fields = ['primaryMuscles', 'level', 'equipment', 'secondaryMuscles', 'force', 'mechanic', 'category']

# Define priority weights
priority_weights = [20, 15, 10, 5, 3, 2, 1]

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

def get_content_based_recommendations(user_input, cosine_sim=cosine_sim):
    secondary_muscles = user_input.get('secondaryMuscles', [])  # Get the list of secondary muscles from the user input
    secondary_muscles_str = ''.join(secondary_muscles)  # Join the list into a string

    user_content = (
        ''.join(map(str, user_input['primaryMuscles'])) * priority_weights[0] + ' ' +
        ''.join(map(str, user_input['level'])) * priority_weights[1] + ' ' +
        ''.join(map(str, user_input['equipment'])) * priority_weights[2] + ' ' +
        secondary_muscles_str * priority_weights[3] + ' ' +
        ''.join(map(str, user_input['force'])) * priority_weights[4] + ' ' +
        ''.join(map(str, user_input['mechanic'])) * priority_weights[5] + ' ' +
        ''.join(map(str, user_input['category'])) * priority_weights[6]
    )

    user_tfidf_matrix = tfidf_vectorizer.transform([user_content])
    user_cosine_sim = linear_kernel(user_tfidf_matrix, tfidf_matrix)
    sim_scores = user_cosine_sim[0]
    exercise_indices = sim_scores.argsort()[::-1][0:5]
    recommended_exercises = df['id'].iloc[exercise_indices]

    exercise_data = []
    for exercise_id in recommended_exercises:
        exercise_doc = collection.find_one({"id": exercise_id})

        # Clean the instructions: Remove square brackets and split into a list
        if 'instructions' in exercise_doc:
            instructions = exercise_doc['instructions']
            instructions = instructions.strip("[]")
            instructions = instructions.split("', '")
            exercise_doc['instructions'] = instructions

        exercise_data.append(exercise_doc)
    return exercise_data

# Define a route for the root URL
@app.route('/')
def index():
    return render_template('index.html', df=df)

@app.route('/recommend', methods=['POST'])
def recommend_exercises():
    user_input = {field: request.form.get(field) for field in priority_fields}
    # Pass the list of secondary muscles
    secondary_muscles = request.form.getlist('secondaryMuscles')
    user_input['secondaryMuscles'] = secondary_muscles
    content_based_recommendations = get_content_based_recommendations(user_input)
    return render_template('recommendations.html', recommendations=content_based_recommendations, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
