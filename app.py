from flask import Flask, render_template, request, send_file
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load spaCy NER model for name entity recognition
nlp = spacy.load("en_core_web_sm")

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Predefined list of technologies for various job roles (this can be extended based on your requirements)
required_skills = {
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js", "PHP", "MySQL", "CSS3", "jQuery", "Bootstrap", "Vue.js", "Angular", "SASS", "MongoDB"],
    "Data Scientist": ["Python", "R", "Machine Learning", "Deep Learning", "TensorFlow", "Keras", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "SQL"],
    "Software Engineer": ["Java", "C++", "Python", "Algorithms", "Data Structures", "Software Development", "Git", "JUnit", "Spring"]
}

# Utility Functions
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def extract_entities(text):
    """Extracts names and emails from text."""
    # Extract emails using regex
    emails = re.findall(r'\S+@\S+', text)
    # Use spaCy to extract names
    doc = nlp(text)
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)
    if names:
        names = [" ".join(names[0])]
    return emails, names

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recruiter', methods=['GET', 'POST'])
def recruiter():
    results = []

    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_files = request.files.getlist('resume_files')

        processed_resumes = []
        for resume_file in resume_files:
            # Sanitize and save file
            sanitized_filename = secure_filename(resume_file.filename)
            resume_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
            resume_file.save(resume_path)

            # Extract and process resume text
            resume_text = extract_text_from_pdf(resume_path)
            emails, names = extract_entities(resume_text)

            # Default to Web Developer if no role is selected
            job_role = "Web Developer"
            required = required_skills.get(job_role, [])
            
            # Analyze present skills (those that exist in both job description and resume)
            resume_words = set(resume_text.split())
            present_skills = [skill for skill in required if skill.lower() in (word.lower() for word in resume_words)]

            # TF-IDF ranking
            tfidf_vectorizer = TfidfVectorizer()
            job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

            # Calculate similarity
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100  # Multiply by 100 to convert to percentage

            processed_resumes.append((names, emails, resume_text, present_skills, similarity))

        # Sort by similarity
        processed_resumes.sort(key=lambda x: x[4], reverse=True)

        # Enumerate to get ranking along with results
        results = [(i + 1, resume[0], resume[1], resume[3], round(resume[4], 2)) for i, resume in enumerate(processed_resumes)]

    return render_template('recruiter.html', results=results)

@app.route('/seeker', methods=['GET', 'POST'])
def seeker():
    present_skills = []
    missing_skills = []

    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_file = request.files.get('resume_file')

        # Sanitize and save file
        sanitized_filename = secure_filename(resume_file.filename)
        resume_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
        resume_file.save(resume_path)

        # Extract and process resume text
        resume_text = extract_text_from_pdf(resume_path)

        # Default to Web Developer if no role is selected
        job_role = "Web Developer"

        # Get required skills based on job role
        required = required_skills.get(job_role, [])
        
        # Analyze present skills (those that exist in both job description and resume)
        resume_words = set(resume_text.split())
        present_skills = [skill for skill in required if skill.lower() in (word.lower() for word in resume_words)]

        # Missing skills: Check which required skills are missing from the resume
        missing_skills = [skill for skill in required if skill.lower() not in (word.lower() for word in resume_words)]

    return render_template('seeker.html', present_skills=present_skills, missing_skills=missing_skills)

@app.route('/download_csv')
def download_csv():
    """Allows downloading the ranked resumes as a CSV."""
    csv_content = "Rank,Name,Email,Skills Present,Similarity\n"
    for rank, (names, emails, _, present_skills, similarity) in enumerate(results, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        skills = ", ".join(present_skills) if present_skills else "N/A"
        csv_content += f"{rank},{name},{email},{skills},{similarity}\n"

    csv_filename = "ranked_resumes.csv"
    csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
    with open(csv_path, "w") as csv_file:
        csv_file.write(csv_content)

    return send_file(csv_path, as_attachment=True, download_name="ranked_resumes.csv")

if __name__ == '__main__':
    app.run(debug=True)
