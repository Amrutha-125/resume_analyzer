 Job Application System

This project is a *Job Application System* built using *Flask*. It provides two main functionalities: a *Recruiter Portal* and a *Job Seeker Portal* .

 Features

 Recruiter Portal
- Upload multiple resumes in PDF format.
- Extract names, emails, and skills from resumes.
- Rank resumes based on their similarity to the provided job description.
- Display skills present and missing in each resume.
 Job Seeker Portal
- Upload a resume in PDF format.
- Compare the resume against a provided job description.
- Display skills present and missing compared to the job description.

 Technologies Used

- **Flask** for web application framework.
- **spaCy** for name extraction.
- **Scikit-learn** for TF-IDF vectorization and similarity scoring.
- **PyPDF2** for PDF text extraction.
- **Regular Expressions** for email extraction.

 Future Improvements

- Support additional file formats such as DOCX.
- Implement advanced NLP techniques for better skill extraction.
- Add user authentication for secure access.



This system is designed to simplify the recruitment and job-seeking process using efficient text analysis techniques.
