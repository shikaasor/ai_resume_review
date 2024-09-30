Resume Review Application

INTRODUCTION
The Resume Review Application is an AI-powered tool designed to streamline the hiring process by automatically analyzing resumes against job descriptions. Built with Streamlit and leveraging the Groq AI model, this application provides quick and insightful candidate assessments, helping HR professionals and recruiters identify the best matches efficiently.
Features

AI-Powered Analysis: Utilizes the Groq AI model to compare resumes against job descriptions.

Bulk Processing: Supports both individual PDF uploads and zip files containing multiple PDFs.

Instant Results: Provides immediate feedback on candidate suitability.

User-Friendly Interface: Built with Streamlit for a smooth, interactive user experience.

Export Functionality: Allows downloading of results in CSV format for further analysis.


INSTALLATION

Clone the repository:
git clone https://github.com/shikaasor/ai_resume_review
cd resume-review-app

Install the required dependencies:
pip install -r requirements.txt

Set up your environment variables:

Create a '.env' file in the root directory.

Add your Groq API key:
GROQ_API_KEY="your_api_key_here"


USAGE

Run the Streamlit app:
python -m streamlit run app.py

Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

Use the application:

Paste the job description in the provided text area.
Upload either a single PDF resume or a zip file containing multiple PDF resumes.
Click the "Submit" button to start the analysis.


View the results:

For single PDFs, the analysis will be displayed directly on the page.
For zip files, a downloadable CSV file will be generated with the analysis results.



Dependencies

streamlit
groq
PyPDF2
pandas
python-dotenv
Pillow

Icon
The application uses the following icon:

Page icon: static/icon.png
Logo: static/kaasor.png

Ensure these files are present in the specified locations for proper display.

Contributions to improve the Resume Review Application are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

License
MIT