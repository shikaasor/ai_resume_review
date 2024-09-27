import streamlit as st
from groq import Groq
import os
import zipfile
import csv
import pandas as pd
import PyPDF2 as pdf
from dotenv import load_dotenv
from typing import List, Dict, Optional
from PIL import Image
import requests
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_pdf_text(uploaded_file) -> Optional[str]:
    try:
        reader = pdf.PdfReader(uploaded_file)
        return "".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return None

def generate_review(text: str, jd: str) -> Optional[Dict[str, str]]:
    try:
        prompt = f"""
        As a seasoned Human Resource Expert, analyze the provided resume and determine if the candidate is a good fit based on the job description.
        Resume: {text}
        Job Description: {jd}

        Please provide the output of your analysis in the following exact format. It is crucial that you follow this format strictly without any additional text:
        
        Candidate Name: [Name]
        Summary of Analysis: [Brief summary]
        Percentage Suitability: [Percentage]

        Do not include any other information or commentary.
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile",
        )
        
        content = response.choices[0].message.content
        lines = content.strip().split('\n')

        if len(lines) < 3:
            st.error("The response format is incorrect. Please check the model output.")
            return None

        return {
            'Candidate Name': lines[0].split(': ', 1)[1],
            'Summary of Analysis': lines[1].split(': ', 1)[1],
            'Percentage Suitability': lines[2].split(': ', 1)[1]
        }
    except Exception as e:
        st.error(f"Error generating review: {e}")
        return None

def process_zip_file(zip_file, jd: str) -> List[Dict[str, str]]:
    reviews = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for doc in zip_ref.namelist():
            if doc.endswith('.pdf'):
                with zip_ref.open(doc) as f:
                    text = extract_pdf_text(f)
                    print(text)
                    if text:
                        review = generate_review(text, jd)
                        if review:
                            reviews.append(review)
    return reviews

def main():
    logo_url = "static\kaasor.png"
    favicon_url = "static\icon.png"
    st.set_page_config(layout="wide", page_title="Resume Review", page_icon="static\icon.png")
    
    # Display logo
    st.image(logo_url, width=200)

    
    st.markdown("<h1 style='text-align: center; color: #A5FFFD; border: 2px solid #30B0C2; border-radius: 10px; padding: 10px;'>RESUME REVIEW</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Built by Asor</p>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #F5FF84;'>The Perfect Hire</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>AI-powered resume review for candidate matching</h3>", unsafe_allow_html=True)

    jd = st.text_area("Paste the Job Description")
    uploaded_file = st.file_uploader("Upload zipped PDF Resumes or a single resume PDF file", type=('zip', 'pdf'), accept_multiple_files=True, help="Please upload a PDF file or a zipped folder containing PDFs")

    if st.button("Submit"):
        if uploaded_file and jd:
            for file in uploaded_file:
                file_type = file.type
                print(file_type)
                
        
                if file_type == "application/x-zip-compressed" or file_type == "application/zip":
                    reviews = process_zip_file(file, jd)
                    if reviews:
                        df = pd.DataFrame(reviews)
                        st.dataframe(df)
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Review", csv, "reviews.csv", "text/csv")
                    else:
                        st.warning("No valid PDFs found in the zip file.")
                
                elif file_type == "application/pdf":
                    text = extract_pdf_text(file)
                    if text:
                        review = generate_review(text, jd)
                        if review:
                            for key, value in review.items():
                                st.subheader(f"{key}:")
                                st.write(value)
        else:
            st.error("Please upload a zipped folder or PDF file")

if __name__ == "__main__":
    main()