# 📚 Course Video Description Extractor

**A Streamlit web application to extract video descriptions from course outlines and generate downloadable TXT files. Supports Coursera 97 and Coursera 247 course formats.**

---

## Features

- Upload Excel course outlines (.xlsx) with video/module information.  
- Extract voiceover scripts or video descriptions.  
- Generate TXT files for each video/module.  
- Store TXT files in AWS S3 for easy download.  
- Supports multiple course types:
  - **Coursera 97** – lesson-based structure
  - **Coursera 247** – module-based structure  
- Simple Streamlit interface with user authentication.  

---

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/course-video-extractor.git
cd course-video-extractor

## Create a virtual environment (optional but recommended):

python -m venv venv
## Activate it
### Windows
venv\Scripts\activate
### Mac/Linux
source venv/bin/activate

## Install dependencies:

pip install -r requirements.txt
Set up environment variables in a .env file (example):

S3_BUCKET_NAME=your_bucket_name
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

## Usage

Run the Streamlit app:

streamlit run streamlit_app.py
Login with your credentials (if authentication is enabled).

Select the course type (Coursera 97 / Coursera 247).

Upload your course outline Excel file.

Click Extract Video Descriptions.

Download generated TXT files from the displayed links.

##  Dependencies
This project uses the following Python libraries:
boto3==1.39.16
python-dotenv==1.0.1
ftfy==6.3.1
pandas==2.3.0
streamlit==1.46.1
streamlit-authenticator==0.3.3
requests==2.32.5
Markdown==3.8.2
python-docx==1.2.0
beautifulsoup4==4.13.4
markitdown==0.1.3
openpyxl==3.1.5
html2docx==1.6.0

##  Project Structure  
course-video-extractor/
|-- streamlit_app.py # Main Streamlit app
|-- tenants/
| |-- coursera_97.py
| |-- coursera_247.py
|-- requirements.txt
|-- README.md
|-- .env.example # Example environment variables

## License
This project is open-source and available under the MIT License.

If you want, I can also make a **short GitHub repository description** (the one-line summary shown on GitHub) for your project so it looks professional and clickable.  

Do you want me to create that too?
