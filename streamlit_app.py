import os
import uuid
import ftfy
import boto3
import traceback
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from typing import Optional

from src.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Course Video Description Extractor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        color: #4caf50;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f5e8;
        margin: 1rem 0;
    }
    .error-message {
        color: #f44336;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffeaea;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "course_id" not in st.session_state:
    st.session_state.course_id = f"course_{uuid.uuid4().hex[:8]}"
if "processing_status" not in st.session_state:
    st.session_state.processing_status = "idle"
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "s3_files" not in st.session_state:
    st.session_state.s3_files = []
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None
if "name" not in st.session_state:
    st.session_state.name = None
if "username" not in st.session_state:
    st.session_state.username = None

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def setup_authenticator():
    """Set up the streamlit authenticator."""
    try:
        # Check if authenticator is already in session state
        if 'authenticator' in st.session_state:
            return st.session_state.authenticator
        
        auth_config = st.secrets["auth"]
        
        # Build credentials dictionary
        credentials = {
            "usernames": {}
        }
        
        for username in auth_config["credentials"]["usernames"]:
            user_data = auth_config["credentials"][username]
            credentials["usernames"][username] = {
                "email": user_data["email"],
                "name": user_data["name"],
                "password": user_data["password"]
            }
        
        # Create authenticator
        authenticator = stauth.Authenticate(
            credentials,
            auth_config["cookie_name"],
            auth_config["cookie_key"],
            auth_config["cookie_expiry_days"]
        )
        
        # Store in session state to avoid recreation
        st.session_state.authenticator = authenticator
        logger.info("Authentication system initialized successfully")
        return authenticator
    except Exception as e:
        logger.error(f"Authentication setup failed: {str(e)}")
        st.error(f"Authentication setup failed: {str(e)}")
        return None

def check_authentication():
    """Check if user is authenticated and handle login/logout."""
    authenticator = setup_authenticator()
    
    if not authenticator:
        st.error("Authentication system not properly configured.")
        st.stop()
    
    # Perform authentication with error handling
    try:
        # Try different methods based on streamlit-authenticator version
        try:
            # Method 1: Try with parameters (newer versions)
            login_result = authenticator.login(
                location='main',
                key='login_form'
            )
        except TypeError:
            # Method 2: Try without parameters (older versions)
            try:
                login_result = authenticator.login('main', 'login_form')
            except TypeError:
                # Method 3: Try with just location
                login_result = authenticator.login('main')
        
        # Handle different return types from streamlit-authenticator
        if login_result is None:
            # Fallback to session state values if login returns None
            authentication_status = st.session_state.get('authentication_status')
            name = st.session_state.get('name')
            username = st.session_state.get('username')
        elif isinstance(login_result, tuple) and len(login_result) == 3:
            # Unpack tuple if returned
            name, authentication_status, username = login_result
        else:
            # Try to get from session state if unexpected return
            authentication_status = st.session_state.get('authentication_status')
            name = st.session_state.get('name')
            username = st.session_state.get('username')
        
        # Store authentication results in session state
        if authentication_status is not None:
            st.session_state.authentication_status = authentication_status
        if name is not None:
            st.session_state.name = name
        if username is not None:
            st.session_state.username = username
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        st.error(f"Authentication error: {e}")
        st.error("Please check your streamlit secrets configuration.")
        st.stop()
    
    # Get current authentication status
    current_auth_status = st.session_state.get('authentication_status')
    
    # Check authentication status
    if current_auth_status == False:
        st.error('Username/password is incorrect')
        st.stop()
    elif current_auth_status == None:
        st.warning('Please enter your username and password')
        st.stop()
    elif current_auth_status:
        return True
    
    return False

def read_excel_from_upload(uploaded_file, sheet_name: str = "Outline") -> Optional[pd.DataFrame]:
    """Read Excel file from Streamlit upload with proper encoding handling."""
    try:
        logger.info(f"Reading Excel file: {uploaded_file.name}, sheet: {sheet_name}")
        
        # Try multiple approaches to handle encoding issues
        df = None
        
        # Method 1: Try with openpyxl engine and specific encoding handling
        try:
            df = pd.read_excel(
                uploaded_file, 
                sheet_name=sheet_name, 
                header=None,
                engine='openpyxl',
                dtype=str  # Read all as strings to preserve encoding
            )
            logger.info("Successfully read Excel with openpyxl engine")
        except Exception as e1:
            logger.warning(f"Method 1 failed: {e1}")
            
            # Method 2: Reset file pointer and try with default settings
            try:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, dtype=str)
                logger.info("Successfully read Excel with default method")
            except Exception as e2:
                logger.warning(f"Method 2 failed: {e2}")
                raise e2
        
        if df is not None:
            # Apply encoding fixes to all string cells in the dataframe
            logger.info("Applying encoding fixes to dataframe")
            for col in df.columns:
                df[col] = df[col].apply(lambda x: clean_text_encoding(str(x)) if pd.notna(x) else x)
            
            logger.info(f"Successfully read and cleaned Excel file with {len(df)} rows")
            return df
        else:
            raise ValueError("Failed to read Excel file with any method")
            
    except Exception as e:
        logger.error(f"Error reading Excel file {uploaded_file.name}: {e}")
        st.error(f"Error reading Excel file: {e}")
        return None

def upload_to_s3(file_content: bytes, s3_key: str, content_type: str = "text/plain") -> bool:
    """Upload file content to S3."""
    try:
        logger.info(f"Uploading file to S3: {s3_key}")
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type,
            ContentDisposition=f'attachment; filename="{os.path.basename(s3_key)}"'
        )
        logger.info(f"Successfully uploaded file to S3: {s3_key}")
        return True
    except Exception as e:
        logger.error(f"S3 upload failed for {s3_key}: {e}")
        st.error(f"S3 upload failed: {e}")
        return False

def get_s3_url(s3_key: str) -> str:
    """Get public S3 URL for a key."""
    return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


def clean_text_encoding(text: str) -> str:
    """
    Use ftfy to repair mojibake while preserving newlines.
    """
    if not text or pd.isna(text):
        return ""
    # ftfy fixes the common mis-encodings automatically
    cleaned = ftfy.fix_text(str(text), normalization="NFC").replace("—", "-")
    # Trim trailing spaces but keep paragraph structure
    return "\n".join(line.rstrip() for line in cleaned.splitlines())

def main():
    """Main Streamlit application with Course 1 & Course 2 support."""
    if not check_authentication():
        return

    user_name = st.session_state.get('name', 'Unknown')
    logger.info(f"User session started: {user_name}")

    # Header
    st.markdown('<div class="main-header">📚 Course Video Description Extractor</div>', unsafe_allow_html=True)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.write(f"👤 Welcome, {st.session_state.get('name', 'User')}")
        authenticator = setup_authenticator()
        if authenticator:
            authenticator.logout('Logout', 'sidebar', key='logout_button')
        st.divider()
        
        # Course ID
        st.write(f"🆔 Course ID: `{st.session_state.course_id}`")
        
        if st.button("🔄 Generate New Course ID"):
            st.session_state.course_id = f"course_{uuid.uuid4().hex[:8]}"
            # Clear previous data and uploaded file
            st.session_state.extracted_data = None
            st.session_state.s3_files = None
            st.session_state.processing_status = None
            st.rerun()

        st.divider()

        # 📚 Course type selection
        course_type = st.radio(
            "Select Course Type:",
            ("Coursera 97", "Coursera 247"),
            index=0
        )

        # Reset previous data if course type changes
        if 'course_type' in st.session_state:
            if st.session_state.course_type != course_type:
                st.session_state.extracted_data = None
                st.session_state.s3_files = None
                st.session_state.processing_status = None

        st.session_state.course_type = course_type  # persist selection

    # ---------- Main upload section ----------
    st.subheader("📤 Upload Course Outline")
    # Use dynamic key based on course type + course ID to reset uploader when needed
    file_uploader_key = f"uploaded_file_{st.session_state.get('course_type','')}_{st.session_state.get('course_id','')}"
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx)",
        type=['xlsx'],
        key=file_uploader_key,  # bind to session state
        help="Upload your course outline Excel file with video descriptions"
    )

    if uploaded_file is not None:
        st.info(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")

        # Sheet name input
        sheet_name = st.text_input("Sheet Name", value="Outline")

        if st.button("🚀 Extract Video Descriptions", type="primary"):
            if not S3_BUCKET_NAME:
                logger.error("S3_BUCKET_NAME environment variable not set")
                st.error("❌ S3_BUCKET_NAME environment variable not set")
                return

            with st.spinner("Processing course outline..."):
                try:
                    df = read_excel_from_upload(uploaded_file, sheet_name)
                    if df is None:
                        st.error("❌ Failed to read Excel file")
                        return

                    # ---------- Choose processor ----------
                    if "Coursera 97" in course_type:
                        from tenants.coursera_97 import process_course_outline
                        result = process_course_outline(df, st.session_state.course_id)
                    elif "Coursera 247" in course_type:
                        from tenants.coursera_247 import process_module_outline
                        result = process_module_outline(df, st.session_state.course_id)

                    # ---------- Display results ----------
                    if result["status"] == "success":
                        st.session_state.extracted_data = result
                        st.session_state.s3_files = result["s3_files"]
                        st.session_state.processing_status = "completed"

                        if "Coursera 97" in course_type:
                            st.markdown(f"""
                            <div class="success-message">
                                ✅ Extraction completed!<br>
                                📊 Lessons: {result.get('total_lessons', result.get('total_modules', 0))}<br>
                                🎥 Videos: {result['total_videos']}<br>
                                📄 TXT Files: {len(result['s3_files'])}
                            </div>
                            """, unsafe_allow_html=True)
                        elif "Coursera 247" in course_type:
                            st.markdown(f"""
                            <div class="success-message">
                                ✅ Extraction completed!<br>
                                📊 Modules: {result.get('total_lessons', result.get('total_modules', 0))}<br>
                                🎥 Videos: {result['total_videos']}<br>
                                📄 TXT Files: {len(result['s3_files'])}
                            </div>
                            """, unsafe_allow_html=True)
                            
                    else:
                        st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    st.code(traceback.format_exc())

    # ---------- Results section ----------
    if st.session_state.get("processing_status") == "completed" and st.session_state.extracted_data:
        st.divider()
        st.subheader("📊 Extraction Results")

        result = st.session_state.extracted_data
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Lessons/Modules", result.get('total_lessons', result.get('total_modules', 0)))
        with c2:
            st.metric("Total Videos", result['total_videos'])
        with c3:
            st.metric("Files in S3", len(result['s3_files']))

        st.subheader("📄 Generated TXT Files")
        for f in result["s3_files"]:
            label = f"Lesson {f.get('lesson')}" if "lesson" in f else f"Module {f.get('module')}"
            st.write(f"📝 **{label} - Video {f['video']}**: {f['title']} → [Download TXT]({f['s3_url']})")

        if st.checkbox("👀 Show Raw JSON Data"):
            st.json(result["course_data"])

    # ---------- Dynamic Instructions ----------
    st.divider()
    st.subheader("📋 Instructions")

    if "Coursera 97" in st.session_state.get("course_type", "Coursera 97"):
        st.markdown("""
        1. **Upload** your course outline Excel file (.xlsx format)
        2. **Specify** the sheet name (default: "Outline")
        3. **Click** "Extract Video Descriptions" to process the file
        4. **Download** individual TXT files from S3 links

        **Excel Format Requirements:**
        - Column A: "Lesson 1", "Lesson 2", etc.
        - Rows starting with "Video" or "Introductory Video"
        - Column B: Video titles
        - Column D: Voiceover scripts

        **Output Format:**
        - One TXT file per video (uploaded to S3)
        - Clean voiceover text
        - Ready for sharing or editing
        """)
    elif  "Coursera 247" in st.session_state.get("course_type", "Coursera 247"):
        st.markdown("""
        1. **Upload** your course outline Excel file (.xlsx format)
        2. **Specify** the sheet name (default: "Outline")
        3. **Click** "Extract Video Descriptions" to process the file
        4. **Download** individual TXT files from S3 links

        **Excel Format Requirements:**
        - Column A: "Module 1", "Module 2", etc.
        - Rows starting with "vignette video", "instructional video", "screencast", "video"
        - Column B: Video titles
        - Column D: Voiceover scripts

        **Output Format:**
        - One TXT file per video (uploaded to S3)
        - Clean video descriptions without metadata
        - Ready for sharing and editing
        """)

if __name__ == "__main__":

    main() 

