import time
from io import BytesIO
import pandas as pd
from typing import List, Tuple, Dict, Any
from streamlit_app import upload_to_s3, get_s3_url
from src.logging_config import get_logger
import streamlit as st

logger = get_logger(__name__)

def detect_header_type(df: pd.DataFrame) -> str:
    """Detect whether the outline uses Module or Lesson (whichever appears first)."""
    col = df[0].astype(str).str.strip().str.lower()

    module_indices = col[col.str.match(r"module\s+\d+")].index
    lesson_indices = col[col.str.match(r"lesson\s+\d+")].index

    first_module = module_indices.min() if len(module_indices) else None
    first_lesson = lesson_indices.min() if len(lesson_indices) else None

    if first_module is None and first_lesson is None:
        return None
    if first_lesson is None:
        return "module"
    if first_module is None:
        return "lesson"

    return "module" if first_module < first_lesson else "lesson"

def create_video_txt(lesson_num: int, video_num: int, video_title: str, voiceover: str, course_id: str) -> BytesIO:
    """Create a TXT file for a single video description."""
    try:
        logger.info(f"Creating TXT for Lesson {lesson_num}, Video {video_num}: {video_title}")
        
        # Create text content (text is already cleaned at Excel reading stage)
        header = st.session_state.get("unit_type", "Module").capitalize()
        content = f"{header} {lesson_num} - Video {video_num}: {video_title}\n"
        content += "=" * len(f"{header} {lesson_num} - Video {video_num}: {video_title}") + "\n\n"
        content += "Video Description:\n"
        content += "-" * 18 + "\n"
        content += voiceover + "\n"
        
        # Save to BytesIO buffer
        buffer = BytesIO()
        buffer.write(content.encode('utf-8'))
        buffer.seek(0)
        logger.info(f"Successfully created TXT for Lesson {lesson_num}, Video {video_num}")
        return buffer
        
    except Exception as e:
        logger.error(f"Error creating TXT file for Lesson {lesson_num}, Video {video_num}: {e}")
        st.error(f"Error creating TXT file: {e}")
        raise

def slice_for_unit(df: pd.DataFrame, unit_num: int, unit_type: str) -> pd.DataFrame:
    """Extract data for Module X or Lesson X dynamically."""
    try:
        tag = f"{unit_type} {unit_num}"
        col = df[0].astype(str).str.strip().str.lower()

        header_mask = col.str.contains(tag, regex=False, na=False)
        if not header_mask.any():
            raise ValueError(f"Header '{tag}' not found.")

        start = header_mask.idxmax()

        # Find next header of same type
        pattern = fr"{unit_type}\s+\d+"
        next_header = col.str.match(pattern, na=False)

        try:
            end = next_header[next_header & (next_header.index > start)].idxmax() - 1
        except ValueError:
            end = len(df) - 1

        return df.loc[start + 1: end]

    except Exception as e:
        logger.error(f"Error slicing {unit_type} {unit_num}: {e}")
        st.error(f"Error slicing {unit_type} {unit_num}: {e}")
        return pd.DataFrame()

def extract_video_data(module_df: pd.DataFrame, include_screencast: bool) -> List[Tuple[str, str]]:
    """Extract video data (title and voiceover) from a module DataFrame."""
    try:
        video_data = []
        col = module_df[0].astype(str).str.strip().str.lower()

        if include_screencast:
            # INCLUDE both video + screencast
            video_mask = col.str.contains("video", na=False) | col.str.contains("screencast", na=False)
        else:
            # EXCLUDE screencast (current behavior)
            video_mask = col.str.contains("video", na=False) & ~col.str.contains("screencast", na=False)

        video_rows = module_df[video_mask]
        
        if video_rows.empty:
            return video_data
            
        for _, row in video_rows.iterrows():
            title = str(row[1]).strip() if pd.notna(row[1]) else "Untitled Video"
            voiceover = str(row[3]).strip() if pd.notna(row[3]) else ""
            
            # Skip if voiceover is empty or invalid
            if not voiceover or voiceover.lower() in {"nan", ""}:
                continue
                
            video_data.append((title, voiceover))
        
        logger.info(f"Extracted {len(video_data)} videos from module data")
        return video_data
        
    except Exception as e:
        logger.error(f"Error extracting video data: {e}")
        st.error(f"Error extracting video data: {e}")
        return []

def find_unit_count(df: pd.DataFrame, unit_type: str) -> int:
    col = df[0].astype(str).str.strip().str.lower()
    pattern = fr"{unit_type}\s+\d+"
    return col.str.match(pattern, na=False).sum()

def process_module_outline(df: pd.DataFrame, course_id: str, include_screencast: bool) -> Dict[str, Any]:
    """Process the course outline and extract all video data."""
    try:
        unit_type = detect_header_type(df)
        st.session_state.unit_type = unit_type

        if not unit_type:
            return {"status": "error", "message": "No Module or Lesson headers found"}
        
        logger.info(f"Starting course outline processing for course: {course_id}")
        # Find number of modules
        num_units = find_unit_count(df, unit_type)
        logger.info(f"Found {num_units} modules in the course outline")
        
        if num_units == 0:
            logger.warning("No modules found in the Excel file")
            return {
                "status": "error",
                "message": "No modules found in the Excel file"
            }
        
        all_video_data = []
        s3_files = []
        
        # Process each module and create individual video files
        for unit_num in range(1, num_units + 1):
            unit_df = slice_for_unit(df, unit_num, unit_type)
            video_data = extract_video_data(unit_df, include_screencast)
            
            if video_data:
                logger.info(f"Processing module {unit_num} with {len(video_data)} videos")
                # Create individual DOCX file for each video
                for idx, (title, voiceover) in enumerate(video_data, 1):
                    # Track video data for summary
                    all_video_data.append({
                        "module": unit_num,
                        "video": idx,
                        "title": title,
                        "voiceover": voiceover
                    })
                    
                    # Create TXT file for this individual video
                    txt_buffer = create_video_txt(unit_num, idx, title, voiceover, course_id)
                    # Clean filename by removing special characters
                    clean_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    clean_title = clean_title.replace(' ', '_')[:50]  # Limit length and replace spaces
                    s3_key = f"video_descriptions/{course_id}/{unit_type}_{unit_num}_video_{idx}_{clean_title}.txt"
                    
                    if upload_to_s3(txt_buffer.getvalue(), s3_key):
                        s3_files.append({
                            "module": unit_num,
                            "video": idx,
                            "title": title,
                            "s3_key": s3_key,
                            "s3_url": get_s3_url(s3_key),
                            "file_type": "txt"
                        })
                    
                    # Clean up buffer
                    txt_buffer.close()
        
        # Create course metadata for return
        course_summary = {
            "course_id": course_id,
            "unit_type": unit_type,
            "total_units": num_units,
            "total_videos": len(all_video_data),
            "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "s3_files": s3_files
        }
        
        logger.info(f"Course processing completed successfully: {len(all_video_data)} videos processed, {len(s3_files)} files uploaded")
        return {
            "status": "success",
            "message": f"Successfully extracted {len(all_video_data)} video descriptions and created individual TXT files",
            "unit_type": unit_type,
            "total_units": num_units,
            "total_videos": len(all_video_data),
            "s3_files": s3_files,
            "course_data": course_summary
        }
        
    except Exception as e:
        logger.error(f"Course processing failed for {course_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }
