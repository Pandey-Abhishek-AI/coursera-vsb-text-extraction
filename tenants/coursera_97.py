import time
import pandas as pd
from io import BytesIO
from typing import List, Tuple, Dict, Any
from streamlit_app import upload_to_s3, get_s3_url
from src.logging_config import get_logger
import streamlit as st

logger = get_logger(__name__)

def create_video_txt(lesson_num: int, video_num: int, video_title: str, voiceover: str, course_id: str) -> BytesIO:
    """Create a TXT file for a single video description."""
    try:
        logger.info(f"Creating TXT for Lesson {lesson_num}, Video {video_num}: {video_title}")
        
        # Create text content (text is already cleaned at Excel reading stage)
        content = f"Lesson {lesson_num} - Video {video_num}: {video_title}\n"
        content += "=" * len(f"Lesson {lesson_num} - Video {video_num}: {video_title}") + "\n\n"
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

def slice_for_lesson(df: pd.DataFrame, lesson_num: int) -> pd.DataFrame:
    """Extract data for a specific lesson."""
    try:
        tag = f"Lesson {lesson_num}"
        header_mask = df[0].astype(str).str.contains(tag, case=False, na=False, regex=False)
        
        if not header_mask.any():
            raise ValueError(f"Header '{tag}' not found in Column A.")
            
        start = header_mask.idxmax()
        
        # Find the next lesson header
        next_header = df[0].astype(str).str.match(r"Lesson\s+\d+", na=False)
        try:
            end = next_header[next_header & (next_header.index > start)].idxmax() - 1
        except ValueError:
            end = len(df) - 1
            
        result_df = df.loc[start + 1: end]
        logger.info(f"Successfully sliced data for Lesson {lesson_num}: {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Error slicing data for Lesson {lesson_num}: {e}")
        st.error(f"Error slicing data for Lesson {lesson_num}: {e}")
        return pd.DataFrame()

def extract_video_data(lesson_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Extract video data (title and voiceover) from a lesson DataFrame."""
    try:
        video_data = []
        # Look for rows starting with "video" or "introductory video"
        video_mask = lesson_df[0].astype(str).str.strip().str.lower().str.startswith(("video", "introductory video"))
        video_rows = lesson_df[video_mask]
        
        if video_rows.empty:
            return video_data
            
        for _, row in video_rows.iterrows():
            title = str(row[1]).strip() if pd.notna(row[1]) else "Untitled Video"
            voiceover = str(row[3]).strip() if pd.notna(row[3]) else ""
            
            # Skip if voiceover is empty or invalid
            if not voiceover or voiceover.lower() in {"nan", ""}:
                continue
                
            video_data.append((title, voiceover))
        
        logger.info(f"Extracted {len(video_data)} videos from lesson data")
        return video_data
        
    except Exception as e:
        logger.error(f"Error extracting video data: {e}")
        st.error(f"Error extracting video data: {e}")
        return []

def find_lesson_count(df: pd.DataFrame) -> int:
    """Find the number of lessons in the DataFrame."""
    try:
        lesson_pattern = df[0].astype(str).str.match(r"Lesson\s+\d+", na=False)
        count = lesson_pattern.sum()
        logger.info(f"Found {count} lessons in dataframe")
        return count
    except Exception as e:
        logger.error(f"Error counting lessons: {e}")
        st.error(f"Error counting lessons: {e}")
        return 0

def process_course_outline(df: pd.DataFrame, course_id: str, course_code: str) -> Dict[str, Any]:
    """Process the course outline and extract all video data."""
    try:
        logger.info(f"Starting course outline processing for course: {course_id}")
        # Find number of lessons
        num_lessons = find_lesson_count(df)
        logger.info(f"Found {num_lessons} lessons in the course outline")
        
        if num_lessons == 0:
            logger.warning("No lessons found in the Excel file")
            return {
                "status": "error",
                "message": "No lessons found in the Excel file"
            }
        
        all_video_data = []
        s3_files = []
        
        # Process each lesson and create individual video files
        for lesson_num in range(1, num_lessons + 1):
            lesson_df = slice_for_lesson(df, lesson_num)
            video_data = extract_video_data(lesson_df)
            
            if video_data:
                logger.info(f"Processing lesson {lesson_num} with {len(video_data)} videos")
                # Create individual DOCX file for each video
                for idx, (title, voiceover) in enumerate(video_data, 1):
                    # Track video data for summary
                    all_video_data.append({
                        "lesson": lesson_num,
                        "video": idx,
                        "title": title,
                        "voiceover": voiceover
                    })
                    
                    # Create TXT file for this individual video
                    txt_buffer = create_video_txt(lesson_num, idx, title, voiceover, course_id)
                    # Clean filename by removing special characters
                    clean_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    clean_title = clean_title.replace(' ', '_')  # Replace spaces
                    s3_key = f"video_descriptions/{course_id}/{course_code}_L{lesson_num}_V{idx}_{clean_title}.txt"
                    
                    if upload_to_s3(txt_buffer.getvalue(), s3_key):
                        s3_files.append({
                            "lesson": lesson_num,
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
            "total_lessons": num_lessons,
            "total_videos": len(all_video_data),
            "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "s3_files": s3_files
        }
        
        logger.info(f"Course processing completed successfully: {len(all_video_data)} videos processed, {len(s3_files)} files uploaded")
        return {
            "status": "success",
            "message": f"Successfully extracted {len(all_video_data)} video descriptions and created individual TXT files",
            "total_lessons": num_lessons,
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

