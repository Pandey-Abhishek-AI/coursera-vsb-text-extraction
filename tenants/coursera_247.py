import time
from io import BytesIO
import pandas as pd
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
        content = f"Module {lesson_num} - Video {video_num}: {video_title}\n"
        content += "=" * len(f"Module {lesson_num} - Video {video_num}: {video_title}") + "\n\n"
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

def slice_for_module(df: pd.DataFrame, module_num: int) -> pd.DataFrame:
    """Extract data for a specific module."""
    try:
        tag = f"module {module_num}"
        header_mask = df[0].astype(str).str.contains(tag, case=False, na=False, regex=False)
        
        if not header_mask.any():
            raise ValueError(f"Header '{tag}' not found in Column A.")
            
        start = header_mask.idxmax()
        
        # Find the next module header
        next_header = df[0].astype(str).str.match(r"Module\s+\d+", na=False)
        try:
            end = next_header[next_header & (next_header.index > start)].idxmax() - 1
        except ValueError:
            end = len(df) - 1
            
        result_df = df.loc[start + 1: end]
        logger.info(f"Successfully sliced data for Module {module_num}: {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Error slicing data for Module {module_num}: {e}")
        st.error(f"Error slicing data for Module {module_num}: {e}")
        return pd.DataFrame()

def extract_video_data(module_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Extract video data (title and voiceover) from a module DataFrame."""
    try:
        video_data = []
        # Look for rows contain "video" and also not contain screencast
        video_mask = module_df[0].astype(str).str.strip().str.lower().str.contains("video", na=False) & ~module_df[0].astype(str).str.strip().str.lower().str.contains("screencast", na=False)
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

def find_module_count(df: pd.DataFrame) -> int:
    """Find the number of modules in the DataFrame."""
    try:
        module_pattern = df[0].astype(str).str.match(r"Module\s+\d+", na=False)
        count = module_pattern.sum()
        logger.info(f"Found {count} modules in dataframe")
        return count
    except Exception as e:
        logger.error(f"Error counting modules: {e}")
        st.error(f"Error counting modules: {e}")
        return 0

def process_module_outline(df: pd.DataFrame, course_id: str) -> Dict[str, Any]:
    """Process the course outline and extract all video data."""
    try:
        logger.info(f"Starting course outline processing for course: {course_id}")
        # Find number of modules
        num_modules = find_module_count(df)
        logger.info(f"Found {num_modules} modules in the course outline")
        
        if num_modules == 0:
            logger.warning("No modules found in the Excel file")
            return {
                "status": "error",
                "message": "No modules found in the Excel file"
            }
        
        all_video_data = []
        s3_files = []
        
        # Process each module and create individual video files
        for module_num in range(1, num_modules + 1):
            module_df = slice_for_module(df, module_num)
            video_data = extract_video_data(module_df)
            
            if video_data:
                logger.info(f"Processing module {module_num} with {len(video_data)} videos")
                # Create individual DOCX file for each video
                for idx, (title, voiceover) in enumerate(video_data, 1):
                    # Track video data for summary
                    all_video_data.append({
                        "module": module_num,
                        "video": idx,
                        "title": title,
                        "voiceover": voiceover
                    })
                    
                    # Create TXT file for this individual video
                    txt_buffer = create_video_txt(module_num, idx, title, voiceover, course_id)
                    # Clean filename by removing special characters
                    clean_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    clean_title = clean_title.replace(' ', '_')[:50]  # Limit length and replace spaces
                    s3_key = f"video_descriptions/{course_id}/module_{module_num}_video_{idx}_{clean_title}.txt"
                    
                    if upload_to_s3(txt_buffer.getvalue(), s3_key):
                        s3_files.append({
                            "module": module_num,
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
            "total_modules": num_modules,
            "total_videos": len(all_video_data),
            "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "s3_files": s3_files
        }
        
        logger.info(f"Course processing completed successfully: {len(all_video_data)} videos processed, {len(s3_files)} files uploaded")
        return {
            "status": "success",
            "message": f"Successfully extracted {len(all_video_data)} video descriptions and created individual TXT files",
            "total_modules": num_modules,
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


