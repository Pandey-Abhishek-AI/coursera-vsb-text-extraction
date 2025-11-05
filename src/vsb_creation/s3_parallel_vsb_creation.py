import os
import re
import json
import boto3
import threading
import traceback
import pandas as pd
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

from src.utils import (
    load_prompt,
    slice_for_lesson,
    extract_video_data,
    convert_markdown_to_html,
    html_table_to_docx,
    create_excel_buffer,
    convert_docx_to_markdown,
    create_docx_in_memory,
    slugify
)

from src.logging_config import get_logger
logger = get_logger(__name__)


class S3ParallelVSBProcessor:
    """S3-enabled parallel VSB processor that uploads generated files to S3."""
    
    def __init__(self, folder_prefix: str = None):
        """
        Initialize the S3 parallel VSB processor.
        
        Args:
            folder_prefix (str, optional): Custom folder prefix for S3 storage.
                                         If None, uses timestamp-based folder.
        """
        self.claude_llm = ChatBedrock(
            model_id=os.getenv("CLAUDE_MODEL"),
            model_kwargs=dict(temperature=0.2),
            config=Config(read_timeout=2400)
        )
        
        # S3 configuration
        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME")
        if not self.s3_bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is required")
        self.aws_region = os.getenv("AWS_REGION", "us-east-2")
        self.s3_client = boto3.client('s3')
        
        # Create folder structure
        if folder_prefix:
            self.s3_folder = folder_prefix
        else:
            raise ValueError("folder_prefix is required")
            
        # Token tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self._token_lock = threading.Lock()
        
        # Upload tracking
        self.uploaded_files = []
        self.failed_uploads = []
        self.uploaded_files_link = []
        self._upload_lock = threading.Lock()

        # Prompts
        self.no_of_lesson_check_prompt = """Analyze the following markdown text and count how many lessons are present. Look for entries that follow the pattern "Lesson X" where X is a number (e.g., "Lesson 1", "Lesson 2", etc.). 

Count only the unique lesson numbers and provide the total count in the following JSON format: {{"no_of_lesson": your_answer}}

Place your response within <output> tags.

markdown text: {markdown_text}
"""

        self.vsb_prompt = """{prompt_text}
        
VOICEOVER SCRIPT: {voiceover}

Return ONLY the storyboard table in a proper markdown format with no extra commentary.
"""

    def _record_usage(self, usage: Dict[str, int]) -> None:
        """Thread-safe increment of input and output token counters."""
        with self._token_lock:
            self.input_tokens += usage.get("prompt_tokens", 0)
            self.output_tokens += usage.get("completion_tokens", 0)

    def _record_upload(self, file_info: Dict[str, Any], success: bool) -> None:
        """Thread-safe tracking of upload results."""
        with self._upload_lock:
            if success:
                self.uploaded_files.append(file_info)
                self.uploaded_files_link.append(file_info["s3_url"])
            else:
                self.failed_uploads.append(file_info)

    def _upload_to_s3(self, file_content: bytes, s3_key: str, content_type: str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document") -> bool:
        """
        Upload file content to S3.
        
        Args:
            file_content (bytes): File content to upload
            s3_key (str): S3 key (path) for the file
            content_type (str): MIME type of the file
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type
            )
            return True
            
        except ClientError as e:
            logger.error(f"S3 upload failed for {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading {s3_key}: {e}")
            return False

    def _find_lesson_number(self, markdown_text: str) -> int:
        """Analyze markdown text and count how many lessons are present."""
        try:
            pattern = re.compile(r"<output>(.*?)</output>", re.DOTALL)
            prompt_template = ChatPromptTemplate.from_messages([("user", self.no_of_lesson_check_prompt)])
            chain = prompt_template | self.claude_llm
            
            ai_msg = chain.invoke({'markdown_text': markdown_text})
            self._record_usage(ai_msg.additional_kwargs.get("usage", {}))
            match = pattern.search(ai_msg.content)
            
            if match:
                res = json.loads(match.group(1).strip())
                count = res.get("no_of_lesson", 0)
                return count
            else:
                logger.warning("No lesson count found in LLM response")
                return 0
                
        except Exception as e:
            logger.error(f"Error finding lesson number: {e}")
            return 0

    def create_vsb_from_llm(self, prompt_text: str, voiceover: str) -> str:
        """Generate a video storyboard using the Claude API."""
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert instructional designer. Return ONLY the storyboard table in a proper markdown format with no extra commentary."),
                ("user", self.vsb_prompt)
            ])
            
            chain = prompt_template | self.claude_llm
            ai_msg = chain.invoke({
                "prompt_text": prompt_text,
                "voiceover": voiceover
            })
            self._record_usage(ai_msg.additional_kwargs.get("usage", {}))

            return ai_msg.content
            
        except Exception as e:
            logger.error(f"Error generating storyboard from LLM: {e}")
            raise

    def _process_lesson(
        self,
        lesson_num: int,
        input_dataframe: pd.DataFrame,
        prompt_text: str
    ) -> Dict[str, Any]:
        """
        Process a single lesson and upload storyboards to S3.
        
        Args:
            lesson_num (int): The lesson number to process
            input_dataframe (pd.DataFrame): The input dataframe containing the course outline
            prompt_text (str): The prompt text for LLM
            
        Returns:
            Dict[str, Any]: Processing results for this lesson
        """
        logger.info(f"Processing lesson - {lesson_num}")
        
        try:
            lesson_df = slice_for_lesson(input_dataframe, lesson_num)
            video_data = extract_video_data(lesson_df)

            if not video_data:
                logger.warning(f"No video data found for lesson {lesson_num}")
                return {
                    "lesson_num": lesson_num,
                    "status": "no_videos",
                    "processed_videos": 0,
                    "total_videos": 0,
                    "uploaded_files": [],
                    "failed_uploads": []
                }
            
            processed_videos = 0
            total_videos = len(video_data)
            lesson_uploads = []
            lesson_failures = []
            
            for idx, (title, voiceover) in enumerate(video_data, start=1):
                logger.info(f"▶️  L{lesson_num} V{idx} | {voiceover[:30]}")
                try:
                    # Generate storyboard
                    sb_table = self.create_vsb_from_llm(prompt_text, voiceover)
                    html_text = convert_markdown_to_html(sb_table)
                    
                    # Create DOCX in memory
                    docx_buffer = create_docx_in_memory(html_text)
                    
                    # Create S3 key
                    filename = f"lesson_{lesson_num}_video_{idx}.docx"
                    s3_key = f"{self.s3_folder}/{filename}"
                    
                    # Upload to S3
                    upload_success = self._upload_to_s3(docx_buffer.getvalue(), s3_key)
                    
                    file_info = {
                        "video_idx": idx,
                        "s3_url": f"https://{self.s3_bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
                    }
                    
                    if upload_success:
                        lesson_uploads.append(file_info)
                        processed_videos += 1
                        logger.info(f"✅ Successfully uploaded L{lesson_num} V{idx} to S3: {file_info['s3_url']}")
                    else:
                        lesson_failures.append({**file_info, "error": "S3 upload failed"})
                        logger.error(f"❌ Failed to upload L{lesson_num} V{idx} to S3")
                    
                    # Clean up buffer
                    docx_buffer.close()

                except Exception as e:
                    logger.error(f"Error processing L{lesson_num} V{idx}: {e}")
                    lesson_failures.append({
                        "video_idx": idx,
                        "error": str(e)
                    })
                    continue
            
            # Record uploads for this lesson
            for upload in lesson_uploads:
                self._record_upload(upload, True)
            for failure in lesson_failures:
                self._record_upload(failure, False)
            
            return {
                "lesson_num": lesson_num,
                "status": "completed",
                "processed_videos": processed_videos,
                "total_videos": total_videos,
                "uploaded_files": lesson_uploads,
                "failed_uploads": lesson_failures
            }
            
        except Exception as e:
            logger.error(f"Error processing lesson {lesson_num}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                "lesson_num": lesson_num,
                "status": "failure",
                "error": str(e),
                "processed_videos": 0,
                "total_videos": 0,
                "uploaded_files": [],
                "failed_uploads": []
            }

    def generate_vsb(
        self,
        input_dataframe: pd.DataFrame,
        prompt_content: str = None,
        prompt_file: str = None
        ) -> Dict[str, Any]:
        """
        Generate video storyboards and upload to S3 with parallel lesson processing.

        Args:
            input_dataframe (pd.DataFrame): The input dataframe containing the course outline
            prompt_content (str, optional): The prompt content as a string
            prompt_file (str, optional): Path to the prompt file (for backward compatibility)

        Returns:
            Dict[str, Any]: Processing results with S3 folder URL
        """
        logger.info("Starting S3 parallel VSB generation")
        
        try:
            # Validate that either prompt_content or prompt_file is provided
            if not prompt_content and not prompt_file:
                logger.error("[generate_vsb] Either 'prompt_content' or 'prompt_file' must be provided")
                return {
                    "status": "failure",
                    "message": "Either 'prompt_content' or 'prompt_file' must be provided",
                    "folder_url": None
                }
            
            # Validate prompt_content if provided
            if prompt_content and (not isinstance(prompt_content, str) or not prompt_content.strip()):
                logger.error("[generate_vsb] 'prompt_content' must be a non-empty string")
                return {
                    "status": "failure",
                    "message": "prompt_content must be a non-empty string",
                    "folder_url": None
                }
            
            # Validate prompt_file if provided (for backward compatibility)
            if prompt_file and (not isinstance(prompt_file, str) or not prompt_file.strip()):
                logger.error("[generate_vsb] 'prompt_file' must be a non-empty string")
                return {
                    "status": "failure",
                    "message": "prompt_file must be a non-empty string",
                    "folder_url": None
                }
            # Validate S3 access
            try:
                self.s3_client.head_bucket(Bucket=self.s3_bucket_name)
            except ClientError as e:
                logger.error(f"S3 bucket access failed: {e}")
                return {
                    "status": "failure",
                    "message": f"S3 bucket access failed: {e}",
                    "folder_url": None
                }
            except NoCredentialsError:
                logger.error("AWS credentials not found")
                return {
                    "status": "failure",
                    "message": "AWS credentials not found",
                    "folder_url": None
                }
            
            # Load prompt text
            if prompt_content:
                prompt_text = prompt_content
                logger.info("Using provided prompt content")
            else:
                prompt_text = load_prompt(prompt_file)
                if not prompt_text:
                    logger.error("Failed to load prompt text from file")
                    return {
                        "status": "failure",
                        "message": "Failed to load prompt text from file",
                        "folder_url": None
                    }
                logger.info(f"Loaded prompt from file: {prompt_file}")
            
            # Create Excel buffer and convert to markdown for lesson counting
            buffer = create_excel_buffer(input_dataframe)

            # Check if the buffer is empty before proceeding
            buffer.seek(0, os.SEEK_END)
            buffer_size = buffer.tell()
            if buffer_size == 0:
                logger.error("Excel buffer is empty")
                buffer.close()
                del buffer
                return {
                    "status": "failure",
                    "message": "Excel buffer is empty",
                    "folder_url": None
                }
            buffer.seek(0)

            excel_content = convert_docx_to_markdown(buffer)
            buffer.close()
            del buffer

            if not excel_content:
                logger.error("Failed to convert Excel to markdown")
                return {
                    "status": "failure",
                    "message": "Failed to parse Excel content",
                    "folder_url": None
                }

            # Find number of lessons
            num_lessons = self._find_lesson_number(excel_content)

            if num_lessons == 0:
                logger.error("No lessons found in the Excel file")
                return {
                    "status": "failure",
                    "message": "No lessons found in the Excel file",
                    "folder_url": None
                }
            
            logger.info(f"Total number of lessons: {num_lessons}")
            
            # Process lessons in parallel
            lesson_results = []
            with ThreadPoolExecutor(max_workers=num_lessons) as executor:
                # Submit all lesson processing tasks
                future_to_lesson = {
                    executor.submit(self._process_lesson, lesson_num, input_dataframe, prompt_text): lesson_num for lesson_num in range(1, num_lessons + 1)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_lesson):
                    lesson_num = future_to_lesson[future]
                    try:
                        result = future.result()
                        lesson_results.append(result)
                        logger.info(f"Completed lesson {lesson_num}: {result['status']}")
                    except Exception as e:
                        logger.error(f"Error in lesson {lesson_num}: {e}")
                        lesson_results.append({
                            "lesson_num": lesson_num,
                            "status": "failure",
                            "error": str(e),
                            "processed_videos": 0,
                            "total_videos": 0,
                            "uploaded_files": [],
                            "failed_uploads": []
                        })
            
            # Generate comprehensive summary
            total_videos_processed = sum(r["processed_videos"] for r in lesson_results)
            total_videos = sum(r["total_videos"] for r in lesson_results)
            successful_lessons = len([r for r in lesson_results if r["status"] == "completed"])
            failed_lessons = len([r for r in lesson_results if r["status"] == "failure"])
            
            # Create S3 folder URL
            folder_url = f"s3://{self.s3_bucket_name}/{self.s3_folder}/"
            
            # Generate success message
            if successful_lessons == num_lessons and len(self.failed_uploads) == 0:
                status = "success"
                message = f"Successfully processed all {num_lessons} lessons and uploaded {total_videos_processed} videos to S3"
            elif successful_lessons > 0:
                status = "partial_success"
                message = f"Processed {successful_lessons}/{num_lessons} lessons successfully. {total_videos_processed}/{total_videos} videos uploaded to S3"
            else:
                status = "failure"
                message = f"Failed to process any lessons successfully"
            
            
            return {
                "status": status,
                "message": message,
                "folder_url": folder_url,
                "s3_folder": self.s3_folder,
                "s3_bucket": self.s3_bucket_name,
                "total_lessons": num_lessons,
                "successful_lessons": successful_lessons,
                "failed_lessons": failed_lessons,
                "total_videos": total_videos,
                "processed_videos": total_videos_processed,
                "uploaded_files": len(self.uploaded_files),
                "failed_uploads": len(self.failed_uploads),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "lesson_results": lesson_results
            }

        except Exception as e:
            logger.error(f"Error in S3 parallel processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "failure",
                "message": f"Processing failed: {str(e)}",
                "folder_url": None,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens
            }

    def get_s3_folder_url(self) -> str:
        """Get the S3 folder URL."""
        return f"s3://{self.s3_bucket_name}/{self.s3_folder}/"

    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of successfully uploaded files."""
        return self.uploaded_files.copy()

    def get_failed_uploads(self) -> List[Dict[str, Any]]:
        """Get list of failed uploads."""
        return self.failed_uploads.copy() 