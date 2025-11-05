import os
import re
import json
import threading
import traceback
import pandas as pd
from io import BytesIO
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from botocore.config import Config
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
    slugify
)

from src.logging_config import get_logger
logger = get_logger(__name__)

class ParallelVSBProcessor:
    """Main class for processing course Excel files and generating video storyboards."""
    def __init__(self):
        self.claude_llm = ChatBedrock(
            model_id=os.getenv("CLAUDE_MODEL"),
            model_kwargs=dict(temperature=0.2),
            config=Config(read_timeout=2400)
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self._token_lock = threading.Lock()

        self.no_of_lesson_check_prompt = """Analyze the following markdown text and count how many lessons are present. Look for entries that follow the pattern \"Lesson X\" where X is a number (e.g., \"Lesson 1\", \"Lesson 2\", etc.). \n
Count only the unique lesson numbers and provide the total count in the following JSON format: {{\"no_of_lesson\": your_answer}}\n
Place your response within <output> tags.\n
markdown text: {markdown_text}\n"""

        self.vsb_prompt = """{prompt_text}\n        
VOICEOVER SCRIPT: {voiceover}\n
Return ONLY the storyboard table in a proper markdown format with no extra commentary.\n"""

    def _record_usage(self, usage: Dict[str, int]) -> None:
        """
        Thread-safe increment of input and output token counters.
        """
        with self._token_lock:
            self.input_tokens += usage.get("prompt_tokens", 0)
            self.output_tokens += usage.get("completion_tokens", 0)

    def _find_lesson_number(self, markdown_text: str) -> int:
        """
        Analyze markdown text and count how many lessons are present.
        """
        try:
            pattern = re.compile(r"<output>(.*?)</output>", re.DOTALL)
            prompt_template = ChatPromptTemplate.from_messages([("user", self.no_of_lesson_check_prompt)])
            chain = prompt_template | self.claude_llm
            ai_msg = chain.invoke({'markdown_text': markdown_text})
            self._record_usage(ai_msg.additional_kwargs.get("usage", {}))
            match = pattern.search(ai_msg.content)
            if match:
                res = json.loads(match.group(1).strip())
                return res.get("no_of_lesson", 0)
            else:
                logger.warning("No lesson count found in LLM response")
                return 0
        except Exception as e:
            logger.error(f"Error finding lesson number: {e}")
            return 0

    def create_vsb_from_llm(self, prompt_text: str, voiceover: str) -> str:
        """
        Generate a video storyboard using the Claude API.
        """
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
        prompt_text: str,
        output_dir: str
        ):
        """
        Generate storyboards for a single lesson (all its videos), sequentially.
        """
        try:
            lesson_df = slice_for_lesson(input_dataframe, lesson_num)
            video_data = extract_video_data(lesson_df)
            if not video_data:
                logger.warning(f"[Lesson {lesson_num}] no video data found; skipping.")
                return

            for idx, (title, voiceover) in enumerate(video_data, start=1):
                logger.info(f"▶️  L{lesson_num} | V{idx} | {title[:20]}")
                try:
                    sb_table = self.create_vsb_from_llm(prompt_text, voiceover)
                    html_text = convert_markdown_to_html(sb_table)

                    # Ensure output directory exists
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        logger.info(f"Created output directory: {output_dir}")

                    output_path = os.path.join(output_dir, f"lesson_{lesson_num}_video_{idx}.docx")
                    html_table_to_docx(html_text, output_path=output_path)
                    logger.info(f"L{lesson_num} | V{idx}] ✅ done")
                except Exception as e:
                    logger.error(f"[Lesson {lesson_num} | V{idx}] error: {e}")
        except Exception as e:
            logger.error(f"[Lesson {lesson_num}] failed: {e}\n{traceback.format_exc()}")

    def generate_vsb(
        self,
        input_dataframe: pd.DataFrame,
        prompt_file: str,
        output_dir: str
        ) -> Dict[str, Any]:
        """
        Parallelize across lessons; within each lesson, videos are processed sequentially.
        """
        logger.info("[generate_vsb] starting")
        # Load prompt
        prompt_text = load_prompt(prompt_file)

        # Determine number of lessons
        buffer = create_excel_buffer(input_dataframe)
        excel_content = convert_docx_to_markdown(buffer)
        buffer.close()
        num_lessons = self._find_lesson_number(excel_content)
        if num_lessons <= 0:
            logger.error("no lessons found; aborting")
            return {"status": "failure", "message": "no lessons found"}

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Use number of lessons as max_workers
        max_workers = num_lessons
        logger.info(f"Running with max_workers={max_workers} (one per lesson)")

        # Submit one task per lesson
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_lesson, lesson_num, input_dataframe, prompt_text, output_dir): lesson_num for lesson_num in range(1, num_lessons + 1)
            }
            for future in as_completed(futures):
                lesson = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[Lesson {lesson}] unhandled exception: {e}")

        logger.info("[generate_vsb] all lessons processed")

        return {"status": "success", "message": "all lessons processed"}
