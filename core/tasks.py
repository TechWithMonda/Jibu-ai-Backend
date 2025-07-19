# analyze/tasks.py
from celery import shared_task
import io
from celery.exceptions import SoftTimeLimitExceeded

import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from easyocr import Reader
import logging
from openai import OpenAI, OpenAIError
from django.conf import settings

logger = logging.getLogger(__name__)
reader = Reader(['en'], gpu=False)
@shared_task(bind=True, soft_time_limit=60, time_limit=90)
def extract_text_task(self, file_bytes, content_type):
    try:
        if content_type == 'application/pdf':
            images = convert_from_bytes(file_bytes, dpi=200)
            if len(images) > 5:
                images = images[:5]
            text = ""
            for idx, img in enumerate(images):
                np_img = np.array(img)
                result = reader.readtext(np_img)
                page_text = "\n".join([res[1] for res in result])
                text += f"Page {idx+1}:\n{page_text}\n"
            return text.strip()
        else:
            img = Image.open(io.BytesIO(file_bytes))
            img = img.resize((img.width // 2, img.height // 2))
            np_img = np.array(img)
            result = reader.readtext(np_img)
            return "\n".join([res[1] for res in result]).strip()

    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise self.retry(exc=e, countdown=5, max_retries=2)
@shared_task(bind=True, soft_time_limit=180, time_limit=210)
def analyze_text_with_openai_task(self, extracted_text, model_type):
    try:
        if not extracted_text.strip():
            return {"error": "No text extracted from document."}

        model_config = {
            'basic': {
                'prompt_template': "Provide concise answers to these exam questions:\n\n{text}",
                'model': "gpt-3.5-turbo",
                'max_tokens': 300
            },
            'standard': {
                'prompt_template': "Provide detailed step-by-step solutions to these exam questions:\n\n{text}",
                'model': "gpt-3.5-turbo",
                'max_tokens': 800
            },
            'advanced': {
                'prompt_template': """Provide comprehensive solutions with:
1. Multiple solution methods where applicable
2. Explanations of key concepts
3. References to relevant formulas/theorems
4. Practical applications

Questions:
{text}""",
                'model': "gpt-3.5-turbo",
                'max_tokens': 1500
            }
        }

        if model_type not in model_config:
            model_type = 'standard'

        config = model_config[model_type]
        prompt = config['prompt_template'].format(text=extracted_text)

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config['model'],
            messages=[
                {"role": "system", "content": "You are a helpful teaching assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config['max_tokens'],
            temperature=0.3,
            top_p=0.9
        )

        return {
            "result": response.choices[0].message.content,
            "model_used": model_type
        }

    except Exception as e:
        logger.error(f"OpenAI analysis failed: {str(e)}")
        raise self.retry(exc=e, countdown=5, max_retries=2)




        # tasks.py

@shared_task(bind=True, soft_time_limit=60, time_limit=90)
def ocr_extract_task(self, job_id, file_bytes, content_type):
    from .models import ExamAnalysisJob
    import pytesseract
    import cv2
    import numpy as np
    from pdf2image import convert_from_bytes
    from PIL import Image

    try:
        job = ExamAnalysisJob.objects.get(id=job_id)
        job.status = 'ocr'
        job.save()

        # OCR processing
        text = ''
        if content_type == 'application/pdf':
            images = convert_from_bytes(file_bytes, dpi=200)
            if len(images) > 5:
                images = images[:5]
            for img in images:
                gray = np.array(img.convert("L"))
                text += pytesseract.image_to_string(gray)
        else:
            img = Image.open(io.BytesIO(file_bytes)).convert("L")
            gray = np.array(img)
            text = pytesseract.image_to_string(gray)

        # Pass text to next stage
        analyze_exam_task.delay(job_id, text)
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        job.save()
        
@shared_task(bind=True, soft_time_limit=60, time_limit=90)
def analyze_exam_task(self, job_id, extracted_text):
    from .models import ExamAnalysisJob
    from openai import OpenAI
    from django.conf import settings

    try:
        job = ExamAnalysisJob.objects.get(id=job_id)
        job.status = 'analyzing'
        job.save()

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = f"Provide detailed step-by-step answers:\n\n{extracted_text}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        job.result = response.choices[0].message.content
        job.status = 'done'
        job.save()
    except Exception as e:
        job.status = 'error'
        job.error = str(e)
        job.save()
