# analyze/tasks.py
from celery import shared_task
import io
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from easyocr import Reader
import logging
from openai import OpenAI, OpenAIError
from django.conf import settings

logger = logging.getLogger(__name__)
reader = Reader(['en'], gpu=False)

def extract_text_from_file_bytes(file_bytes, content_type):
    if content_type == 'application/pdf':
        images = convert_from_bytes(file_bytes, dpi=300)
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

@shared_task
def analyze_exam_task(file_bytes, content_type, model_type):
    try:
        text = extract_text_from_file_bytes(file_bytes, content_type)
        if not text.strip():
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
        prompt = config['prompt_template'].format(text=text)
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

    except OpenAIError as e:
        logger.error(f"OpenAI error: {str(e)}")
        return {"error": "AI service error"}
    except Exception as e:
        logger.error(f"Task error: {str(e)}")
        return {"error": "Task failed"}
