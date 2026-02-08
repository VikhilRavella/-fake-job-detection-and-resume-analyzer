# import json
# import asyncio
# import os
# from dotenv import load_dotenv
# import fitz  # PyMuPDF
# import google.generativeai as genai
# from django.http import JsonResponse
# from django.shortcuts import render
# from PIL import Image
# import pytesseract

# Import the globally loaded model from your apps.py
# from .apps import MainAppConfig

# Import your prediction logic
# from .ml_logic import predict, is_valid_job_post

# Load environment variables (.env file)
# load_dotenv()


# def home_view(request):
#     """Renders the main homepage."""
#     return render(request, 'home.html')


# def about_view(request):
#     """Renders the about page."""
#     return render(request, 'about.html')


# def contact_view(request):
#     """Renders the contact page."""
#     return render(request, 'contact.html')


# # In main_app/views.py
# def manual_predict_view(request):
#     """
#     Handles the form submissions for manual text and image analysis
#     using your trained PyTorch model.
#     """
#     context = {}
#     if request.method == 'POST':
#         # Get the globally loaded model and artifacts from apps.py
#         model = MainAppConfig.MODEL
#         vocab = MainAppConfig.VOCAB
#         train_cols = MainAppConfig.TRAIN_COLS
#         device = MainAppConfig.DEVICE
#
#         # Get the data from the single form
#         uploaded_image = request.FILES.get('uploaded_file')
#         job_text = request.POST.get('job_description', '')
#
#         # Prioritize image if it exists
#         if uploaded_image:
#             try:
#                 image = Image.open(uploaded_image)
#                 extracted_text = pytesseract.image_to_string(image)
#
#                 if extracted_text and is_valid_job_post(extracted_text):
#                     prediction, probability = predict(
#                         extracted_text, model, vocab, train_cols, device
#                     )
#                     context['result'] = {
#                         'prediction': prediction,
#                         'probability': probability
#                     }
#                     context['extracted_text'] = extracted_text
#                 else:
#                     context['error'] = "Could not extract valid job text from the image."
#
#             except Exception as e:
#                 context['error'] = f"An error occurred during image processing: {e}"
#
#         # If no image, check for text
#         elif job_text:
#             if is_valid_job_post(job_text):
#                 prediction, probability = predict(
#                     job_text, model, vocab, train_cols, device
#                 )
#                 context['result'] = {
#                     'prediction': prediction,
#                     'probability': probability
#                 }
#             else:
#                 context['error'] = "This doesn't seem to be a job description."
#
#         # If both are empty
#         else:
#             context['error'] = (
#                 "Please either paste a job description "
#                 "or upload an image to analyze."
#             )
#
#     return render(request, 'predict.html', context)


# async def parse_pdf_resume(file_object):
#     """Asynchronously parses a PDF resume file."""
#     text = ""
#     try:
#         pdf_document = fitz.open(
#             stream=file_object.read(), filetype="pdf"
#         )
#         for page in pdf_document:
#             text += page.get_text()
#         pdf_document.close()
#     except Exception as e:
#         print(f"Error parsing PDF: {e}")
#         return None
#     return text


# async def search_for_jobs(query: str):
#     """Simulates searching for jobs."""
#     print(f"Simulating search for: {query}")
#     await asyncio.sleep(1)
#     return [
#         {
#             "title": "Senior Python Developer",
#             "company": "Innovatech",
#             "location": "Remote",
#             "description": "..."
#         },
#         {
#             "title": "Frontend Engineer (React)",
#             "company": "Creative Minds",
#             "location": "New York, NY",
#             "description": "..."
#         }
#     ]


# async def verify_job_with_real_model(job: dict):
#     """
#     Uses your trained PyTorch model to verify a job.
#     """
#     model = MainAppConfig.MODEL
#     vocab = MainAppConfig.VOCAB
#     train_cols = MainAppConfig.TRAIN_COLS
#     device = MainAppConfig.DEVICE
#
#     job_description = job.get("description", "")
#     if not job_description:
#         return False
#
#     prediction, _ = predict(
#         job_description, model, vocab, train_cols, device
#     )
#
#     return prediction == "Real Job"


# async def agent_view(request):
#     """
#     Handles the AI Agent chat using Gemini API and verification model.
#     """
#     if request.method == 'GET':
#         return render(request, 'agent.html')
#
#     if request.method == 'POST':
#         try:
#             resume_file = request.FILES.get('resume')
#             user_prompt = request.POST.get('prompt', '')
#
#             if not resume_file:
#                 return JsonResponse(
#                     {'error': 'Resume file is required.'}, status=400
#                 )
#
#             api_key = os.getenv("GEMINI_API_KEY")
#             if not api_key:
#                 return JsonResponse(
#                     {'error': 'GEMINI_API_KEY not found.'}, status=500
#                 )
#
#             genai.configure(api_key=api_key)
#
#             resume_text = await parse_pdf_resume(resume_file)
#             if resume_text is None:
#                 return JsonResponse(
#                     {'error': 'Could not parse the resume PDF.'}, status=400
#                 )
#
#             model = genai.GenerativeModel('gemini-1.5-flash')
#             prompt_template_query = "..."
#             prompt = prompt_template_query.format(
#                 resume_text=resume_text,
#                 user_prompt=user_prompt
#             )
#
#             response = await model.generate_content_async(prompt)
#             search_query = response.text.strip()
#
#             jobs_list = await search_for_jobs(search_query)
#             verified_jobs = []
#
#             for job in jobs_list:
#                 if await verify_job_with_real_model(job):
#                     verified_jobs.append(job)
#
#             if not verified_jobs:
#                 final_response = (
#                     "I couldn't find any verified job opportunities."
#                 )
#             else:
#                 prompt_template_response = "..."
#                 prompt = prompt_template_response.format(
#                     verified_jobs_list=json.dumps(verified_jobs)
#                 )
#                 response = await model.generate_content_async(prompt)
#                 final_response = response.text
#
#             return JsonResponse({'agent_response': final_response})
#
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
#
#     return JsonResponse({'error': 'Invalid request method.'}, status=405)


from django.shortcuts import render

def home(request):
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def contact(request):
    return render(request, "contact.html")

def predict(request):
    return render(request, "predict.html")

def agent(request):
    return render(request, "agent.html")
