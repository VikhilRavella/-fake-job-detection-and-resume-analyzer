This project was developed as part of the Infosys Springboard 6.0 Internship Program, where our team built a Fake Job Detection Platform to help users identify fraudulent job postings and highlight legitimate opportunities.

The platform combines:

NLP (Natural Language Processing)

PyTorch Deep Learning Models

GPU-accelerated training

Django Web Application

User Feedback Loop for model improvement

🧑‍💼 My Role – Team Lead (Leadership + Technical)

As the Team Lead, I was responsible for:

✔ 1. Project Coordination & Git Management

Managed entire version control using Git & GitHub.

Created individual branches for each member.

Performed code reviews and merged final updates.

Shared daily updates with the project mentor.

✔ 2. Core NLP Model Development (PyTorch)

Built multiple NLP models using Python + PyTorch.

Trained models on my local NVIDIA GPU for faster experimentation.

Performed data preprocessing, tokenization, evaluation, and tuning.

Compared two final models and selected the best performer.

✔ 3. Web Integration using Django

Integrated the final model into a Django web app.

Built a user feedback system to collect new data for future retraining.

Ensured end-to-end flow: Input → Prediction → Feedback → Storage.

⚙️ Tech Stack Component Technology Language Python Model Framework PyTorch (GPU Accelerated) NLP Tools SpaCy, TorchText, Transformers Deployment Django Version Control Git / GitHub Hardware NVIDIA GPU 🚀 GPU Setup (PyTorch)

Follow these commands to check GPU availability and install CUDA-enabled PyTorch.

Install GPU-version PyTorch pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Verify GPU

Run:

import torch print(torch.cuda.is_available()) print(torch.cuda.get_device_name())

Expected output if GPU is detected:

True NVIDIA GeForce ...

🛠️ How to Run the Django Project

Clone the Repository git clone <your_repo_link> cd <project_folder>

Create a Virtual Environment python -m venv env source env/bin/activate # Linux/Mac env\Scripts\activate # Windows

Install Dependencies pip install -r requirements.txt

Run Database Migrations python manage.py makemigrations python manage.py migrate

Run the Development Server python manage.py runserver

The app is now live at: ➡ http://127.0.0.1:8000/

🧠 Model Workflow Job Description ↓ Text Preprocessing → Tokenization ↓ PyTorch Neural Network (trained on GPU) ↓ Prediction (Fake / Legitimate) ↓ User Feedback Collection ↓ Stored for Model Retraining

📈 Results

Compared two NLP models using accuracy, precision, recall.

Selected the best-performing model for deployment.

Integrated the model into Django with real-time predictions.

Implemented a feedback loop to gather new training data.

📂 Project Proof & Certificate

🔗 Project Drive Link (Proof): https://www.linkedin.com/posts/vikhil-ravella-a098b72b2_infosys-springboot-intern-activity-7401307135629762560-D1rS?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEs9b-kB6hXAvj4Ir8IMnCPdEo6-eW8B4l8 🔗 Infosys Springboard Internship Certificate: https://www.linkedin.com/posts/vikhil-ravella-a098b72b2_infosysspringboard-teamlead-fakejobdetection-activity-7401304500554342400-yGtq?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEs9b-kB6hXAvj4Ir8IMnCPdEo6-eW8B4l8

🙌 Team & Credits

This project was completed under the Infosys Springboard 6.0 Internship Program with valuable support from teammates and the project mentor.
>>>>>>> d5cca9eee60eb093f9147153f0d9ccb8e9bcc4f3
