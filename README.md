# 🏥 Multi-Organ Disease Prediction & Medical RAG Assistant

An AI-powered medical assistance application that combines **Deep Learning-based organ and disease classification** with a **Retrieval-Augmented Generation (RAG) chatbot** to provide disease-related information.

The project follows a **two-stage classification architecture** to identify the affected organ and subsequently predict the specific disease associated with that organ. Additionally, it integrates a RAG-based chatbot that answers basic disease-related questions using a collection of medical PDF documents.

---

## 🚀 Key Features

### 1. Two-Stage Organ and Disease Prediction

The application uses a hierarchical classification architecture consisting of two stages:

- **Stage 1 – Organ Detection:** Identifies the organ associated with the uploaded medical image.
- **Stage 2 – Disease Classification:** Based on the detected organ, the corresponding specialized classification model predicts the particular disease.

This architecture separates organ identification from disease classification, making the prediction pipeline modular and easier to extend.

### 2. Medical RAG Assistant

A separate RAG-based application is integrated into the Streamlit interface to answer basic disease-related questions.

**RAG Pipeline:**

1. Medical PDF documents are loaded from the `pdf` directory.
2. Documents are processed and divided into smaller text chunks.
3. Text chunks are converted into vector embeddings.
4. Embeddings are stored in a FAISS vector database.
5. Relevant information is retrieved based on the user's query.
6. Retrieved context is passed to the language model to generate a response.

The chatbot uses the provided medical documents as its knowledge source.

### 3. Interactive Streamlit Interface

The application provides two main sections:

- **Disease Prediction:** Upload a medical image and receive the predicted organ and corresponding disease, along with prediction confidence.
- **Medical Q&A:** Ask basic questions about diseases and receive context-based answers through the RAG pipeline.

### 4. Model Training and Inference

The repository includes separate Python scripts for model training and inference, making it easier to understand, reproduce, and maintain the machine learning workflow.

### 5. Sample Images

A collection of sample test images is included in the repository to test the prediction pipeline and explore the application's functionality.

---

## 🧠 Model Architecture

The system follows a hierarchical two-stage deep learning architecture.

```text
              Input Medical Image
                      |
                      ▼
             Stage 1: Organ Classifier
                      |
                      ▼
              Predicted Organ
                      |
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        Brain       Lung        Breast
          |           |           |
          ▼           ▼           ▼
      Brain Model  Lung Model  Breast Model
          |           |           |
          ▼           ▼           ▼
      Predicted   Predicted   Predicted
       Disease     Disease     Disease
          |           |           |
          └───────────┼───────────┘
                      |
                      ▼
              Final Prediction
```

**Workflow:**

- The uploaded image is first passed through the organ classification model.
- The predicted organ determines which specialized disease classification model will be used.
- The selected model performs disease-level classification.
- The application displays the final prediction and its confidence score.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | NumPy, Pillow |
| Frontend | Streamlit |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers |
| LLM Framework | LangChain |
| Document Processing | PyPDF |
| Version Control | Git, GitHub |

---

## 📂 Project Structure

```text
Multi-Organ-Disease-Prediction/
│
├── app.py                         # Main Streamlit application
│
├── train.py                    # Model training scripts
├── inference.py                   # Model inference scripts
│
├── new_model/
│   ├── Organ_classifier_256.keras
│   ├── Brain_classifier_256.keras
│   ├── Lung_classifier_256.keras
│   ├── Breast_Cancer_classifier_256.keras
|   ├── skin_Cancer_classifier_256.keras
│   └── Eye_classifier_256.keras
│
├── pdf/
│   ├── brain_disease.pdf
│   ├── lung_disease.pdf
│   ├── breast_disease.pdf
│   └── other_disease_documents.pdf
│
├── sample_images/
│   ├── brain/
│   ├── lung/
│   ├── breast/
│   └── other/
│
├── requirements.txt               # Project dependencies
│
└── README.md
```

*Note: Adjust filenames and directory names according to the actual repository structure.*

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

During Use of RAG for queries apply your api key and required model in app.py file.
Also use embedding model named as all-miniLM-L6-v2 

Never upload API keys or other sensitive credentials to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## 🖥️ Application Workflow

### Disease Prediction

1. Open the Disease Prediction section.
2. Upload a medical image.
3. The first-stage classifier identifies the organ.
4. The corresponding second-stage classifier predicts the specific disease.
5. View the predicted organ, disease, and confidence score.

### Medical RAG Assistant

1. Open the Medical Q&A section.
2. Enter a disease-related question.
3. The system retrieves relevant information from the medical PDF knowledge base.
4. The LLM generates an answer using the retrieved context.

---

## 🧪 Testing

The `sample_images` directory contains test images that can be uploaded directly through the Streamlit interface.

These images help evaluate the complete two-stage prediction pipeline.

You can also use the training and inference scripts independently to explore model development and prediction.

---

## 🔮 Future Improvements

- Improve classification performance on unseen real-world medical images.
- Expand the number of supported organs and disease categories.
- Introduce Grad-CAM for visual model interpretability.
- Improve RAG retrieval using hybrid search and reranking.
- Add medical document citations and source references to chatbot responses.
- Deploy the application using Docker and cloud infrastructure.

---

## ⚠️ Disclaimer
This project is developed for **educational and research purposes only**.
The disease predictions and chatbot responses are not intended to replace professional medical diagnosis, clinical evaluation, or treatment recommendations.
Always consult qualified healthcare professionals for medical advice.

---

## 👨‍💻 Author
**Bharat Chauhan**
⭐ If you find this project interesting, consider giving the repository a star!
