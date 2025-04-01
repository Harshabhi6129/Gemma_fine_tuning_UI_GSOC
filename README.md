# Google Summer of Code 2025 Proposal

## Personal Information
- **Name**: Harsha Abhinav Kusampudi 
- **Email**: harshakusampudi@gmail.com
- **GitHub Profile**: Harshabhi6129 
- **University**: University at Buffalo  
- **Degree Program**: MS in Computer Science  
  

---

## Title:  
**Gemma & Beyond: A No-Code Fine-tuning and Machine Learning Training Interface for Everyone**

---

## Synopsis

This project aims to build an intuitive, extensible, and open-source Streamlit/Gradio-based interface for training and fine-tuning machine learning models — from large pre-trained language models like Gemma to classical ML/DL models like Random Forests, Logistic Regression, or CNNs. Users can upload datasets, describe their problem using prompts, select the task type (classification, regression, or text generation), and configure hyperparameters and evaluation strategies — all within a user-friendly UI. It will offer training progress visualization, evaluation metrics (e.g., F1, accuracy, loss curves), and integration with Google Cloud (GCS + Vertex AI) for scalable, production-grade training.

This platform will democratize model experimentation, making powerful AI tooling accessible to students, educators, and non-coders.

---

## Benefits to the Community

- **Educational Access**: Students can learn ML/DL and LLM fine-tuning by experimenting with real datasets and algorithms — no coding needed.
- **Open-Source Ecosystem**: Builds tools around Gemma and other public models, strengthening the open-source ML community.
- **Scalable Training via Google Cloud**: By integrating with GCS and Vertex AI, users can transition from local demos to real cloud-based training.
- **No-Code Experimentation**: Greatly reduces the barrier to entry for experimenting with fine-tuning and ML pipelines.

---

## Deliverables

### 📦 Core Features
- Upload tabular or textual datasets
- Select task: classification, regression, or text generation
- Automatic parsing and validation of columns
- Auto-suggest ML/DL algorithms based on the dataset
- Configure training: hyperparameters, loss, metrics, and cross-validation
- Train models using `scikit-learn`, `PyTorch`, `TensorFlow`, or HuggingFace
- Real-time visualization of training progress
- Final report of evaluation metrics and plots
- Export trained model + logs + metrics

### 🌐 Advanced Features
- Support fine-tuning Gemma models using HuggingFace + Vertex AI
- Google Cloud Storage (GCS) integration for dataset/model upload
- Vertex AI backend for scalable training of larger models

---

## Timeline

| Period | Deliverables |
|--------|--------------|
| **Community Bonding** | Interact with mentors, finalize technical stack, gather feedback on architecture. Deploy demo on HuggingFace Spaces / Streamlit Cloud. |
| **Week 1–2** | Modularize Streamlit app: UI components for data upload, task selection, prompt interface, and evaluation config |
| **Week 3–4** | Add classic ML model training (Logistic Regression, Random Forest, XGBoost, etc.) with sklearn and visual metric tracking |
| **Week 5–6** | Integrate deep learning models (CNNs, LSTMs, etc.) using TensorFlow/PyTorch |
| **Week 7–8** | Add Gemma fine-tuning via HuggingFace `Trainer` API (local) with real-time evaluation and model download |
| **Midterm Evaluation** | Complete working pipeline for local training: classification/regression + Gemma fine-tuning. Deployed demo. |
| **Week 9–10** | Add cloud-based GCS upload, checkpoint storage, and Vertex AI backend for scalable training |
| **Week 11–12** | Add detailed evaluation reports and performance graphs (loss, accuracy, confusion matrix, etc.) |
| **Final 2 weeks** | Code polish, testing, error handling, documentation. Write tutorials and notebooks. Publish on GitHub + Google Cloud. |

---

## Technical Details

- **Frontend**: Streamlit (core), Gradio (for possible deployment UI)
- **ML Training**: `scikit-learn`, `TensorFlow`, `PyTorch`, `transformers`
- **Text Fine-tuning**: HuggingFace Transformers + Gemma + `Trainer` API
- **Data Handling**: Pandas, NumPy, Tokenizers
- **Visualization**: Matplotlib, Seaborn, Streamlit charts
- **Cloud**: Google Cloud Storage for datasets/models, Vertex AI for cloud-based model training
- **Deployment**: Streamlit Cloud, HuggingFace Spaces, optional Docker

---

## Future Scope

- Add AutoML tab: suggest best model for dataset
- Integrate LLM Agents to auto-configure tasks and parameters
- Add experiment tracking (e.g., with Weights & Biases or MLflow)
- Support multiple users and team collaboration
- Launch as a full web service for public access

---

## GitHub Repo (Demo Link)

🔗 [GitHub Repository – Gemma Fine-Tuning UI (Demo)](https://github.com/yourusername/gemma-finetune-ui-demo) *(Upload your files here and update this link)*

---

## Why Me?

- MS in CS, focused on AI and ML systems
- Strong foundation in Python, ML/DL, Streamlit, and HuggingFace
- Passionate about education and accessibility in AI
- Experienced in building no-code tools and visually intuitive apps
- Already built and deployed a working demo version of this project, which will serve as the foundation

---

## License

This project will be released under the Apache 2.0 license to maximize compatibility with other open-source AI projects, including Google's Gemma.

