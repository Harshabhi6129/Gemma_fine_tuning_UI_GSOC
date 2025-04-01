# Google Summer of Code 2025 Proposal

## Personal Information
- **Name**: Harsha Abhinav Kusampudi
- **Email**: harshakusampudi@gmail.com
- **GitHub Profile**: Harshabhi6129 
- **University**: University at Buffalo  
- **Degree Program**: MS in Computer Science  
 

---

## Title:  
**Gemma Fine-Tuning Interface: Developer Tools and Public Usability for Open Models**
DEMO LINK: https://gemmafinetuninguigsoc-nuu5zce6ssmbwm3jt4fl6h.streamlit.app/

---

## Synopsis

This project proposes the development of a full-featured, professional-grade UI for fine-tuning and evaluating open models from the Gemma family. The interface will be built using Streamlit and HuggingFace, supporting dataset upload, hyperparameter configuration, training visualization, and export functionality. The application will also integrate with Google Cloud (GCS for storage and Vertex AI for scalable training) as outlined in the Google DeepMind GSoC project brief.

In the second phase of the project, we propose expanding this platform into a powerful public-use application for training and comparing machine learning and deep learning models. With no-code access to tools for classification, regression, text modeling, and more, this system will empower students and non-programmers to learn and explore ML by example.

---

## Part I: Implementation of the Gemma Fine-Tuning UI (Core Scope)

The primary objective is to create a user interface that supports fine-tuning Google’s Gemma models. This includes:

### 1. Dataset Upload and Validation
- Users can upload datasets in CSV, JSONL, or TXT formats.
- App validates file structure and schema (requires `input` and `output` columns).
- Sample previews and automatic data checks are provided.

### 2. Hyperparameter Configuration
- Interface to select learning rate, batch size, number of epochs.
- Tooltips for each parameter help educate the user.
- Smart defaults and input validation.

### 3. Prompt-based Testing
- Before training, the user can test the base model’s performance using custom prompts.
- HuggingFace pipelines (`text-generation`) provide this functionality.

### 4. Simulated Training Phase
- In the demo phase, the app simulates training progress with dummy metrics and visualizations.
- Real-time feedback loop through Streamlit progress bars and Matplotlib loss plots.

### 5. Actual Fine-tuning (Planned for Full Phase)
- Using HuggingFace’s `Trainer` API and LoRA-based fine-tuning for Gemma.
- Tokenization, training loop, checkpoint saving, and evaluation using built-in trainer support.

### 6. Model Export and Inference
- Save fine-tuned model locally or to cloud storage (in formats like PyTorch `.bin`, TensorFlow SavedModel, or GGUF).
- Provide endpoint or download button for inference.

### 7. Google Cloud Integration
- Connect to Google Cloud Storage for uploading large datasets and saving checkpoints.
- Vertex AI backend integration for distributed training and TPU/GPU acceleration.

### Outcome
- A working demo with the full fine-tuning pipeline of Gemma models.
- Integrated cloud support for scale.
- Ready-to-use UI with documentation and tutorials.

---

## Part II: Generalizing into a Public ML/DL Training Platform

Building upon the core foundation, the project will evolve into a general-purpose training interface that:

### 🧠 Allows Users To:
- Upload their dataset.
- Select a problem type: classification, regression, text generation.
- Choose ML/DL algorithms: Random Forests, SVM, XGBoost, CNNs, RNNs, LSTMs, or Transformers.
- Select a target variable and relevant features.
- Specify training config (cross-validation, metrics, loss functions).

### 📊 Offers:
- Real-time training logs and metric updates.
- Visualization of confusion matrices, ROC curves, F1-score graphs, and loss curves.
- Downloadable reports of metrics and plots.
- Export trained model weights and configuration.

### 🧰 Technologies Used
- `scikit-learn`, `XGBoost`, `TensorFlow`, `PyTorch`, `transformers`
- `matplotlib`, `seaborn`, and Streamlit charts for visuals
- GCS for storage, Vertex AI for cloud training jobs

### 🎓 Designed For:
- Students and educators who want to run experiments easily.
- Beginners learning ML through play.
- Researchers quickly validating hypotheses.

---

## Timeline

| Period | Deliverables |
|--------|--------------|
| **Community Bonding** | Finalize architecture, cloud integration plan, connect with mentors, deploy prototype on Streamlit Cloud |
| **Week 1–2** | Implement data upload + validation module; test dataset schemas and formats |
| **Week 3–4** | Add hyperparameter configuration UI + prompt-based text generation module |
| **Week 5–6** | Simulated training loop with loss visualization and tokenized sample previews |
| **Week 7–8** | Integrate HuggingFace `Trainer` for real fine-tuning of Gemma models |
| **Midterm Evaluation** | Complete Gemma fine-tuning pipeline + testing suite + documentation |
| **Week 9–10** | Add ML/DL generalization: support scikit-learn models, classification/regression pipelines |
| **Week 11–12** | Implement metric visualizations and model comparison dashboard |
| **Final 2 Weeks** | Cloud deployment (Vertex AI + GCS), model export, user docs, tutorial notebooks, and final polish |

---

## Technical Details

- **Frontend**: Streamlit (core), optional Gradio for hosted demos
- **Model APIs**: `transformers`, `scikit-learn`, `xgboost`, `tensorflow`, `torch`
- **Cloud**: Google Cloud Storage, Vertex AI Pipelines or Training APIs
- **Evaluation**: Accuracy, F1-score, Precision, Recall, ROC, Log Loss
- **Visuals**: Matplotlib, Seaborn, Streamlit charts

---

## Future Scope

- AutoML-style recommendation engine for model type
- Natural language agent to auto-configure pipelines
- Multi-user support with login/session history
- Public demo deployment with HuggingFace Spaces or Firebase hosting
- Reusable components for other open model projects

---

## GitHub Repo (Demo Link)

🔗 [GitHub Repository – Gemma Fine-Tuning UI (Demo)](https://github.com/yourusername/gemma-finetune-ui-demo) *(replace with actual link)*

---

## Why Me?

- Pursuing MS in CS, focused on AI systems
- Strong in Python, ML/DL pipelines, visualization, and frontend tools like Streamlit
- Passionate about making machine learning accessible and visual
- Already implemented and demoed a partial version of the project
- Experienced in both open-source contribution and technical writing

---

## License

Apache 2.0 — to align with Google's open-source model policy and maximize public usability

