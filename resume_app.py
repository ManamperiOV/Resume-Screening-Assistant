import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from gradio_pdf import PDF
from pdfminer.high_level import extract_text
import os

# === Load Model and Tokenizer ===
model_path = "./resume_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# === Label Mapping ===
label_map = {
    0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain",
    4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science", 7: "Database",
    8: "DevOps Engineer", 9: "DotNet Developer", 10: "ETL Developer",
    11: "Electrical Engineering", 12: "HR", 13: "Hadoop", 14: "Health and Fitness",
    15: "Java Developer", 16: "Mechanical Engineer", 17: "Network Security Engineer",
    18: "Operations Manager", 19: "PMO", 20: "Python Developer", 21: "SAP Developer",
    22: "Sales", 23: "Testing", 24: "Web Designing"
}

# === Helper: Truncate text by BERT token length ===
def truncate_text_to_bert_limit(text, tokenizer, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# === Prediction Function with Confidence Score ===
def predict_resume_text(text):
    if not text.strip():
        return "❌ No content found."
    # Truncate by tokens, not words
    text = truncate_text_to_bert_limit(text, tokenizer)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs).item()
        confidence = probs[0][pred_id].item()
    label = label_map.get(pred_id, "Unknown")
    return f"🎯 {label} ({confidence * 100:.2f}% confidence)"

# === Main Handler ===
def classify_resume(pdf_file, raw_text):
    if pdf_file:
        try:
            text = extract_text(pdf_file)
            if not text.strip():
                return "❌ Could not extract any text from the PDF.", os.path.basename(pdf_file)
            return predict_resume_text(text), os.path.basename(pdf_file)
        except Exception as e:
            return f"❌ Error reading PDF: {str(e)}", os.path.basename(pdf_file)
    elif raw_text:
        return predict_resume_text(raw_text), ""
    else:
        return "❗ Please provide resume content either by uploading a PDF or pasting text.", ""

# === Gradio UI ===
with gr.Blocks(title="Resume Screening Assistant") as app:
    gr.Markdown("## 🧠 Resume Screening Assistant")
    gr.Markdown("Upload your resume **PDF** or paste the text. The model predicts the most relevant job category.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = PDF(label="📄 Upload Resume (PDF Viewer)", height=400)
        with gr.Column(scale=1):
            textbox_input = gr.Textbox(label="✍️ Or Paste Resume Text", lines=18, placeholder="Paste plain text resume here...")

    with gr.Row():
        output_label = gr.Label(label="🎯 Predicted Job Category")
        file_name_display = gr.Textbox(label="📁 PDF File Name", interactive=False)

    with gr.Row():
        submit_btn = gr.Button("🔍 Predict")
        clear_btn = gr.Button("🧹 Clear")

    submit_btn.click(
        fn=classify_resume,
        inputs=[pdf_input, textbox_input],
        outputs=[output_label, file_name_display]
    )

    clear_btn.click(
        fn=lambda: (None, None, "", ""),
        inputs=None,
        outputs=[pdf_input, textbox_input, output_label, file_name_display]
    )

# === Launch App ===
if __name__ == "__main__":
    app.launch()