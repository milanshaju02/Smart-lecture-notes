import os
import re
import torch
from flask import Flask, render_template, request, send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from transformers import pipeline
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

# ==============================
# PATH SETUP
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
FONT_FOLDER = os.path.join(BASE_DIR, "fonts")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# REGISTER FONTS
# ==============================

mal_font = os.path.join(FONT_FOLDER, "NotoSansMalayalam-Regular.ttf")
hin_font = os.path.join(FONT_FOLDER, "NotoSansDevanagari-Regular.ttf")

if os.path.exists(mal_font):
    pdfmetrics.registerFont(TTFont("MalayalamFont", mal_font))

if os.path.exists(hin_font):
    pdfmetrics.registerFont(TTFont("HindiFont", hin_font))

# ==============================
# FLASK APP
# ==============================

app = Flask(__name__)
torch.set_num_threads(2)

# ==============================
# LAZY MODEL LOADING
# ==============================

whisper_model = None
summarizer = None


def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )
    return whisper_model


def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading summarizer...")
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"
        )
    return summarizer

# ==============================
# CLEAN TRANSCRIPT
# ==============================

def clean_transcript(text):
    sentences = re.split(r'(?<=[.!?]) +', text)

    seen = set()
    cleaned = []

    for s in sentences:
        s = s.strip()

        if not s:
            continue

        if s.lower() not in seen:
            cleaned.append(s)
            seen.add(s.lower())

    return " ".join(cleaned)

# ==============================
# SPLIT TEXT INTO CHUNKS
# ==============================

def chunk_text(text, chunk_size=700):
    words = text.split()

    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# ==============================
# IMPROVED SUMMARIZATION
# ==============================

def summarize_text(text):
    summarizer_model = get_summarizer()
    chunk_summaries = []

    for chunk in chunk_text(text):
        result = summarizer_model(
            chunk,
            max_length=180,
            min_length=70,
            do_sample=False
        )
        chunk_summaries.append(result[0]["summary_text"])

    combined_summary = " ".join(chunk_summaries)

    final_result = summarizer_model(
        combined_summary,
        max_length=250,
        min_length=120,
        do_sample=False
    )

    return final_result[0]["summary_text"]

# ==============================
# STRUCTURE NOTES
# ==============================

def structure_notes(summary_text):
    sentences = re.split(r'(?<=[.!?]) +', summary_text)

    sections = {
        "Introduction": [],
        "Key Concepts": [],
        "Important Points": [],
        "Applications": [],
        "Conclusion": []
    }

    total = len(sentences) if len(sentences) > 0 else 1

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()

        if not sentence:
            continue

        ratio = i / total

        if ratio < 0.2:
            sections["Introduction"].append(sentence)
        elif ratio < 0.5:
            sections["Key Concepts"].append(sentence)
        elif ratio < 0.75:
            sections["Important Points"].append(sentence)
        elif ratio < 0.9:
            sections["Applications"].append(sentence)
        else:
            sections["Conclusion"].append(sentence)

    return sections

# ==============================
# HEADING TRANSLATION
# ==============================

def translate_headings(language):
    if language == "ml":
        return {
            "Introduction": "ആമുഖം",
            "Key Concepts": "പ്രധാന ആശയങ്ങൾ",
            "Important Points": "പ്രധാനപ്പെട്ട കാര്യങ്ങൾ",
            "Applications": "പ്രയോഗങ്ങൾ",
            "Conclusion": "സമാപനം"
        }

    if language == "hi":
        return {
            "Introduction": "परिचय",
            "Key Concepts": "मुख्य अवधारणाएँ",
            "Important Points": "महत्वपूर्ण बिंदु",
            "Applications": "अनुप्रयोग",
            "Conclusion": "निष्कर्ष"
        }

    return {
        "Introduction": "Introduction",
        "Key Concepts": "Key Concepts",
        "Important Points": "Important Points",
        "Applications": "Applications",
        "Conclusion": "Conclusion"
    }

# ==============================
# TRANSLATE SUMMARY
# ==============================

def translate_summary(text, target_language):
    if target_language == "en":
        return text

    translated_lines = []
    sentences = re.split(r'(?<=[.!?]) +', text)

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        translated = GoogleTranslator(
            source="auto",
            target=target_language
        ).translate(sentence)

        translated_lines.append(translated)

    return ". ".join(translated_lines)

# ==============================
# GENERATE PDF
# ==============================

def generate_pdf(summary_text, language):
    pdf_path = os.path.join(OUTPUT_FOLDER, "Lecture_Notes.pdf")
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]

    if language == "ml" and "MalayalamFont" in pdfmetrics.getRegisteredFontNames():
        font_name = "MalayalamFont"
    elif language == "hi" and "HindiFont" in pdfmetrics.getRegisteredFontNames():
        font_name = "HindiFont"
    else:
        font_name = "Helvetica"

    title_style.fontName = font_name
    heading_style.fontName = font_name
    normal_style.fontName = font_name

    elements = []
    elements.append(Paragraph("AI Generated Lecture Notes", title_style))
    elements.append(Spacer(1, 20))

    sections = structure_notes(summary_text)
    translated_headings = translate_headings(language)

    for title, lines in sections.items():
        if not lines:
            continue

        heading = translated_headings.get(title, title)
        elements.append(Paragraph(heading, heading_style))
        elements.append(Spacer(1, 10))

        for line in lines:
            elements.append(Paragraph("• " + line, normal_style))
            elements.append(Spacer(1, 8))

        elements.append(Spacer(1, 15))

    doc.build(elements)
    return pdf_path

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return "OK", 200


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("audio")
    language = request.form.get("language")

    if not file or file.filename == "":
        return "No file uploaded.", 400

    if language not in ["en", "ml", "hi"]:
        return "Invalid language selected.", 400

    filepath = os.path.join(UPLOAD_FOLDER, "uploaded_audio.mp3")
    file.save(filepath)

    print("\n===== TRANSCRIBING AUDIO =====\n")

    model = get_whisper_model()
    segments, info = model.transcribe(filepath, beam_size=1)

    transcript = ""
    for segment in segments:
        transcript += segment.text + " "

    if not transcript.strip():
        return "Transcription failed.", 500

    cleaned = clean_transcript(transcript)

    print("Summarizing lecture...")
    summary = summarize_text(cleaned)
    summary = summary.replace("\n", " ")

    print("Translating summary...")
    summary = translate_summary(summary, language)

    print("Generating PDF...")
    pdf_path = generate_pdf(summary, language)

    return send_file(pdf_path, as_attachment=True)

# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    print("Starting AI Lecture Notes Server...")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)