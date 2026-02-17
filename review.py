import re
import json
import time
from io import BytesIO
import streamlit as st
import os
import zipfile
import pandas as pd
import PyPDF2 as pdf
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

# ── Provider configuration ────────────────────────────────────────────────────
PROVIDERS: Dict[str, dict] = {
    "Gemini": {
        "models": ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"],
        "env_key": "GEMINI_API_KEY",
        "requires_key": True,
    },
    "Claude": {
        "models": ["claude-opus-4-6", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"],
        "env_key": "CLAUDE_API_KEY",
        "requires_key": True,
    },
    "OpenAI": {
        "models": ["gpt-4o", "gpt-4.1-mini", "gpt-4-turbo"],
        "env_key": "OPENAI_API_KEY2",
        "requires_key": True,
    },
    "Ollama": {
        "models": ["llama3.2", "llama3.1", "mistral", "phi3", "gemma2", "qwen2.5"],
        "env_key": None,
        "requires_key": False,
    },
    "Groq": {
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "env_key": "GROQ_API_KEY",
        "requires_key": True,
    },
}

# ── File-type config ──────────────────────────────────────────────────────────
CHANDRA_API_URL = "https://www.datalab.to/api/v1/marker"

IMAGE_MIME_TYPES = {
    "png":  "image/png",
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "tif":  "image/tiff",
    "bmp":  "image/bmp",
    "webp": "image/webp",
}

SUPPORTED_EXTENSIONS = {"pdf", "docx", "doc"} | set(IMAGE_MIME_TYPES)

# Streamlit accepts these in the file uploader
UPLOAD_TYPES = ["zip", "pdf", "docx", "doc", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a senior HR screening specialist conducting structured first-pass longlisting for an Applied AI Specialist Advisor position in the international development / donor-funded sector.

Your task:
Evaluate the candidate’s resume strictly against the provided Job Description (JD) and produce structured, evidence-based scoring aligned to the organization’s longlist scorecard.

You must prioritize REQUIRED qualifications in the JD over general resume strength.

────────────────────────────────────────
EVALUATION PRINCIPLES
────────────────────────────────────────

1. Evidence-Based Only
- Use only information explicitly stated in the resume.
- Do NOT infer unstated skills or experience.
- Do NOT hallucinate contact details.
- If information is missing, state “Not provided” and score accordingly.

2. JD Priority Rule
- Missing REQUIRED JD qualifications must significantly reduce relevant scores.
- Preferred qualifications may increase scores but cannot compensate for missing required criteria.
- Thematic and technical scoring must reflect alignment with the JD, not generic AI experience.

3. Bias Control
- Ignore name, gender, nationality, or demographic indicators.
- Evaluate strictly on professional merit and documented experience.

4. Consistency Rule
- Ensure logical consistency across related scores.
  - Technical Expertise ≥ AI Knowledge is acceptable.
  - AI Knowledge = 5 cannot coexist with Technical Expertise ≤ 2 unless explicitly justified.
  - AI Capacity should not exceed Technical Expertise unless evidence shows strategic-only role.
- Avoid uniform scoring (e.g., all 4s). Differentiate meaningfully.

5. Conciseness Rule
- All text fields under 300 characters.
- overall_assessment under 500 characters.

────────────────────────────────────────
SCORING SCALE (MANDATORY CALIBRATION)
────────────────────────────────────────

Rate all scored criteria from 1–5 using the following anchors:

1 = No evidence or clearly unqualified  
2 = Limited or tangential evidence; major gaps  
3 = Meets minimum JD requirements; noticeable gaps  
4 = Strong alignment with JD; minor gaps only  
5 = Exceptional alignment; exceeds required criteria with clear impact  

Avoid score inflation. A score of 5 should be rare and clearly justified by strong evidence.

────────────────────────────────────────
FIELD DEFINITIONS (STRICT DISTINCTIONS)
────────────────────────────────────────

1. Name Applicant  
Full name as shown. "Not provided" if absent.

2. Phone Number  
Primary phone with country code if shown. "Not provided" if absent.

3. Email Address  
Primary email. "Not provided" if absent.

4. Educational Qualifications  
Highest relevant degrees with institution names. Note relevance to AI, data science, CS, digital transformation, or development policy.

5. Professional Certification  
Formal certifications only (AWS ML, Google Cloud AI, PMP, PRINCE2, Agile, sector credentials). "None listed" if absent.

6. Relevant Work Experience  
2–3 sentence summary of experience directly aligned to JD. Include years, advisory level, donor/digital/AI alignment.

7. AI Knowledge (1–5)  
Conceptual understanding of AI/ML frameworks, methods, and applications. Evidence: coursework, strategy papers, AI initiatives, technical discussions.  
(Not hands-on depth — that is Technical Expertise.)

8. Technical Expertise (1–5)  
Hands-on implementation depth: ML/NLP development, analytics pipelines, system architecture, automation, deployment, production systems.

9. AI Capacity (1–5)  
Evidence of building AI capacity in organizations: training delivery, AI strategy development, readiness assessments, governance frameworks, adoption leadership.

10. Specialized Donor Experience (1–5)  
Direct work with multilateral/donor agencies (UN, World Bank, USAID, FCDO, EU, GIZ, AfDB, etc.). Consider depth, duration, and relevance.

11. Thematic Relevance (1–5)  
Alignment with JD thematic areas (e.g., governance, health, agriculture, climate, education, digital public infrastructure).

12. Scale of Experience (1–5)  
Complexity and scope: multi-country programs, national systems, large budgets, cross-functional teams, regional/global impact.

13. Strategic & Advisory Skills (1–5)  
Evidence of high-level advisory work: policy briefs, executive presentations, digital roadmaps, stakeholder engagement, C-suite influence.

14. Communication & Reporting (1–5)  
Demonstrated written/oral communication: published reports, proposal writing, donor documentation, multilingual ability, structured outputs.

15. Application Completeness (1–5)  
Resume structure, clarity, completeness (dates, roles, measurable impact, contact info). Penalize missing or vague sections.

16. Overall Assessment  
Begin with one of:
- "Strongly Recommend"
- "Recommend"
- "Consider"
- "Do Not Recommend"

Then provide a concise justification referencing strongest and weakest dimensions.

────────────────────────────────────────
VERDICT LOGIC (DETERMINISTIC GUIDANCE)
────────────────────────────────────────

Strongly Recommend:
- No critical JD-required gaps
- No score below 3 in core areas (AI Knowledge, Technical Expertise, Donor Experience)
- At least three scores of 4 or 5

Recommend:
- Meets most required criteria
- Minor gaps only
- Majority scores ≥3

Consider:
- Meets some core criteria but clear gaps in required areas
- Multiple scores of 2

Do Not Recommend:
- Missing required JD qualifications
- Multiple scores of 1 in core dimensions

────────────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────────────

Respond with ONLY a valid JSON object.  
No markdown. No commentary. No explanation.

{
  "name_applicant": "",
  "phone_number": "",
  "email_address": "",
  "educational_qualifications": "",
  "professional_certification": "",
  "relevant_work_experience": "",
  "overall_assessment": "",
  "ai_knowledge": 0,
  "technical_expertise": 0,
  "ai_capacity": 0,
  "specialized_donor_experience": 0,
  "thematic_relevance": 0,
  "scale_of_experience": 0,
  "strategic_advisory_skills": 0,
  "communication_reporting": 0,
  "application_completeness": 0
}

RULES:
- All scored fields MUST be integers 1–5.
- Do not output null.
- If information is absent, score 1 and note gap in relevant text field.
- Ensure score consistency and alignment with JD.
""".strip()

USER_PROMPT = """RESUME:
{text}

JOB DESCRIPTION:
{jd}"""

# Scored fields in display order
SCORE_FIELDS = [
    ("ai_knowledge",                "AI Knowledge"),
    ("application_completeness",    "Application Completeness"),
    ("specialized_donor_experience","Donor Experience"),
    ("thematic_relevance",          "Thematic Relevance"),
    ("technical_expertise",         "Technical Expertise"),
    ("ai_capacity",                 "AI Capacity Building"),
    ("scale_of_experience",         "Scale of Experience"),
    ("strategic_advisory_skills",   "Strategic & Advisory"),
    ("communication_reporting",     "Communication & Reporting"),
]


# ── LLM dispatch ──────────────────────────────────────────────────────────────
def call_llm(
    user_prompt: str,
    provider: str,
    model: str,
    api_key: str = "",
    ollama_host: str = "http://localhost:11434",
    system_prompt: str = "",
) -> str:
    """Route a prompt to the selected LLM provider and return the response text."""

    if provider == "Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        kwargs = {"system_instruction": system_prompt} if system_prompt else {}
        m = genai.GenerativeModel(model, **kwargs)
        response = m.generate_content(user_prompt)
        return response.text

    elif provider == "Claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {"system": system_prompt} if system_prompt else {}
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": user_prompt}],
            **kwargs,
        )
        return message.content[0].text

    elif provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content

    elif provider == "Ollama":
        import requests
        payload: dict = {"model": model, "prompt": user_prompt, "stream": False}
        if system_prompt:
            payload["system"] = system_prompt
        response = requests.post(
            f"{ollama_host}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    elif provider == "Groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> Optional[str]:
    """Extract text from a PDF given its raw bytes."""
    try:
        reader = pdf.PdfReader(BytesIO(file_bytes))
        return "".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return None


def extract_docx_text(file_bytes: bytes) -> Optional[str]:
    """Extract text from a .docx file given its raw bytes."""
    try:
        import docx  # python-docx
        doc = docx.Document(BytesIO(file_bytes))
        parts: List[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts) if parts else None
    except Exception as e:
        st.error(f"Error parsing DOCX: {e}")
        return None


def ocr_with_chandra(file_bytes: bytes, filename: str, api_key: str) -> Optional[str]:
    """Submit a file (image or PDF) to Chandra (Datalab Marker) OCR and return markdown."""
    import requests as req

    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    mime = IMAGE_MIME_TYPES.get(ext, "application/pdf")

    headers = {"X-Api-Key": api_key}
    files = {
        "file": (filename, file_bytes, mime),
        "output_format": (None, "markdown"),
        "mode": (None, "accurate"),
    }
    try:
        resp = req.post(CHANDRA_API_URL, files=files, headers=headers, timeout=60)
        resp.raise_for_status()
        check_url = resp.json()["request_check_url"]

        for _ in range(300):  # poll up to 10 minutes (2 s × 300)
            time.sleep(2)
            poll = req.get(check_url, headers=headers, timeout=30)
            result = poll.json()
            if result["status"] == "complete":
                return result.get("markdown", "")
            elif result["status"] == "failed":
                st.error(f"Chandra OCR failed: {result.get('error', 'Unknown error')}")
                return None

        st.error("Chandra OCR timed out after 10 minutes.")
        return None
    except Exception as e:
        st.error(f"Chandra OCR error: {e}")
        return None


def extract_text_from_file(
    file_bytes: bytes,
    filename: str,
    chandra_api_key: str = "",
) -> Optional[str]:
    """Unified dispatcher: route to the right extractor based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return extract_pdf_text(file_bytes)

    elif ext == "docx":
        return extract_docx_text(file_bytes)

    elif ext == "doc":
        # Try python-docx; it may partially work on some .doc files
        text = extract_docx_text(file_bytes)
        if not text:
            st.warning(
                f"`{filename}`: Legacy .doc format could not be parsed. "
                "Convert to .docx for best results."
            )
        return text

    elif ext in IMAGE_MIME_TYPES:
        if not chandra_api_key:
            st.warning(
                f"`{filename}`: Image OCR requires a Chandra API key "
                "(set `CHANDRA_API_KEY` in the sidebar)."
            )
            return None
        return ocr_with_chandra(file_bytes, filename, chandra_api_key)

    else:
        st.warning(f"Unsupported file type: `{filename}`")
        return None


# ── Response parsing & review generation ─────────────────────────────────────
REQUIRED_KEYS = {
    "name_applicant", "phone_number", "email_address",
    "educational_qualifications", "professional_certification",
    "relevant_work_experience", "overall_assessment",
    "ai_knowledge", "application_completeness", "specialized_donor_experience",
    "thematic_relevance", "technical_expertise", "ai_capacity",
    "scale_of_experience", "strategic_advisory_skills", "communication_reporting",
}


def parse_response(content: str) -> Optional[Dict]:
    """Extract and validate the JSON scorecard from the LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not REQUIRED_KEYS.issubset(data.keys()):
        return None
    for key, _ in SCORE_FIELDS:
        try:
            data[key] = max(1, min(5, int(data[key])))
        except (ValueError, TypeError):
            data[key] = 1
    return data


def generate_review(
    text: str,
    jd: str,
    provider: str,
    model: str,
    api_key: str,
    ollama_host: str,
) -> Optional[Dict]:
    try:
        user_prompt = USER_PROMPT.format(text=text, jd=jd)
        content = call_llm(
            user_prompt, provider, model, api_key, ollama_host,
            system_prompt=SYSTEM_PROMPT,
        )
        result = parse_response(content)
        if result is None:
            st.error("Could not parse the model response. Raw output:")
            st.code(content)
        else:
            # Store the computed percentage so it lands in every CSV export
            result["percentage_suitability"] = overall_score(result)
        return result
    except Exception as e:
        st.error(f"Error from {provider}: {e}")
        return None


def process_zip_file(
    zip_file,
    jd: str,
    provider: str,
    model: str,
    api_key: str,
    ollama_host: str,
    chandra_api_key: str = "",
    progress_bar=None,
) -> List[Dict]:
    reviews = []
    with zipfile.ZipFile(zip_file, "r") as zf:
        candidates = [
            n for n in zf.namelist()
            if not n.endswith("/")
            and n.lower().rsplit(".", 1)[-1] in SUPPORTED_EXTENSIONS
        ]
        total = len(candidates)
        if total == 0:
            return reviews

        for idx, entry in enumerate(candidates):
            if progress_bar:
                progress_bar.progress(
                    idx / total,
                    text=f"Analyzing {entry} ({idx + 1}/{total})…",
                )
            with zf.open(entry) as f:
                file_bytes = f.read()

            ext = entry.lower().rsplit(".", 1)[-1]
            if ext in IMAGE_MIME_TYPES:
                # Images need OCR — show a spinner since it's async
                with st.spinner(f"OCR via Chandra: `{entry}`…"):
                    text = extract_text_from_file(file_bytes, entry, chandra_api_key)
            else:
                text = extract_text_from_file(file_bytes, entry, chandra_api_key)

            if text:
                review = generate_review(text, jd, provider, model, api_key, ollama_host)
                if review:
                    review["File"] = entry
                    reviews.append(review)

        if progress_bar:
            progress_bar.progress(1.0, text="Done!")
    return reviews


# ── Score helpers ─────────────────────────────────────────────────────────────
def overall_score(review: Dict) -> float:
    """Compute an overall percentage from the nine 1-5 scored fields."""
    total = sum(review.get(k, 1) for k, _ in SCORE_FIELDS)
    return round(total / (len(SCORE_FIELDS) * 5) * 100, 1)


def score_css_class(pct: float) -> str:
    if pct >= 70:
        return "score-high"
    elif pct >= 40:
        return "score-mid"
    return "score-low"


def _score_bar_html(value: int) -> str:
    """Render a 1-5 score as five filled/empty pips."""
    pips = ""
    for i in range(1, 6):
        color = "#00E5FF" if i <= value else "rgba(255,255,255,0.12)"
        pips += (
            f"<span style='display:inline-block;width:14px;height:14px;"
            f"border-radius:50%;background:{color};margin-right:3px;'></span>"
        )
    return pips


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

    .stApp {
        background-image: linear-gradient(to bottom right, #0E1117, #1A1C24);
        font-family: 'Roboto', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00E5FF !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }

    .stButton>button {
        background: linear-gradient(45deg, #00E5FF, #2979FF);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
        width: 100%;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.6);
    }

    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 10px;
        color: #FAFAFA;
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #00E5FF;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
    }

    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    .stAlert {
        background-color: rgba(0, 229, 255, 0.1);
        border: 1px solid #00E5FF;
        color: #FAFAFA;
    }

    .score-high { color: #00E676; font-weight: 700; font-size: 1.25em; }
    .score-mid  { color: #FFD740; font-weight: 700; font-size: 1.25em; }
    .score-low  { color: #FF5252; font-weight: 700; font-size: 1.25em; }

    .candidate-card {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #00E5FF;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    .provider-badge {
        display: inline-block;
        background: linear-gradient(45deg, #00E5FF22, #2979FF22);
        border: 1px solid #00E5FF55;
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 0.8em;
        color: #00E5FF;
        margin-top: 4px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ── Result renderers ──────────────────────────────────────────────────────────
def render_candidate_card(review: Dict):
    pct = overall_score(review)
    css_cls = score_css_class(pct)
    name = review.get("name_applicant", "Unknown")
    phone = review.get("phone_number", "Not provided")
    email = review.get("email_address", "Not provided")
    edu = review.get("educational_qualifications", "")
    cert = review.get("professional_certification", "")
    exp = review.get("relevant_work_experience", "")
    assessment = review.get("overall_assessment", "")
    fname = review.get("File", "")

    score_rows = ""
    items = list(SCORE_FIELDS)
    for i in range(0, len(items), 2):
        row_html = "<div style='display:flex;gap:12px;margin-bottom:8px;'>"
        for key, label in items[i:i + 2]:
            val = review.get(key, 1)
            row_html += (
                f"<div style='flex:1;background:rgba(255,255,255,0.04);border-radius:6px;padding:8px 10px;'>"
                f"<div style='font-size:0.72em;color:#90A4AE;margin-bottom:4px;'>{label}</div>"
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"{_score_bar_html(val)}"
                f"<span style='font-size:0.85em;color:#FAFAFA;font-weight:600;'>{val}/5</span>"
                f"</div></div>"
            )
        row_html += "</div>"
        score_rows += row_html

    file_html = f"<small style='color:#90A4AE;'>{fname}</small><br>" if fname else ""

    st.markdown(
        f"""
        <div class="candidate-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:1.15em;font-weight:700;color:#FAFAFA;">{name}</span>
                <span class="{css_cls}">{pct}%</span>
            </div>
            {file_html}
            <div style="display:flex;gap:16px;margin:8px 0;flex-wrap:wrap;">
                <span style="color:#90A4AE;font-size:0.82em;">📞 {phone}</span>
                <span style="color:#90A4AE;font-size:0.82em;">✉️ {email}</span>
            </div>
            <p style="margin:4px 0;font-size:0.85em;"><strong style="color:#00E5FF;">Education:</strong> <span style="color:#CFD8DC;">{edu}</span></p>
            <p style="margin:4px 0 10px;font-size:0.85em;"><strong style="color:#00E5FF;">Certifications:</strong> <span style="color:#CFD8DC;">{cert}</span></p>
            <p style="margin:0 0 8px;color:#CFD8DC;font-size:0.88em;">{exp}</p>
            {'<p style="margin:0 0 12px;padding:8px 12px;background:rgba(0,229,255,0.07);border-left:3px solid #00E5FF;border-radius:4px;color:#FAFAFA;font-size:0.88em;"><strong>Assessment:</strong> ' + assessment + '</p>' if assessment else ''}
            <div style="margin-top:4px;">{score_rows}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Column spec: (display header, review dict key, col width, wrap_text)
_EXCEL_COLUMNS = [
    ("Name",                    "name_applicant",          28,  True),
    ("Overall %",               "percentage_suitability",  12,  False),
    ("Overall Assessment",      "overall_assessment",      58,  True),
    ("Email",                   "email_address",           30,  False),
    ("Phone",                   "phone_number",            18,  False),
] + [(label, key, 20, False) for key, label in SCORE_FIELDS] + [
    ("Education",               "educational_qualifications", 42, True),
    ("Certifications",          "professional_certification", 36, True),
    ("Experience Summary",      "relevant_work_experience",   58, True),
    ("File",                    "File",                       30, False),
]

_SCORE_FILLS = {
    1: ("FF5252", "FFFFFF"),   # red   / white text
    2: ("FF7043", "FFFFFF"),   # deep-orange / white
    3: ("FFF176", "212121"),   # yellow / dark text
    4: ("A5D6A7", "1B5E20"),   # light-green / dark green text
    5: ("66BB6A", "FFFFFF"),   # green / white text
}

_ASSESS_FILLS = {
    "strongly recommend": ("C8E6C9", "1B5E20"),
    "recommend":          ("DCEDC8", "33691E"),
    "consider":           ("FFF9C4", "F57F17"),
    "do not recommend":   ("FFCDD2", "B71C1C"),
}

_SCORE_KEYS = {k for k, _ in SCORE_FIELDS}


def generate_excel(reviews: List[Dict]) -> bytes:
    """Build a colour-coded, formatted Excel workbook from the review list."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    ws = wb.active
    ws.title = "Scorecard"

    thin = Side(style="thin", color="D0D0D0")
    cell_border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def _fill(hex_color: str) -> PatternFill:
        return PatternFill("solid", fgColor=hex_color)

    def _font(bold=False, color="212121", size=10) -> Font:
        return Font(bold=bold, color=color, size=size, name="Calibri")

    def _align(wrap=False, h="left", v="top") -> Alignment:
        return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

    # ── Header row ────────────────────────────────────────────────────────────
    ws.row_dimensions[1].height = 42
    header_fill = _fill("1A237E")

    for col_idx, (header, _key, width, wrap) in enumerate(_EXCEL_COLUMNS, 1):
        from openpyxl.utils import get_column_letter
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.fill = header_fill
        cell.font = _font(bold=True, color="FFFFFF", size=10)
        cell.alignment = _align(wrap=True, h="center", v="center")
        cell.border = cell_border
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row_idx, review in enumerate(reviews, 2):
        # Alternate row background
        row_bg = "F8F9FA" if row_idx % 2 == 0 else "FFFFFF"
        ws.row_dimensions[row_idx].height = 80

        for col_idx, (header, key, _width, wrap) in enumerate(_EXCEL_COLUMNS, 1):
            value = review.get(key, "")
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.border = cell_border

            # Defaults
            cell.fill = _fill(row_bg)
            cell.font = _font(size=10)
            cell.alignment = _align(wrap=wrap, h="left" if wrap else "center")

            # Score cells (1–5)
            if key in _SCORE_KEYS and isinstance(value, int):
                score = max(1, min(5, value))
                bg, fg = _SCORE_FILLS[score]
                cell.fill = _fill(bg)
                cell.font = _font(bold=True, color=fg, size=11)
                cell.alignment = _align(h="center", v="center")

            # Overall % cell
            elif key == "percentage_suitability":
                try:
                    pct = float(value)
                except (TypeError, ValueError):
                    pct = 0.0
                if pct >= 70:
                    cell.fill = _fill("C8E6C9")
                    cell.font = _font(bold=True, color="1B5E20", size=11)
                elif pct >= 40:
                    cell.fill = _fill("FFF9C4")
                    cell.font = _font(bold=True, color="F57F17", size=11)
                else:
                    cell.fill = _fill("FFCDD2")
                    cell.font = _font(bold=True, color="B71C1C", size=11)
                cell.number_format = "0.0"
                cell.alignment = _align(h="center", v="center")

            # Assessment cell — colour by verdict keyword
            elif key == "overall_assessment":
                verdict = str(value).lower()
                for kw, (bg, fg) in _ASSESS_FILLS.items():
                    if verdict.startswith(kw):
                        cell.fill = _fill(bg)
                        cell.font = _font(color=fg, size=10)
                        break
                cell.alignment = _align(wrap=True, h="left", v="top")

    # ── Freeze header & enable auto-filter ────────────────────────────────────
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


def render_results_table(reviews: List[Dict], source_name: str):
    scores = [overall_score(r) for r in reviews]
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates", len(reviews))
    c2.metric("Avg. Score", f"{sum(scores) / len(scores):.1f}%" if scores else "N/A")
    c3.metric("Top Score", f"{max(scores):.0f}%" if scores else "N/A")

    rows = []
    for r in reviews:
        row = {
            "Name": r.get("name_applicant", ""),
            "Overall %": r.get("percentage_suitability", overall_score(r)),
            "Assessment": r.get("overall_assessment", ""),
            "Email": r.get("email_address", ""),
            "Phone": r.get("phone_number", ""),
        }
        for key, label in SCORE_FIELDS:
            row[label] = r.get(key, 1)
        row["Education"] = r.get("educational_qualifications", "")
        row["Certifications"] = r.get("professional_certification", "")
        row["Experience Summary"] = r.get("relevant_work_experience", "")
        row["File"] = r.get("File", "")
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    xlsx = generate_excel(reviews)
    st.download_button(
        "⬇ Download Scorecard (Excel)",
        xlsx,
        f"{source_name.replace('.zip', '')}_scorecard.xlsx",
        EXCEL_MIME,
        use_container_width=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        layout="wide",
        page_title="Resume Review",
        page_icon="icon.png",
    )
    inject_custom_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ AI Provider")

        provider = st.selectbox("Provider", list(PROVIDERS.keys()))
        cfg = PROVIDERS[provider]

        model = st.selectbox("Model", cfg["models"])

        ollama_host = "http://localhost:11434"
        if provider == "Ollama":
            custom_model = st.text_input(
                "Custom model name",
                placeholder="e.g. llama3.2:3b",
                help="Leave blank to use the selection above.",
            )
            if custom_model.strip():
                model = custom_model.strip()
            ollama_host = st.text_input(
                "Ollama host URL",
                value="http://localhost:11434",
            )

        api_key = ""
        if cfg["requires_key"]:
            env_val = os.getenv(cfg["env_key"], "")
            api_key = st.text_input(
                cfg["env_key"],
                value=env_val,
                type="password",
                help=f"Set {cfg['env_key']} in your .env file, or paste it here.",
            )
            if not api_key:
                st.warning(f"API key required: `{cfg['env_key']}`")

        st.divider()
        st.markdown("## 🔍 OCR (Images)")
        chandra_api_key = st.text_input(
            "CHANDRA_API_KEY",
            value=os.getenv("CHANDRA_API_KEY", ""),
            type="password",
            help=(
                "Required for image files (PNG, JPG, TIFF, BMP, WebP). "
                "Get your key at datalab.to"
            ),
        )
        if not chandra_api_key:
            st.caption("No Chandra key — image OCR disabled.")

        st.divider()
        st.markdown("### ℹ️ About")
        st.caption(
            "Upload PDFs, Word docs, images, or ZIP archives. "
            "Provides instant HR scorecard assessments."
        )
        st.markdown(
            f"<div class='provider-badge'>{provider} · {model}</div>",
            unsafe_allow_html=True,
        )

    # ── Header ────────────────────────────────────────────────────────────────
    _, col_hdr, _ = st.columns([1, 2, 1])
    with col_hdr:
        st.image("kaasor.png", width=150)
        st.markdown(
            "<h1 style='text-align:center; margin-bottom:0;'>RESUME REVIEW</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; font-size:1.2em; color:#B0BEC5; "
            "font-family:\"Orbitron\",sans-serif;'>The Perfect Hire</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center; color:#00E5FF;'>"
            f"Powered by <strong>{provider}</strong> · <code>{model}</code></p>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Job Description")
        jd = st.text_area(
            "Paste the Job Description",
            height=300,
            label_visibility="collapsed",
            placeholder="Paste the full job description here…",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.subheader("Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload Resumes",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help=(
                "Supported: PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF, BMP, WebP, ZIP. "
                "Images are OCR'd via Chandra (API key required)."
            ),
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Action ────────────────────────────────────────────────────────────────
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        submit = st.button("ANALYZE CANDIDATES", use_container_width=True)

    if not submit:
        return

    # Validation
    errors = []
    if not jd.strip():
        errors.append("Job Description is empty.")
    if not uploaded_files:
        errors.append("No resume files uploaded.")
    if cfg["requires_key"] and not api_key:
        errors.append(f"API key missing: `{cfg['env_key']}`")
    if errors:
        for err in errors:
            st.error(err)
        return

    st.markdown("---")
    st.markdown("### Results")

    all_reviews: List[Dict] = []

    for file in uploaded_files:
        ftype = file.type
        fname = file.name.lower()
        is_zip = (
            ftype in ("application/x-zip-compressed", "application/zip")
            or fname.endswith(".zip")
        )

        if is_zip:
            st.markdown(f"**Archive:** `{file.name}`")
            progress_bar = st.progress(0, text="Starting…")
            reviews = process_zip_file(
                file, jd, provider, model, api_key, ollama_host,
                chandra_api_key=chandra_api_key,
                progress_bar=progress_bar,
            )
            progress_bar.empty()

            if reviews:
                reviews.sort(key=overall_score, reverse=True)
                all_reviews.extend(reviews)
                render_results_table(reviews, file.name)
            else:
                st.warning(f"No supported resume files found in `{file.name}`.")

        else:
            ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
            is_image = ext in IMAGE_MIME_TYPES

            spinner_label = (
                f"OCR via Chandra: `{file.name}`…"
                if is_image
                else f"Analyzing `{file.name}`…"
            )
            with st.spinner(spinner_label):
                file_bytes = file.read()
                text = extract_text_from_file(file_bytes, file.name, chandra_api_key)
                if text:
                    review = generate_review(text, jd, provider, model, api_key, ollama_host)
                    if review:
                        review["File"] = file.name
                        all_reviews.append(review)
                        render_candidate_card(review)

    if len(all_reviews) > 1:
        st.markdown("---")
        xlsx_all = generate_excel(all_reviews)
        st.download_button(
            "⬇ Download Full Scorecard (Excel)",
            xlsx_all,
            "scorecard_all.xlsx",
            EXCEL_MIME,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
