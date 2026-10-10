
import os
import html
import joblib
import numpy as np
import streamlit as st

from pathlib import Path
from scipy.sparse import hstack


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="QueryIQ | Question Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "question_similarity_model.pkl"


# --------------------------------------------------
# MODERN UI STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d19;
    --panel: #111a2b;
    --border: #25334b;
    --text: #edf3ff;
    --muted: #91a0b9;
    --blue: #6c8cff;
    --purple: #a78bfa;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #17274b 0%, transparent 38%),
        radial-gradient(ellipse at 95% 15%, #251b48 0%, transparent 30%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0b1221;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.main .block-container {
    max-width: 1220px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
}

.brand-icon {
    width: 45px;
    height: 45px;
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 14px;
    background: linear-gradient(135deg, #6288ff, #a477ff);
    font-size: 23px;
    box-shadow: 0 8px 30px #657dff35;
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #ffffff;
}

.brand-sub {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.eyebrow {
    color: #9caeff;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(32px, 4.5vw, 51px);
    line-height: 1.12;
    letter-spacing: -1.8px;
    font-weight: 700;
    color: #f5f7ff;
    margin-bottom: 16px;
}

.gradient-text {
    background: linear-gradient(90deg, #83a4ff, #c4a5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    max-width: 720px;
    color: #9cabc4;
    font-size: 15px;
    line-height: 1.8;
    margin-bottom: 25px;
}

.pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 30px;
    background: #172846;
    color: #b8ccff;
    border: 1px solid #30476e;
    font-size: 12px;
    margin-right: 7px;
    margin-bottom: 7px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 600;
    color: #f4f7ff;
    margin-top: 12px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 20px;
}

.question-card {
    background: linear-gradient(145deg, #121d30, #0e1727);
    border: 1px solid #2a3953;
    border-radius: 18px;
    padding: 19px 20px 12px 20px;
    min-height: 110px;
}

.question-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    font-size: 15px;
    color: #f0f4ff;
    margin-bottom: 6px;
}

.question-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 29px;
    height: 29px;
    border-radius: 9px;
    background: #233b68;
    color: #b6cbff;
    font-size: 12px;
}

.question-hint {
    font-size: 12px;
    color: #91a0b9;
}

div[data-testid="stTextArea"] textarea {
    background: #0d1626 !important;
    color: #f2f5ff !important;
    border: 1px solid #293951 !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #829bff !important;
    box-shadow: 0 0 0 2px #829bff20 !important;
}

div.stButton > button {
    min-height: 48px;
    border: 1px solid #7e91ff;
    border-radius: 12px;
    background: linear-gradient(100deg, #547bff, #8a66ed);
    color: white;
    font-size: 15px;
    font-weight: 700;
    transition: all 0.2s ease;
    box-shadow: 0 8px 25px #6178ff25;
}

div.stButton > button:hover {
    border-color: #c2caff;
    transform: translateY(-2px);
    box-shadow: 0 12px 30px #6178ff35;
    color: white;
}

div[data-testid="stMetric"] {
    background: #111b2c;
    border: 1px solid #283852;
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #a7b5cc;
}

div[data-testid="stMetricValue"] {
    color: #f4f7ff;
    font-family: 'Space Grotesk', sans-serif;
}

.result-card {
    border-radius: 18px;
    padding: 25px;
    margin-top: 10px;
    margin-bottom: 15px;
    border: 1px solid #344b70;
    background: linear-gradient(140deg, #142542, #111a2c);
}

.result-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    color: #a8b8d4;
    margin-bottom: 10px;
}

.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 29px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}

.result-description {
    color: #a6b5cc;
    line-height: 1.7;
    font-size: 14px;
}

.feature-card {
    background: #101a2a;
    border: 1px solid #27364e;
    padding: 19px;
    border-radius: 15px;
    height: 100%;
}

.feature-icon {
    font-size: 24px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 14px;
    font-weight: 700;
    color: #f0f4ff;
    margin-bottom: 7px;
}

.feature-description {
    font-size: 12px;
    line-height: 1.7;
    color: #94a5bf;
}

.footer {
    border-top: 1px solid #26344b;
    padding-top: 22px;
    margin-top: 45px;
    color: #8292ab;
    text-align: center;
    font-size: 12px;
}

hr {
    border-color: #26344b !important;
}

.stCaption {
    color: #91a0b9;
}

@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
    }

    .hero-title {
        letter-spacing: -1px;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD EXISTING MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "q1_input" not in st.session_state:
    st.session_state.q1_input = ""

if "q2_input" not in st.session_state:
    st.session_state.q2_input = ""


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">⌕</div>
        <div>
            <div class="brand-name">QueryIQ</div>
            <div class="brand-sub">Question Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        "Choose a page",
        ["Compare Questions", "About the Project"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Information")
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">TF-IDF Vectorization</div>
        <div class="feature-description">
            Converts questions into numerical text features.
        </div>
        <br>
        <div class="feature-title">Logistic Regression</div>
        <div class="feature-description">
            Predicts whether a question pair belongs to the
            similar or not-similar class.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.caption("Built with Python · NLP · Machine Learning")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="eyebrow">AI-POWERED TEXT ANALYSIS</div>

<div class="hero-title">
    One question. Another question.<br>
    <span class="gradient-text">Are they really the same?</span>
</div>

<div class="hero-description">
    Detect duplicate and semantically related questions using
    Natural Language Processing and a trained Machine Learning model.
    Compare two questions and explore the model's prediction.
</div>

<span class="pill">✦ NLP Powered</span>
<span class="pill">◈ TF-IDF Features</span>
<span class="pill">◎ ML Prediction</span>
""", unsafe_allow_html=True)


# --------------------------------------------------
# COMPARE PAGE
# --------------------------------------------------

if page == "Compare Questions":

    st.markdown(
        '<div class="section-title">Compare your questions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Enter two questions below '
        'to analyze whether they express a similar meaning.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="question-card">
            <div class="question-heading">
                <span class="question-number">01</span>
                Question One
            </div>
            <div class="question-hint">
                Enter your first question
            </div>
        </div>
        """, unsafe_allow_html=True)

        question1 = st.text_area(
            "Question one",
            key="q1_input",
            placeholder="e.g. What is the capital of India?",
            height=145,
            label_visibility="collapsed"
        )

        st.caption(f"Characters: {len(question1)}")

    with col2:
        st.markdown("""
        <div class="question-card">
            <div class="question-heading">
                <span class="question-number">02</span>
                Question Two
            </div>
            <div class="question-hint">
                Enter the question you want to compare
            </div>
        </div>
        """, unsafe_allow_html=True)

        question2 = st.text_area(
            "Question two",
            key="q2_input",
            placeholder="e.g. Which city is India's capital?",
            height=145,
            label_visibility="collapsed"
        )

        st.caption(f"Characters: {len(question2)}")

    # Sample questions
    with st.expander("Try an example"):
        st.write("Load an example pair into the input fields.")

        sample = st.selectbox(
            "Example pair",
            [
                "Capital cities",
                "Python programming",
                "Different topics"
            ]
        )

        if st.button("Load example", use_container_width=True):
            examples = {
                "Capital cities": (
                    "What is the capital of India?",
                    "Which city is India's capital?"
                ),
                "Python programming": (
                    "What is Python programming?",
                    "Can you explain the Python language?"
                ),
                "Different topics": (
                    "What is machine learning?",
                    "How do I make a chocolate cake?"
                )
            }

            st.session_state.q1_input, st.session_state.q2_input = (
                examples[sample]
            )
            st.rerun()

    st.write("")

    analyze = st.button(
        "✦  Analyze Question Similarity",
        type="primary",
        use_container_width=True
    )

    if analyze:
        q1_text = question1.strip()
        q2_text = question2.strip()

        if not q1_text or not q2_text:
            st.warning(
                "Please enter both questions before starting the analysis."
            )

        elif not MODEL_PATH.exists():
            st.error(
                f"Model file not found: {MODEL_PATH}\n\n"
                "Check that your trained model is inside the models folder."
            )

        else:
            try:
                with st.spinner(
                    "Analyzing text features and generating prediction..."
                ):
                    saved = load_model()

                    vectorizer = saved["vectorizer"]
                    model = saved["model"]

                    # Preserve your existing feature-engineering pipeline.
                    q1 = vectorizer.transform([q1_text])
                    q2 = vectorizer.transform([q2_text])

                    difference = abs(q1 - q2)
                    shared = q1.multiply(q2)

                    features = hstack(
                        [difference, shared],
                        format="csr",
                        dtype=np.float32
                    )

                    prediction = int(model.predict(features)[0])

                    probabilities = model.predict_proba(features)[0]
                    confidence = (
                        float(probabilities[prediction]) * 100
                    )

                is_similar = prediction == 1

                st.session_state.history.insert(
                    0,
                    {
                        "question1": q1_text,
                        "question2": q2_text,
                        "result": (
                            "Similar" if is_similar else "Not Similar"
                        ),
                        "confidence": confidence
                    }
                )

                st.markdown("---")
                st.markdown(
                    '<div class="section-title">Analysis results</div>',
                    unsafe_allow_html=True
                )

                result_title = (
                    "Questions appear similar"
                    if is_similar
                    else "Questions appear different"
                )

                result_description = (
                    "The model classified this pair as similar. "
                    "The questions may express a related intent."
                    if is_similar
                    else
                    "The model classified this pair as not similar. "
                    "Their text features did not support the similar class."
                )

                result_icon = "✓" if is_similar else "↔"

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">
                        MODEL PREDICTION
                    </div>
                    <div class="result-value">
                        {result_icon} {result_title}
                    </div>
                    <div class="result-description">
                        {result_description}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric(
                        "Model confidence",
                        f"{confidence:.2f}%"
                    )

                with m2:
                    st.metric(
                        "Predicted class",
                        "Similar" if is_similar else "Not Similar"
                    )

                with m3:
                    st.metric(
                        "Questions analyzed",
                        "02"
                    )

                st.markdown("#### Confidence indicator")
                st.progress(min(max(confidence / 100, 0.0), 1.0))

                st.caption(
                    "Confidence is the model's predicted class probability. "
                    "It is not a calibrated semantic similarity score, "
                    "and it does not guarantee that the questions mean "
                    "the same thing."
                )

                with st.expander("View compared questions"):
                    st.markdown("**Question 1**")
                    st.write(q1_text)
                    st.markdown("**Question 2**")
                    st.write(q2_text)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Session history
    st.markdown("---")
    st.markdown(
        '<div class="section-title">Recent comparisons</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:
        for i, item in enumerate(st.session_state.history[:5], start=1):
            with st.expander(
                f"{i}. {item['result']} · {item['confidence']:.1f}% confidence"
            ):
                st.markdown("**Question 1**")
                st.write(item["question1"])
                st.markdown("**Question 2**")
                st.write(item["question2"])
                st.write(f"**Prediction:** {item['result']}")

        if st.button("Clear comparison history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption(
            "Your recent comparisons will appear here during this session."
        )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

else:
    st.markdown(
        '<div class="section-title">About QueryIQ</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero-description">
        QueryIQ is a Question Similarity Detection System that uses
        Natural Language Processing and supervised Machine Learning
        to classify pairs of questions.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⌕</div>
            <div class="feature-title">TF-IDF</div>
            <div class="feature-description">
                Term Frequency-Inverse Document Frequency converts
                text into numerical representations based on
                word importance within the text collection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">◈</div>
            <div class="feature-title">Logistic Regression</div>
            <div class="feature-description">
                A supervised classification algorithm used by the
                trained model to predict the class of a question pair.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### How the system works")

    st.markdown("""
    1. **Input:** The user enters two questions.
    2. **Vectorization:** The saved TF-IDF vectorizer transforms both questions.
    3. **Feature engineering:** The application calculates absolute
       differences and shared feature values.
    4. **Classification:** The trained Logistic Regression model predicts
       the class.
    5. **Results:** The interface displays the predicted class and
       model confidence.
    """)

    st.info(
        "The model's output depends on its training data and learned "
        "features. It may not recognize every pair of questions with "
        "equivalent meanings."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    QueryIQ — Question Similarity Detection System<br>
    Built with Python · Streamlit · NLP · Machine Learning
</div>
""", unsafe_allow_html=True)
