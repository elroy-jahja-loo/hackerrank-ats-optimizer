import sys
import os
import tempfile
import logging

from flask import Flask, request, jsonify, render_template

_app_dir = os.path.dirname(os.path.abspath(__file__))
_hiring_agent_path = os.path.join(_app_dir, "hiring-agent")
sys.path.insert(0, _hiring_agent_path)

from dotenv import load_dotenv
load_dotenv(os.path.join(_hiring_agent_path, ".env"))

from pdf import PDFHandler
from evaluator import ResumeEvaluator
from transform import convert_json_resume_to_text
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORY_MAXES = {
    "open_source": 35,
    "self_projects": 30,
    "production": 25,
    "technical_skills": 10,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(tmp_path)

        if not resume_data:
            return jsonify({"error": "Failed to parse resume. Make sure it is a readable PDF."}), 500

        model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9})
        evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)
        resume_text = convert_json_resume_to_text(resume_data)
        evaluation = evaluator.evaluate_resume(resume_text)

        scores = evaluation.scores
        total_score = 0
        scores_out = {}
        for cat, max_val in CATEGORY_MAXES.items():
            cat_obj = getattr(scores, cat)
            capped = min(cat_obj.score, max_val)
            total_score += capped
            scores_out[cat] = {
                "score": capped,
                "raw_score": cat_obj.score,
                "max": cat_obj.max,
                "evidence": cat_obj.evidence,
            }

        total_score += evaluation.bonus_points.total
        total_score -= evaluation.deductions.total
        total_score = max(-20, min(120, total_score))

        return jsonify({
            "resume": resume_data.model_dump(),
            "evaluation": {
                "scores": scores_out,
                "bonus_points": evaluation.bonus_points.model_dump(),
                "deductions": evaluation.deductions.model_dump(),
                "key_strengths": evaluation.key_strengths,
                "areas_for_improvement": evaluation.areas_for_improvement,
                "total_score": round(total_score, 1),
                "max_score": 120,
            },
        })
    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(debug=True, port=5050)
