from pathlib import Path
import re
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user
from flask_login import login_required
from presidioui.models import User, SessionLocal, init_db, DetectionRule, RuleSample
from presidioui.ai import (
    generate_and_test_pattern,
    generate_sample_examples,
    run_wizard,
    WizardMessage,
    run_tests,
)
import uuid
import json
from flask_sock import Sock
from presidioui.dialog import WebsocketDialog
from presidioui.ai import improve_workflow
from presidioui import SETTINGS
import httpx
from flask import Blueprint
from presidioui.presidio import list_recognizers, anonymize_text

app = Flask(__name__)
app.config["PREFERRED_URL_SCHEME"] = "http"
app.secret_key = SETTINGS.SECRET_KEY

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "admin.login"

# Create Blueprint for all routes
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# Add after Flask app initialization
sock = Sock(app)


@login_manager.user_loader
def load_user(user_id):
    with SessionLocal() as session:
        return session.get(User, int(user_id))


@admin_bp.route("/")
@login_required
def index():
    with SessionLocal() as session:
        rules = session.query(DetectionRule).all()
        return render_template("index.html", rules=rules)


@admin_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        with SessionLocal() as session:
            if session.query(User).filter_by(email=email).first():
                flash("Email already registered")
                return redirect(url_for("register"))

            user = User(email=email)
            user.set_password(password)
            session.add(user)
            session.commit()

            login_user(user)
            return redirect(url_for("admin.index"))

    return render_template("register.html")


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        with SessionLocal() as session:
            user = session.query(User).filter_by(email=email).first()
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for("admin.index"))

            flash("Invalid email or password")

    return render_template("login.html")


@admin_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("admin.login"))


@admin_bp.route("/api/rules", methods=["POST"])
@login_required
def create_rule():
    data = request.json
    name = data.get("name")
    samples = data.get("samples", [])

    with SessionLocal() as session:
        rule = DetectionRule(name=name)
        session.add(rule)

        for sample in samples:
            rule_sample = RuleSample(
                rule=rule,
                sample_text=sample["text"],
                trigger_alert=sample["trigger_alert"],
                explanation=sample.get("explanation", ""),
            )
            session.add(rule_sample)

        session.commit()

    pattern, test_results, all_passed = generate_and_test_pattern(samples)

    # If all tests pass and we have a name, save to database
    if all_passed and name:
        with SessionLocal() as session:
            rule = DetectionRule(name=name, pattern=pattern)
            session.add(rule)

            for sample in samples:
                rule_sample = RuleSample(
                    rule=rule,
                    sample_text=sample["text"],
                    trigger_alert=sample["trigger_alert"],
                )
                session.add(rule_sample)

            session.commit()

    return jsonify(
        {"success": all_passed, "pattern": pattern, "test_results": test_results}
    )


@admin_bp.route("/api/rules/<int:rule_id>/test", methods=["POST"])
@login_required
def test_rule(rule_id):
    samples = request.json.get("samples", [])

    # Convert trigger_alert to should_trigger_alert
    converted_samples = [
        {
            "text": sample["text"],
            "should_trigger_alert": sample["trigger_alert"],
            "explanation": sample.get("explanation", ""),
        }
        for sample in samples
    ]

    with SessionLocal() as session:
        rule = session.get(DetectionRule, rule_id)
        if not rule:
            return jsonify({"error": "Rule not found"}), 404

        test_results, pct_passed, total, passed = run_tests(
            rule.pattern, converted_samples
        )

        return jsonify(
            {
                "success": pct_passed == 100.0,
                "test_results": test_results,
                "stats": {"total": total, "passed": passed, "pct_passed": pct_passed},
            }
        )


@admin_bp.route("/api/rules/<int:rule_id>/load")
@login_required
def load_rule(rule_id):
    with SessionLocal() as session:
        rule = session.get(DetectionRule, rule_id)
        if not rule:
            return jsonify({"error": "Rule not found"}), 404

        return jsonify(
            {
                "id": rule.id,
                "name": rule.name,
                "pattern": rule.pattern,
                "explanation": rule.explanation,
                "samples": [
                    {
                        "text": sample.sample_text,
                        "trigger_alert": sample.trigger_alert,
                        "explanation": sample.explanation,
                    }
                    for sample in rule.samples
                ],
            }
        )


@admin_bp.route("/api/rules/<int:rule_id>", methods=["PUT"])
@login_required
def update_rule(rule_id):
    data = request.json
    name = data.get("name")
    samples = data.get("samples", [])

    with SessionLocal() as session:
        rule = session.get(DetectionRule, rule_id)
        if not rule:
            return jsonify({"error": "Rule not found"}), 404

        rule.name = name
        # Delete existing samples
        for sample in rule.samples:
            session.delete(sample)

        # Add new samples
        for sample in samples:
            rule_sample = RuleSample(
                rule=rule,
                sample_text=sample["text"],
                trigger_alert=sample["trigger_alert"],
                explanation=sample.get("explanation", ""),
            )
            session.add(rule_sample)

        session.commit()

    pattern, test_results, all_passed = generate_and_test_pattern(samples)

    # If all tests pass, update the rule
    if all_passed:
        rule.pattern = pattern
        session.commit()

    return jsonify(
        {"success": all_passed, "pattern": pattern, "test_results": test_results}
    )


@admin_bp.route("/api/rules/<int:rule_id>", methods=["DELETE"])
@login_required
def delete_rule(rule_id):
    with SessionLocal() as session:
        rule = session.get(DetectionRule, rule_id)
        if not rule:
            return jsonify({"error": "Rule not found"}), 404

        session.delete(rule)
        session.commit()

        return jsonify({"success": True})


@admin_bp.route("/api/rules/<int:rule_id>/generate-samples", methods=["GET", "POST"])
@login_required
def generate_samples(rule_id):
    with SessionLocal() as session:
        rule = session.get(DetectionRule, rule_id)
        if not rule:
            return jsonify({"error": "Rule not found"}), 404

        # If POST, use the samples from the request
        if request.method == "POST":
            data = request.json
            existing_samples = data.get("samples", [])
        else:
            # If GET, use samples from the database
            existing_samples = [
                {"text": sample.sample_text, "trigger_alert": sample.trigger_alert}
                for sample in rule.samples
            ]

        try:
            new_samples = generate_sample_examples(rule.name, existing_samples)
            return jsonify({"samples": new_samples})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@admin_bp.route("/api/rules/generate-samples", methods=["POST"])
@login_required
def generate_samples_new():
    data = request.json
    name = data.get("name")
    existing_samples = data.get("samples", [])

    if not name:
        return jsonify({"error": "Rule name is required"}), 400

    try:
        new_samples = generate_sample_examples(name, existing_samples)
        return jsonify({"samples": new_samples})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@admin_bp.route("/wizard", methods=["GET", "POST"])
@login_required
def wizard():
    if request.method == "POST":
        data = request.json
        messages = data.get("messages", [])

        # Convert messages to WizardMessage objects
        wizard_messages = [
            WizardMessage(role=m["role"], content=m["content"]) for m in messages
        ]

        # Run the wizard
        state = run_wizard(wizard_messages)

        # If we have a complete rule, run tests
        test_results = None
        if state.is_done and state.rule and state.test_cases:
            results, pct_passed, total, passed = run_tests(state.rule, state.test_cases)
            test_results = {
                "results": results,
                "passed": passed,
                "total": total,
                "pass_percentage": pct_passed,
            }

        return jsonify(
            {
                "messages": [m.model_dump() for m in state.messages],
                "is_done": state.is_done,
                "rule": state.rule,
                "test_cases": (
                    [t.model_dump() for t in state.test_cases]
                    if state.test_cases
                    else None
                ),
                "test_results": test_results,
                "explanation": state.explanation,
                "name": state.name,
            }
        )

    return render_template("wizard.html")


@admin_bp.route("/api/rules/save-from-wizard", methods=["POST"])
@login_required
def save_wizard_rule():
    data = request.json
    name = data.get("name")
    pattern = data.get("pattern")
    explanation = data.get("explanation")
    test_cases = data.get("test_cases", [])

    with SessionLocal() as session:
        rule = DetectionRule(
            name=name,
            pattern=pattern,
            explanation=explanation,
        )
        session.add(rule)

        for test_case in test_cases:
            rule_sample = RuleSample(
                rule=rule,
                sample_text=test_case["text"],
                trigger_alert=test_case["should_trigger_alert"],
                explanation=test_case.get("explanation", ""),
            )
            session.add(rule_sample)

        session.commit()

    return jsonify({"success": True})


@admin_bp.route("/improve/<int:rule_id>")
@login_required
def improve(rule_id):
    """Render the improve workflow page."""
    return render_template("improve.html", rule_id=rule_id)


@sock.route("/admin/improve/ws/<int:rule_id>")
@login_required
def improve_websocket(ws, rule_id):
    """WebSocket endpoint for improve workflow."""
    with SessionLocal() as session:
        rule = session.get(DetectionRule, rule_id)
        if not rule:
            ws.send(json.dumps({"error": "Rule not found"}))
            return

        # Create dialog instance for websocket communication
        dialog = WebsocketDialog(ws)

        # Get rule data in the expected format
        rule_data = {
            "rule": rule.pattern,
            "name": rule.name,
            "explanation": rule.explanation,
            "test_cases": [
                {
                    "text": sample.sample_text,
                    "should_trigger_alert": sample.trigger_alert,
                    "explanation": sample.explanation,
                }
                for sample in rule.samples
            ],
        }

        # Run the improve workflow
        try:
            improved_rule, explanation, test_cases, name = improve_workflow(
                rule_data, dialog, verbose=True
            )

            # Update the rule with improved pattern and explanation
            rule.pattern = improved_rule
            rule.explanation = explanation
            rule.name = name
            # Delete old test cases
            for sample in rule.samples:
                session.delete(sample)

            # Add new test cases
            for test_case in test_cases:
                sample = RuleSample(
                    rule=rule,
                    sample_text=test_case["text"],
                    trigger_alert=test_case["should_trigger_alert"],
                    explanation=test_case.get("explanation", ""),
                )
                session.add(sample)

            session.commit()

            ws.send(json.dumps({"success": True, "pattern": improved_rule}))

        except Exception as e:
            ws.send(json.dumps({"error": str(e)}))


@admin_bp.route("/api/rules/<int:rule_id>/download")
@login_required
def download_rule(rule_id):
    with SessionLocal() as session:
        rule = session.get(DetectionRule, rule_id)
        if not rule:
            return jsonify({"error": "Rule not found"}), 404

        # Generate unique filename
        uuid_digits = str(uuid.uuid4())[:4]
        # Replace any non-alphanumeric characters with dashes
        rule_name = re.sub(r"[^a-zA-Z0-9-]", "-", rule.name)

        # Create output directory if it doesn't exist
        output_dir = Path("rules")
        output_dir.mkdir(exist_ok=True)

        # Create rule data in same format as CLI
        rule_data = {
            "rule": rule.pattern,
            "explanation": rule.explanation,
            "test_cases": [
                {
                    "text": sample.sample_text,
                    "should_trigger_alert": sample.trigger_alert,
                    "explanation": sample.explanation,
                }
                for sample in rule.samples
            ],
        }

        # Save rule to JSON file
        rule_file = output_dir / f"{uuid_digits}-{rule_name}-rule.json"
        rule_file.write_text(json.dumps(rule_data, indent=2))

        return jsonify(
            {
                "success": True,
                "filename": str(rule_file),
                "message": f"Rule saved to: {rule_file}",
            }
        )


@admin_bp.route("/reload", methods=["POST"])
@login_required
def reload_server():
    """Endpoint to reload the proxy server engines."""
    try:
        # Make POST request to proxy server
        response = httpx.post("http://localhost:8080/reload-engines")
        data = response.json()

        return jsonify(
            {
                "success": True,
                "message": data.get("message", "Server restarted successfully"),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@admin_bp.route("/playground")
@login_required
def playground():
    """Render the playground page."""
    recognizers = list_recognizers()
    return render_template("playground.html", rules=recognizers)


@admin_bp.route("/api/anonymize", methods=["POST"])
@login_required
def anonymize():
    data = request.json
    text = data.get("text")
    recognizers = data.get("recognizers", None)
    anonymized = anonymize_text(text, recognizers)
    return jsonify({"success": True, "anonymized": anonymized})


# Register the blueprint
app.register_blueprint(admin_bp)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
