from jinja2 import Environment, DictLoader, StrictUndefined


# Define macros
macro_template = """
{% macro rule_report(rule, explanation, results, pct_passed) %}
# Rule summary
- Regex: {{ rule }}
- Explanation: {{ explanation }}

# Results

Overall passing: {{ "%.1f"|format(pct_passed) }}%

{% for result in results %}
## Test case {{ loop.index }}
- Test case: {{ result.text }}
- Should {{ "match" if result.should_trigger else "not match" }}
- Actually {{ "matches" if result.actually_triggered else "does not match" }}
- Result: {{ "PASS" if result.passed else "FAIL" }}
- Explanation: {{ result.explanation }}
{% endfor %}
{% endmacro %}

{% macro conversation_history(messages) %}
# Conversation History (most recent at the end)
{% for message in messages %}
## {{ message.role }}
{{ message.content }}
{% endfor %}
{% endmacro %}
"""

# Create environment with macros
env = Environment(
    loader=DictLoader(
        {
            "macros.html": macro_template,
        }
    ),
    undefined=StrictUndefined,
)


def create_template(content: str):
    return env.from_string(content)
