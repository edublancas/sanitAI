"""AI module for generating regex patterns using OpenAI."""

import json
import uuid
from pathlib import Path
from copy import deepcopy
import re

import openai
import click
import structlog
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from presidioui.presidio import SingleRulePresidioAnalyzer, Rule
from presidioui.io_models import (
    WizardMessage,
    WizardState,
    WizardResponse,
    ModifyRuleResponse,
    RouterResponse,
    RuleImprovement,
    NextActionResponse,
)
from presidioui.templates import create_template
from presidioui import SETTINGS
from presidioui.dialog import CLIDialog, DialogProtocol

logger = structlog.get_logger()


def generate_and_test_pattern(samples):
    """Generate a regex pattern using OpenAI and test it against samples.

    Parameters
    ----------
    samples : list of dict
        List of dictionaries containing 'text' and 'trigger_alert' keys

    Returns
    -------
    tuple
        (pattern, test_results, all_passed)
        - pattern: str, the generated regex pattern
        - test_results: list of test result dictionaries
        - all_passed: bool, whether all tests passed
    """
    # Generate regex using OpenAI
    sample_descriptions = [
        f"Match: {s['text']}" if s["trigger_alert"] else f"Don't match: {s['text']}"
        for s in samples
    ]

    prompt = f"""Generate a regular expression that satisfies these requirements:
    {'\n'.join(sample_descriptions)}
    
    Return only the regular expression pattern, nothing else."""

    response = openai.OpenAI().chat.completions.create(
        model=SETTINGS.MODEL_REASONING,
        messages=[
            {
                "role": "system",
                "content": "You are a regex expert. Respond only with the regex pattern.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    pattern = response.choices[0].message.content.strip()
    logger.info("generated_regex_pattern", pattern=pattern)

    # Test the pattern against samples
    test_results = []
    for sample in samples:
        matches = bool(re.search(pattern, sample["text"]))
        is_correct = matches == sample["trigger_alert"]
        test_results.append(
            {
                "sample": sample["text"],
                "expected": sample["trigger_alert"],
                "actual": matches,
                "passed": is_correct,
            }
        )

    all_passed = all(r["passed"] for r in test_results)
    logger.info(
        "pattern_test_results", test_results=test_results, all_passed=all_passed
    )
    return pattern, test_results, all_passed


def generate_sample_examples(rule_name: str, existing_samples: list) -> list:
    """Generate new sample examples using AI based on existing samples.

    Parameters
    ----------
    rule_name : str
        Name of the rule to generate samples for
    existing_samples : list
        List of dictionaries containing 'text' and 'trigger_alert' keys

    Returns
    -------
    list
        List of generated sample texts
    """
    # Create a prompt that explains the current rule and samples
    examples = []
    for s in existing_samples:
        trigger_status = "Triggers alert" if s["trigger_alert"] else "Does not trigger"
        examples.append(f"Example: {s['text']}\nStatus: {trigger_status}")

    examples_desc = "\n\n".join(examples)

    prompt = f"""Based on the following rule named "{rule_name}" and its examples:

{examples_desc}

Generate 3 new, different examples that could be used to test this rule. 
Make them diverse but related to the same concept as the existing examples.
Return ONLY the example text, one per line, without any additional explanation or status.

For example, if generating email addresses, return like this:
john.doe@company.com
support@website.net
user123@domain.org
"""

    response = openai.OpenAI().chat.completions.create(
        model=SETTINGS.MODEL_REASONING,
        messages=[
            {
                "role": "system",
                "content": "You are a data expert helping to generate test cases for PII detection rules. Respond only with the examples, one per line.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    logger.info("model_response", content=response.choices[0].message.content)

    # Split the response into lines and clean them up
    samples = [
        line.strip()
        for line in response.choices[0].message.content.strip().split("\n")
        if line.strip() and ":" not in line and "(" not in line
    ]
    return samples[:3]  # Ensure we only return 3 samples


def run_wizard(previous_messages: list) -> WizardResponse:
    """Interactive wizard for generating test samples."""
    messages = deepcopy(previous_messages)

    if not previous_messages:
        messages.append(
            WizardMessage(
                role="system",
                content="Describe the PII you want to detect.",
            )
        )
        return WizardState(
            is_done=False,
            rule=None,
            test_cases=None,
            messages=messages,
            explanation=None,
            name=None,
        )

    messages_dict = [m.model_dump() for m in messages]

    client = openai.OpenAI()

    SYSTEM_PROMPT = """
You are a helpful assistant that helps users generate test cases for PII detection rules.

Your job is to ask clarifying questions (output them as a numbered list) so you can
generate a PII detection rule (a Python regex pattern) that can be used to
detect PII in text. You'll also need to generate test cases that can be used to
test the rule.

You'll start with the rule name. Then, on each subsequent
call, I'll pass the previous questions and answers to you.

# Response Format

You'll respond with a JSON object that contains the following fields:

- `reply_to_user`: str, the message to reply to the user
- `is_done`: bool, whether you have enough information to generate the rule
- `rule`: str, the regex pattern to use for the rule, only if `is_done` is True. The pattern should be a Python regex pattern.
- `test_cases`: list[WizardTestCase], the test cases to use for the rule, only if `is_done` is True. Create a variety of test cases that cover different scenarios, both that should trigger an alert and that should not trigger an alert.
- `explanation`: str, an explanation of the rule so users can understand it better. This should be a natural language explanation of the regex, without making any references to the regex syntax.
- `name`: str, an identifier for the entity that the rule is detecting. This should be a single word or short phrase. For example: EMAIL_ADDRESS, US_SOCIAL_SECURITY_NUMBER, US_BANK_NUMBER, etc.

The `WizardTestCase` object has the following fields:

- `text`: str, the text to test
- `should_trigger_alert`: bool, whether the text should trigger an alert or not
- `explanation`: str, a short explanation of why the text should trigger an alert or not
"""

    response = (
        client.beta.chat.completions.parse(
            model=SETTINGS.MODEL_REASONING,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages_dict,
            ],
            response_format=WizardResponse,
        )
        .choices[0]
        .message.parsed
    )

    logger.info("wizard_response", response=response)

    messages.append(
        WizardMessage(
            role="system",
            content=response.reply_to_user,
        )
    )

    return WizardState(
        is_done=response.is_done,
        rule=response.rule,
        test_cases=response.test_cases,
        messages=messages,
        explanation=response.explanation,
        name=(
            None
            if response.name is None
            else re.sub(r"[^a-zA-Z0-9]", "_", response.name).upper()
        ),
    )


@click.group()
def cli():
    """Command line interface for Presidio UI tools."""
    pass


@cli.command()
@click.argument("rule_file", type=click.Path(exists=True))
def test(rule_file: str):
    """Test a rule against its test cases."""
    rule_data = json.load(open(rule_file))
    rule = rule_data["rule"]
    test_cases = rule_data["test_cases"]
    run_tests_and_display_results(rule, test_cases, rule_data["explanation"])


def run_tests(
    rule: str, test_cases: list, dialog: DialogProtocol = None
) -> tuple[list[dict], float, int, int]:
    """Run tests against a rule and return results.

    Parameters
    ----------
    rule : str
        Regex pattern to test
    test_cases : list
        List of test cases to validate
    dialog : DialogProtocol, optional
        The dialog implementation

    Returns
    -------
    tuple[list[dict], float, int, int]
        - List of test result dictionaries containing:
            - text: str, the test text
            - should_trigger: bool, whether it should trigger
            - actually_triggered: bool, whether it actually triggered
            - passed: bool, whether the test passed
            - explanation: str, explanation for the test case
        - Number of passed tests
        - Total number of tests
    """
    analyzer = SingleRulePresidioAnalyzer(Rule(name="test_rule", regex=rule))

    passed = 0
    total = 0
    results = []

    # Handle both Pydantic model and dict test cases
    for test_case in test_cases:
        total += 1
        text = test_case.text if hasattr(test_case, "text") else test_case["text"]
        should_trigger = (
            test_case.should_trigger_alert
            if hasattr(test_case, "should_trigger_alert")
            else test_case["should_trigger_alert"]
        )
        explanation = (
            test_case.explanation
            if hasattr(test_case, "explanation")
            else test_case.get("explanation", "")
        )

        test_results = analyzer.test_samples([text])
        actually_triggered = test_results[0]
        passed_test = actually_triggered == should_trigger
        if passed_test:
            passed += 1

        results.append(
            {
                "text": text,
                "should_trigger": should_trigger,
                "actually_triggered": actually_triggered,
                "passed": passed_test,
                "explanation": explanation,
            }
        )

    if dialog:
        dialog.display_table(
            title="Test results",
            columns=[
                "text",
                "should_trigger",
                "actually_triggered",
                "passed",
                "explanation",
            ],
            rows=results,
        )

    pct_passed = sum(1 for r in results if r["passed"]) / len(results) * 100

    return results, pct_passed, total, passed


def run_tests_and_display_results(rule, test_cases, explanation, title="Test Cases"):
    """Run tests and display results in a table.

    Parameters
    ----------
    rule : str
        Regex pattern to test
    test_cases : list
        List of test cases to validate
    explanation : str
        Explanation of the rule
    title : str, optional
        Title for the results table

    Returns
    -------
    tuple
        (passed, total) counts of test results
    """
    # Create table for test cases and validation results
    table = Table(title=title)
    table.add_column("Text")
    table.add_column("Should Trigger Alert", justify="center")
    table.add_column("Actually Triggered", justify="center")
    table.add_column("Result", justify="center")
    table.add_column("Explanation")

    results, pct_passed, total, passed = run_tests(rule, test_cases)

    for result in results:
        table.add_row(
            result["text"],
            "✓" if result["should_trigger"] else "✗",
            "✓" if result["actually_triggered"] else "✗",
            "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]",
            result["explanation"],
        )

    # Print results
    rprint(Panel.fit(f"Rule: {rule}", title="Rule"))
    rprint(Panel.fit(f"Explanation: {explanation}", title="Rule Explanation"))
    rprint(table)
    rprint(f"\nPassed {passed}/{total} tests ({pct_passed:.1f}%)")

    return pct_passed


@cli.command()
def wizard():
    """Interactive wizard for generating test samples."""

    state = WizardState(
        is_done=False,
        rule=None,
        test_cases=None,
        messages=[],
        explanation=None,
    )

    while not state.is_done:
        state = run_wizard(state.messages)
        question = state.messages[-1].content

        if state.is_done:
            break

        answer = click.prompt(question, prompt_suffix="")
        state.messages.append(WizardMessage(role="user", content=answer))

    # Run tests and display results
    pct_passed = run_tests_and_display_results(
        state.rule,
        state.test_cases,
        state.explanation,
    )

    # Generate unique filename
    uuid_digits = str(uuid.uuid4())[:4]
    # Replace any non-alphanumeric characters with dashes
    rule_name = re.sub(r"[^a-zA-Z0-9-]", "-", state.messages[1].content)

    # Create output directory if it doesn't exist
    output_dir = Path("rules")
    output_dir.mkdir(exist_ok=True)

    # Save messages to JSON file
    messages_file = output_dir / f"{uuid_digits}-{rule_name}-messages.json"
    with messages_file.open("w") as f:
        json.dump([m.model_dump() for m in state.messages], f, indent=2)

    # Save rule and test cases to JSON file
    rule_file = output_dir / f"{uuid_digits}-{rule_name}-rule.json"
    rule_data = {
        "rule": state.rule,
        "test_cases": [t.model_dump() for t in state.test_cases],
        "explanation": state.explanation,
    }
    with rule_file.open("w") as f:
        json.dump(rule_data, f, indent=2)

    rprint(f"\nMessages saved to: {messages_file}")
    rprint(f"Rule and test cases saved to: {rule_file}")


def run_fixer(
    rule: str, test_cases: list[dict], max_attempts: int = 3
) -> tuple[str, str]:
    """Run the fixer to improve the regex rule based on test results.

    Parameters
    ----------
    rule : str
        The current regex pattern to fix
    test_cases : list[dict]
        List of test case dictionaries containing 'text', 'should_trigger_alert',
        'explanation' keys
    max_attempts : int, optional
        Maximum number of attempts to fix the rule, by default 3

    Returns
    -------
    tuple[str, str]
        The fixed regex pattern and failure explanation
    """
    analyzer = SingleRulePresidioAnalyzer(Rule(name="test_rule", regex=rule))
    test_results = analyzer.test_samples_full_report(test_cases)

    current_rule = rule
    attempt = 0
    attempted_rules = []
    attempted_results = []

    while attempt < max_attempts:
        # Check if all tests pass
        if all(r["passed"] for r in test_results):
            # TODO: generate explanation so we can return it
            return True, current_rule, None, test_results

        # Create prompt describing current rule and all test cases
        system_prompt = create_template(
            """
You are a Python regex expert that can improve Python regex patterns. You'll be
given a regex pattern and a list of test cases. Your job is to improve the regex
pattern to handle all the test cases correctly.

# Current Pattern

{{ current_rule }}

# Test Cases

{% for test_case in test_results %}
## Test Case {{ loop.index }}
- Test case: {{ test_case.text }}
- Should {{ "match" if test_case.should_trigger_alert else "not match" }} 
  (Expected: {{ test_case.should_trigger_alert }}, 
   Actual: {{ "matches" if test_case.actual else "does not match" }}, 
   Result: {{ "PASS" if test_case.passed else "FAIL" }})
- Explanation: {{ test_case.explanation|default("No explanation provided", true) }}
{% endfor %}

# Response

Generate an improved and valid Python regex pattern that correctly handles these
cases. Return only the regex pattern, nothing else.
"""
        ).render(
            current_rule=current_rule,
            test_results=test_results,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        logger.info("fixer_messages", messages=messages)

        response = openai.OpenAI().chat.completions.create(
            model=SETTINGS.MODEL_REASONING,
            messages=messages,
        )

        # Update rule and test results
        new_rule = response.choices[0].message.content.strip()
        attempted_rules.append(new_rule)
        logger.info("improved_regex_pattern", pattern=new_rule, attempt=attempt + 1)

        # Track changes in test results
        previous_passed = sum(1 for r in test_results if r["passed"])
        previous_results = {r["text"]: r["passed"] for r in test_results}

        # Update test results with new pattern
        for result in test_results:
            matches = bool(re.search(new_rule, result["text"]))
            result["actual"] = matches
            result["passed"] = matches == result["should_trigger_alert"]

        # Store attempt results
        attempted_results.append([{**r} for r in test_results])

        new_passed = sum(1 for r in test_results if r["passed"])

        # Log changes
        logger.info("rule_changes", old_rule=current_rule, new_rule=new_rule)
        for result in test_results:
            was_passing = previous_results[result["text"]]
            now_passing = result["passed"]
            if was_passing != now_passing:
                status = "fixed" if now_passing else "newly failing"
                logger.info(
                    "test_case_change",
                    text=result["text"],
                    status=status,
                    should_trigger_alert=result["should_trigger_alert"],
                )

        improvement = (new_passed - previous_passed) / len(test_results) * 100
        logger.info(
            "improvement_summary",
            previous_passing=previous_passed,
            now_passing=new_passed,
            total_cases=len(test_results),
            improvement_percentage=f"{improvement:+.1f}%",
        )

        current_rule = new_rule
        attempt += 1

    # If we get here, we failed to fix the rule after max attempts
    analysis_prompt = create_template(
        """

You're a regex expert and test case expert. You're part of a system that allows users
detect PII in text using regex patterns with a user interface. One feature in the
system allows updating the regex pattern to ensure all its test cases pass. However,
the system has failed to fix the rule after a few attempts.

Your job is to analyze the initial pattern and the attempts made, and provide an
explanation of why the system has failed to fix the rule. The regex is not displayed
to the user so phrase your explanation in terms of the test cases.

Your answer should be short and provide potential solutions to update the test cases
so it's easier to fix the rule. For example, you might find that test cases are
inconsistent (e.g., the same text is flagged as "match" and "not match").


# Initial Pattern

{{ initial_rule }}

# Attempts

{% for attempt in attempts %}
## Attempt {{ loop.index }}

Pattern: {{ attempt.rule }}

Test Results:
{% for result in attempt.results %}
Test case: {{ result.text }}
Should {{ "match" if result.should_trigger_alert else "not match" }}
Actual: {{ "matches" if result.actual else "does not match" }}
Result: {{ "PASS" if result.passed else "FAIL" }}
Explanation: {{ result.explanation }}
---
{% endfor %}

{% endfor %}

# Response

Your response should be in markdown format.
"""
    ).render(
        initial_rule=rule,
        attempts=[
            {"rule": r, "results": res}
            for r, res in zip(attempted_rules, attempted_results)
        ],
    )

    messages = [
        {
            "role": "system",
            "content": "You are a regex expert. Analyze why the attempted patterns failed to handle all test cases.",
        },
        {"role": "user", "content": analysis_prompt},
    ]

    response = openai.OpenAI().chat.completions.create(
        model=SETTINGS.MODEL_REASONING,
        messages=messages,
    )

    failure_explanation = response.choices[0].message.content.strip()

    rprint(
        Panel.fit(
            Markdown(failure_explanation),
            title="Analysis of Failed Attempts",
            border_style="red",
        )
    )

    return False, current_rule, failure_explanation, test_results


@cli.command()
@click.argument("rule_file", type=click.Path(exists=True))
@click.option("--no-save", "-n", is_flag=True, help="Skip saving the fixed rule")
@click.option(
    "--max-attempts",
    "-m",
    type=int,
    default=3,
    help="Maximum number of attempts to fix the rule",
)
def fixer(rule_file, no_save, max_attempts):
    """Run the fixer to improve the regex rule based on test results.

    Parameters
    ----------
    rule_file : str
        Path to JSON file containing rule and test cases
    no_save : bool
        If True, skip saving the fixed rule
    max_attempts : int
        Maximum number of attempts to fix the rule
    """
    # Load rule file
    rule_path = Path(rule_file)
    rule_data = json.loads(rule_path.read_text())
    rule = rule_data["rule"]
    test_cases = rule_data["test_cases"]
    explanation = rule_data["explanation"]
    # Run initial tests
    pct_passed = run_tests_and_display_results(rule, test_cases, explanation)

    if pct_passed < 100:
        # Run fixer to improve rule
        fixed_rule, failure_explanation = run_fixer(
            rule, test_cases, max_attempts=max_attempts
        )

        # Test fixed rule
        pct_passed = run_tests_and_display_results(
            fixed_rule,
            test_cases,
            explanation,
            title="Test Cases with Fixed Rule",
        )

        # Only save if all tests pass and --no-save not specified
        if pct_passed == 100 and not no_save:
            # Save fixed rule with version suffix
            stem = rule_path.stem
            suffix = rule_path.suffix

            # Extract current version if it exists
            if match := re.search(r"-v(\d+)$", stem):
                version = int(match.group(1)) + 1
                new_stem = re.sub(r"-v\d+$", f"-v{version}", stem)
            else:
                new_stem = f"{stem}-v2"

            new_name = f"{new_stem}{suffix}"
            new_path = rule_path.parent / new_name

            # Save fixed rule
            rule_data["rule"] = fixed_rule
            rule_data["name"] = new_name
            new_path.write_text(json.dumps(rule_data, indent=2))
            rprint(f"\nFixed rule saved to: {new_path}")
        else:
            if pct_passed == 100:
                rprint("\nFixed rule found but not saved (--no-save specified)")
            else:
                rprint(
                    "\nNo improved rule found that passes all tests. "
                    f"Explanation: {failure_explanation}"
                )


@cli.command()
@click.argument("rule")
@click.argument("sample")
def test_sample(rule: str, sample: str):
    """Test a single sample against a regex rule.

    Parameters
    ----------
    rule : str
        Regex pattern to test
    sample : str
        Text sample to test against the rule
    """
    analyzer = SingleRulePresidioAnalyzer(Rule(name="test_rule", regex=rule))
    result = analyzer.test_samples([sample])[0]

    rprint(Panel.fit(f"Rule: {rule}", title="Rule"))
    rprint(Panel.fit(f"Sample: {sample}", title="Sample"))
    rprint(f"\nResult: {'[green]MATCH[/green]' if result else '[red]NO MATCH[/red]'}")


def get_next_action(
    dialog: DialogProtocol,
    user_input: str,
    messages: list[WizardMessage],
    verbose: bool,
):
    """Route the user input to the appropriate suggestion function.

    Parameters
    ----------
    dialog : DialogProtocol
        The dialog implementation
    user_input : str
        The user input
    messages : list[WizardMessage]
        The history of the conversation so far
    verbose : bool
        Whether to print the system prompt and other information
    """

    system_prompt = create_template(
        """
{% from "macros.html" import conversation_history %}

You're an action classifier that decides what to do based on the user's input. You're
part of an AI assistant that allows users improve their regex patterns for PII
detection.

You'll be given the history of the conversation so far, and you should decide what
to do next based on the user's input.

# Actions

You should respond with one of the following actions:


## EDIT_TEST_CASES

Respond with this action if the user wants to add, remove (or both) test cases from
the rule. They might say something like "Help me review the test cases, suggest some
to add and some to remove", "Delete redundant test cases", "Add a test case for this
new scenario", etc.

## MODIFY_RULE

Respond with this action if the user wants to modify the rule. For example they
might say something like "I think the rule is not good, please suggest a better
one" or "I want to change the rule to be more specific" or "I want to change the
rule to be more general".

## UNKNOWN

Respond with this action if the user's input does not match any of the above actions.

{{ conversation_history(messages) }}
"""
    ).render(messages=messages)

    messages_all = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": "the user input is: " + user_input},
    ]

    dialog.set_as_loading()
    response = openai.OpenAI().beta.chat.completions.parse(
        model=SETTINGS.MODEL_DEFAULT,
        messages=messages_all,
        response_format=RouterResponse,
    )

    response_parsed = response.choices[0].message.parsed

    if verbose:
        dialog.log("next_action_suggested", action=response_parsed.action)

    return response_parsed.action


def suggest_test_cases(
    dialog: DialogProtocol,
    rule_data: dict,
    user_prompt: str,
    messages: list[WizardMessage] = None,
):
    """Suggest test cases to add or remove to improve a rule.

    Parameters
    ----------
    dialog : DialogProtocol
        The dialog implementation
    rule_data : dict
        Dictionary containing rule and test cases

    user_prompt : str
        The user prompt

    messages : list[WizardMessage], optional
        Chat history
    """
    rule = rule_data["rule"]
    test_cases = rule_data["test_cases"]
    explanation = rule_data["explanation"]

    dialog.set_as_loading()
    results, pct_passed, _, _ = run_tests(rule, test_cases)

    system_prompt = create_template(
        """
{% from "macros.html" import rule_report %}
        
You're a helpful assistant that helps users improve their regex patterns. Helping
involves strictly the following scenarios:

- Suggest new test cases that could be added to improve the rule (if you see any
  test cases that are not being tested)
- Suggest removing test cases that are not needed

You'll be given a user prompt and you should reply with the a message to show
to the user, summarizing the changes you'll make to the rule and test cases.

# Response Format

You should respond with a JSON object that contains the following fields:

## explanation

An explanation of the proposed changes. Do not include the test cases here, we'll
display them separately. Don't use the term "regular expression" or "regex" in your
explanation, use the term "rule" instead.

## test_cases_suggested

A list of test cases that could be added to improve the rule. Each one should have the
following fields:

- `text`: str, the text to test
- `explanation`: str, a short explanation of why the text should trigger an alert or not
- `should_trigger_alert`: bool, whether the text should trigger an alert or not

## test_cases_removed

A list of test cases that could be removed to improve the rule. Each one should
have the following fields:

- `text`: str, the text to test
- `explanation`: str, a short explanation of why we should remove this test case

{{ rule_report(rule, explanation, results, pct_passed) }}
"""
    ).render(
        rule=rule,
        explanation=explanation,
        results=results,
        pct_passed=pct_passed,
    )

    # Display the system prompt in a nicely formatted panel
    rprint(
        Panel.fit(
            Markdown(system_prompt),
            title="System Prompt",
            border_style="blue",
        )
    )

    messages_all = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": "The user prompt is: " + user_prompt,
        },
    ]

    response = openai.OpenAI().beta.chat.completions.parse(
        model=SETTINGS.MODEL_REASONING,
        messages=messages_all,
        response_format=RuleImprovement,
    )

    response_parsed = response.choices[0].message.parsed

    # Display improvement suggestions
    dialog.display(response_parsed.explanation)

    if response_parsed.test_cases_suggested:
        dialog.display_table(
            title="Suggested Test Cases to Add",
            columns=["Text", "Should Trigger Alert", "Explanation"],
            rows=[
                {
                    "Text": case.text,
                    "Should Trigger Alert": "✓" if case.should_trigger_alert else "✗",
                    "Explanation": case.explanation,
                }
                for case in response_parsed.test_cases_suggested
            ],
        )

    if response_parsed.test_cases_removed:
        dialog.display_table(
            title="Suggested Test Cases to Remove",
            columns=["Text", "Explanation"],
            rows=[
                {"Text": case.text, "Explanation": case.explanation}
                for case in response_parsed.test_cases_removed
            ],
        )

    return response_parsed


def modify_rule(
    dialog: DialogProtocol, rule_data: dict, user_input: str
) -> tuple[bool, str, str, list[dict]]:
    """Modify the rule.

    Parameters
    ----------
    dialog : DialogProtocol
        The dialog implementation
    rule_data : dict
        Dictionary containing rule information and test cases
    user_input : str
        The user input
    """
    rule = rule_data["rule"]
    test_cases = rule_data["test_cases"]

    results, pct_passed, _, _ = run_tests(rule, test_cases)

    system_prompt = create_template(
        """
You're a helpful assistant that helps users improve their regex patterns. Your job
is to modify the rule based on the user's request.

# Response Format

You should respond with a JSON object that contains the following fields:

## explanation

an explanation of the rule so users can understand it better. This should be a
natural language explanation of the regex, without making any references to the
regex syntax. Don't use the term "regular expression" or "regex" in your
explanation, use the term "rule" instead.

## summary_of_changes

An explanation of the proposed changes. Do not include the test cases here, we'll
display them separately. Just explain the difference between the old and new rule.
Use natural language, without making any references to the regex syntax.

## rule

The new regex pattern to use for the rule.

## test_cases_suggested

A list of test cases that could be added. They should be aimed to test the new rule.

Each test case should have the following fields:

- `text`: str, the text to test
- `explanation`: str, a short explanation of why the text should trigger an alert or not
- `should_trigger_alert`: bool, whether the text should trigger an alert or not

## test_cases_removed

A list of test cases that could be removed (because they are not needed anymore).

Each test case should have the following fields:

- `text`: str, the text to test
- `explanation`: str, a short explanation of why we should remove this test case

# Rule

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
    """
    ).render(
        rule=rule_data["rule"],
        explanation=rule_data["explanation"],
        results=results,
        pct_passed=pct_passed,
    )

    response = openai.OpenAI().beta.chat.completions.parse(
        model=SETTINGS.MODEL_REASONING,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "the user request is: " + user_input,
            },
        ],
        response_format=ModifyRuleResponse,
    )

    response_parsed = response.choices[0].message.parsed

    # Display the modified rule
    dialog.display(response_parsed.summary_of_changes)

    # Display added test cases if any
    if response_parsed.test_cases_suggested:
        rows = []
        for i, test_case in enumerate(response_parsed.test_cases_suggested, 1):
            rows.append(
                {
                    "#": str(i),
                    "Text": test_case.text,
                    "Should Trigger Alert": (
                        "✓" if test_case.should_trigger_alert else "✗"
                    ),
                    "Explanation": test_case.explanation,
                }
            )
        dialog.display_table(
            title="Added Test Cases",
            columns=["#", "Text", "Should Trigger Alert", "Explanation"],
            rows=rows,
        )

    # Display removed test cases if any
    if response_parsed.test_cases_removed:
        rows = []
        for i, test_case in enumerate(response_parsed.test_cases_removed, 1):
            rows.append(
                {
                    "#": str(i),
                    "Text": test_case.text,
                    "Explanation": test_case.explanation,
                }
            )
        dialog.display_table(
            title="Removed Test Cases", columns=["#", "Text", "Explanation"], rows=rows
        )

    test_cases_combined = combine_test_cases(rule_data, response_parsed)
    _, pct_passed, _, _ = run_tests(rule, test_cases_combined)
    all_tests_passed = pct_passed == 100.0

    if all_tests_passed:
        dialog.display("Rule improvement successful. All test cases passed.")
    else:
        dialog.display(
            f"Rule improvement failed. Only {pct_passed:.1f}% of test cases passed."
        )

    return (
        all_tests_passed,
        response_parsed.rule,
        response_parsed.explanation,
        test_cases_combined,
    )


def display_test_cases(dialog: DialogProtocol, rule_data: dict, heading: str = ""):
    """Display test cases in a nicely formatted table.

    Parameters
    ----------
    dialog : DialogProtocol
        The dialog implementation
    rule_data : dict
        Dictionary containing rule information and test cases
    heading : str, optional
        Custom heading to display, by default empty string
    """
    if heading:
        dialog.display(heading)

    rows = []
    for i, test_case in enumerate(rule_data["test_cases"], 1):
        rows.append(
            {
                "#": str(i),
                "Text": test_case["text"],
                "Should Trigger Alert": (
                    "✓" if test_case["should_trigger_alert"] else "✗"
                ),
                "Explanation": test_case["explanation"],
            }
        )

    dialog.display_table(
        title="Test Cases",
        columns=["#", "Text", "Should Trigger Alert", "Explanation"],
        rows=rows,
    )


def display_rule(dialog: DialogProtocol, rule_data: dict):
    """Display the rule and test cases in a nicely formatted way.

    Parameters
    ----------
    dialog : DialogProtocol
        The dialog implementation
    rule_data : dict
        Dictionary containing rule information and test cases
    """

    if rule_data["explanation"]:
        dialog.display_markdown(
            create_template(
                """
# Rule Information

{{ explanation }}
"""
            )
            .render(explanation=rule_data["explanation"])
            .strip()
        )

    # Display test cases table
    # NOTE: not showing it because we show the one with the test results
    # display_test_cases(dialog, rule_data)


@cli.command()
@click.argument("rule_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def improve(rule_file, verbose):
    dialog = CLIDialog()
    rule_data = json.loads(Path(rule_file).read_text())
    improve_workflow(rule_data, dialog, verbose)


def improve_workflow(rule_data: dict, dialog: DialogProtocol, verbose: bool):
    """Improve a rule.

    Parameters
    ----------
    rule_data : dict
        Dictionary containing rule information and test cases
    dialog : DialogProtocol
        The dialog implementation
    verbose : bool
        Whether to display verbose output
    """
    done = False
    messages = []

    display_rule(dialog, rule_data)

    while not done:
        # TODO: limit the context passed to suggest_next_action
        next_action_suggested = suggest_next_action(
            dialog,
            rule_data,
            messages,
            verbose,
        ).text

        user_prompt = dialog.input(next_action_suggested)

        messages.append(WizardMessage(role="assistant", content=next_action_suggested))
        messages.append(WizardMessage(role="user", content=user_prompt))

        next_action = get_next_action(
            dialog,
            user_prompt,
            messages[-2:],
            verbose=verbose,
        )

        # modify test cases, keep the rule the same
        # i.e. create a better test suite for the existing rule
        if next_action == "EDIT_TEST_CASES":
            all_tests_passed, rule, explanation, test_cases = edit_test_cases(
                dialog, rule_data, user_prompt
            )
        # TODO: this should ask a few clarifying questions before modifying the rule
        # modify rule, modify test cases to match the new rule
        # i.e. make the rule more specific or more general
        elif next_action == "MODIFY_RULE":
            all_tests_passed, rule, explanation, test_cases = modify_rule(
                dialog, rule_data, user_prompt
            )
        # modify the rule, keep test cases the same
        # i.e. make the rule pass the new test cases
        elif next_action == "RUN_FIXER":
            # TODO: currently we only return an explanation if things go wrong,
            # but we should do it in both cases
            all_tests_passed, rule, explanation, test_cases = run_fixer(
                rule_data["rule"],
                rule_data["test_cases"],
            )

            if explanation:
                dialog.display("Improvement Failed: " + explanation)
            else:
                dialog.display("Improvement successful. All test cases passed.")
        else:
            # invalid action
            raise ValueError(f"Invalid action: {next_action}")

        # TODO: maybe add a deterministic action that shows the test cases
        # and ask the user to manually review them

        # if not all_tests_pssed, skip suggest_next_action and automatically suggest
        # running the fixer
        if all_tests_passed:
            done = True

    return rule, explanation, test_cases


def combine_test_cases(rule_data: dict, changes: RuleImprovement):
    test_cases_to_remove = [test_case.text for test_case in changes.test_cases_removed]
    test_cases_after_removal = []

    # TODO: we should ask the user to confirm add/remove
    for test_case in rule_data["test_cases"]:
        if test_case["text"] not in test_cases_to_remove:
            test_cases_after_removal.append(test_case)
        else:
            test_cases_to_remove.remove(test_case["text"])

    # Add both suggested and removed test cases
    test_cases_combined = test_cases_after_removal + [
        test_case.model_dump() for test_case in changes.test_cases_suggested
    ]

    return test_cases_combined


def edit_test_cases(
    dialog: DialogProtocol,
    rule_data: dict,
    user_prompt: str,
) -> tuple[bool, str, str, list[dict]]:
    changes = suggest_test_cases(dialog, rule_data, user_prompt)
    test_cases_combined = combine_test_cases(rule_data, changes)

    _, pct_passed, _, _ = run_tests(rule_data["rule"], test_cases_combined)

    all_tests_passed = pct_passed == 100.0

    if all_tests_passed:
        dialog.display(
            "Test cases added and removed successfully. All test cases passed."
        )

    else:
        dialog.display(
            f"Test cases improvement failed. Only {pct_passed:.1f}% of test cases passed."
        )

    return (
        all_tests_passed,
        rule_data["rule"],
        rule_data["explanation"],
        test_cases_combined,
    )


def suggest_next_action(
    dialog: DialogProtocol,
    rule_data: dict,
    messages: list[WizardMessage],
    verbose: bool,
):
    """Suggest the next action based on the history of messages.

    Parameters
    ----------
    rule_data : dict
        Dictionary containing rule information and test cases
    """
    rule = rule_data["rule"]
    test_cases = rule_data["test_cases"]
    explanation = rule_data["explanation"]
    results, pct_passed, _, _ = run_tests(rule, test_cases, dialog)

    system_prompt = create_template(
        """
{% from "macros.html" import rule_report %}
    
You're a helpful assistant embedded in a system that allows users to define
regex patterns (rules) to detect PII in text.

You'll be given the list of messages exchanged between the user and the system (most
recent at the end), a summary of the rule (and its test cases), and you should
suggest the next action to take.

{{ rule_report(rule, explanation, results, pct_passed) }}

# Response Format

You should respond with a JSON object that contains the following fields:

## action

The action to take. It can be one of the following:

### `RUN_FIXER`

Run the fixer to improve the rule. This is a good action if some of the test cases are
failing. The fixer modifies the regex pattern to make the tests pass.

### `EDIT_TEST_CASES`

Edit the test cases. This is a good action if the test cases are not covering all the
cases and you want to add more test cases. Or if there are some test cases that
we should remove (e.g., because they are redundant).

### `MODIFY_RULE`

Modify the rule. This is a good action if you think the rule is not properly defined
and might miss some important PII data. For example, if a user us detecting US phone
numbers but it's not accounting for the +1 prefix, you might suggest modifying the
rule to account for this.

### `UNKNOWN`

The action is not clear. You should ask the user to be more specific.

## text

The text to display to the user. This should be a short explanation of the action to
take.

### Example: `RUN_FIXER`

Some of the test cases are failing. Do you want to run the fixer to fix the rule?

### Example: `EDIT_TEST_CASES`

I detected that the following scenarios are not covered by the test cases:

- [ ] ...
- [ ] ...

Do you want to add these scenarios to the test cases?

### Example: `MODIFY_RULE`

I detected that the following scenarios are not covered by the rule:

- [ ] ...
- [ ] ...

Do you want to modify the rule to account for these scenarios?

### Example: `UNKNOWN`

Let me know what can I do for you! I can help you to:

1. Improve the rule to fix existing test cases
2. Add more test cases to cover more scenarios
3. Modify the rule to make it more specific or more general

---

For each of the above actions, you can see the `Rule summary` section to come up with
good examples. Ensure to list the three elements in the list so the user knows
what you can help with.
"""
    ).render(
        rule=rule,
        explanation=explanation,
        results=results,
        pct_passed=pct_passed,
    )

    if verbose:
        dialog.log(
            "suggest_next_action",
            is_markdown=True,
            system_prompt=system_prompt,
        )

    messages_all = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *[message.model_dump() for message in messages],
    ]

    response = openai.OpenAI().beta.chat.completions.parse(
        model=SETTINGS.MODEL_DEFAULT,
        messages=messages_all,
        response_format=NextActionResponse,
    )

    response_parsed = response.choices[0].message.parsed

    return response_parsed


if __name__ == "__main__":
    cli()
