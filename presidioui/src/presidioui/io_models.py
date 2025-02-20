from pydantic import BaseModel
from typing import Literal


class WizardMessage(BaseModel):
    """Message from the wizard."""

    role: Literal["system", "user", "assistant"]
    content: str


class WizardTestCase(BaseModel):
    """Test case for the wizard."""

    text: str
    explanation: str
    should_trigger_alert: bool


class WizardState(BaseModel):
    """State of the wizard."""

    is_done: bool
    rule: str | None
    test_cases: list[WizardTestCase] | None
    messages: list[WizardMessage]
    explanation: str | None
    name: str | None


class WizardResponse(BaseModel):
    """Response from the wizard."""

    reply_to_user: str
    is_done: bool
    rule: str | None
    explanation: str | None
    test_cases: list[WizardTestCase] | None
    name: str | None


class TestCaseCandidate(BaseModel):
    text: str
    explanation: str
    should_trigger_alert: bool


class TestCaseCandidateRemoved(TestCaseCandidate):
    text: str
    explanation: str


class RuleImprovement(BaseModel):
    explanation: str
    test_cases_suggested: list[TestCaseCandidate]
    test_cases_removed: list[TestCaseCandidateRemoved]


class RouterResponse(BaseModel):
    action: Literal[
        "EDIT_TEST_CASES",
        "MODIFY_RULE",
        "RUN_FIXER",
        "UNKNOWN",
    ]


class ModifyRuleResponse(BaseModel):
    explanation: str
    summary_of_changes: str
    rule: str
    test_cases_suggested: list[TestCaseCandidate]
    test_cases_removed: list[TestCaseCandidateRemoved]


class NextActionResponse(BaseModel):
    action: Literal[
        "RUN_FIXER",
        "EDIT_TEST_CASES",
        "MODIFY_RULE",
        "UNKNOWN",
    ]
    text: str
