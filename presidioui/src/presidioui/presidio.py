from typing import List

from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel

from presidioui.models import DetectionRule, SessionLocal


class Rule(BaseModel):
    name: str
    regex: str


class PresidioAnalyzer:
    def __init__(self, rules: List[Rule]):
        """Initialize analyzer with custom rules.

        Parameters
        ----------
        rules : List[Rule]
            List of Rule objects containing name and regex pattern for each rule
        """
        self.analyzer = AnalyzerEngine()
        self.analyzer.registry.recognizers.clear()  # Remove all built-in recognizers
        self.anonymizer = AnonymizerEngine()

        for rule in rules:
            pattern = Pattern(name=f"{rule.name}_pattern", regex=rule.regex, score=0.5)
            recognizer = PatternRecognizer(
                supported_entity=rule.name, patterns=[pattern]
            )
            self.analyzer.registry.add_recognizer(recognizer)

    def test_samples(self, samples: List[str]) -> List[List[str]]:
        """Test samples against defined rules.

        Parameters
        ----------
        samples : List[str]
            List of text samples to analyze

        Returns
        -------
        List[List[str]]
            List of triggered rule names for each sample
        """
        results = []
        for sample in samples:
            analysis = self.analyzer.analyze(text=sample, language="en")
            triggered_rules = [result.entity_type for result in analysis]
            results.append(triggered_rules)
        return results

    def anonymize(self, text: str) -> str:
        """Anonymize text using defined rules.

        Parameters
        ----------
        text : str
            Text to anonymize

        Returns
        -------
        str
            Anonymized text
        """
        results = self.analyzer.analyze(text=text, language="en")
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text


class SingleRulePresidioAnalyzer(PresidioAnalyzer):
    """Analyzer for testing a single rule.

    Parameters
    ----------
    rule : Rule
        Rule object containing name and regex pattern

    Example
    -------
    >>> from presidioui.presidio import Rule, SingleRulePresidioAnalyzer
    >>> analyzer = SingleRulePresidioAnalyzer(Rule(name="PHONE_NUMBER", regex=r"\d{3}-\d{3}-\d{4}"))
    >>> analyzer.test_samples(["My phone number is 123-456-7890", "My email is test@example.com"])
    [True, False]
    """

    def __init__(self, rule: Rule):
        super().__init__([rule])
        self.rule_name = rule.name

    def test_samples(self, samples: list[str]) -> list[bool]:
        """Test multiple samples against the rule.

        Parameters
        ----------
        samples : list[str]
            List of text samples to analyze

        Returns
        -------
        list[bool]
            List of bools indicating whether each sample triggered the rule
        """
        results = super().test_samples(samples)
        return [self.rule_name in triggered_rules for triggered_rules in results]

    def test_samples_full_report(self, samples: list[dict]) -> list[dict]:
        """Test multiple samples against the rule.

        Parameters
        ----------
        samples : list[dict]
            List of dictionaries containing 'text', 'should_trigger_alert',
            and 'explanation' keys

        Returns
        -------
        list[dict]
            List of dictionaries containing 'text', 'should_trigger_alert', 'actual',
            'passed', 'explanation' keys
        """
        triggered_results = self.test_samples([sample["text"] for sample in samples])

        return [
            {
                "text": sample["text"],
                "should_trigger_alert": sample["should_trigger_alert"],
                "actual": triggered,
                "passed": triggered == sample["should_trigger_alert"],
                "explanation": sample["explanation"],
            }
            for sample, triggered in zip(samples, triggered_results)
        ]


def get_recognizers(names: list[str] | None = None):
    """Get pattern recognizers from detection rules in database.

    Parameters
    ----------
    names : list[str] or None, optional
        If provided, only return recognizers for rules with these names.
        Rules not found in the database will be ignored.

    Returns
    -------
    list
        List of pattern recognizers
    """
    recognizers = []
    with SessionLocal() as session:
        query = session.query(DetectionRule)
        if names is not None:
            query = query.filter(DetectionRule.name.in_(names))
        rules = query.all()

        for rule in rules:
            print("Adding recognizer for", rule.name)
            pattern = Pattern(name=rule.name, regex=rule.pattern, score=0.5)
            recognizer = PatternRecognizer(
                supported_entity=rule.name.upper(), patterns=[pattern]
            )
            recognizers.append(recognizer)
    return recognizers


def list_recognizers() -> list[str]:
    """Get list of all detection rule names from database.

    Returns
    -------
    list[str]
        List of detection rule names
    """
    with SessionLocal() as session:
        rules = session.query(DetectionRule.name).all()
        return [rule[0] for rule in rules]


def anonymize_text(text: str, recognizers: list[str] | None = None) -> str:
    """Anonymize sensitive information in text using Presidio with custom patterns.

    Uses the specified recognizers to detect and anonymize sensitive information.
    If no recognizers are configured or found, returns the original text unchanged.

    Parameters
    ----------
    text : str
        The input text to anonymize
    recognizers : list[str]
        List of recognizer names to use for detection. Only recognizers that exist
        in the database will be used.

    Returns
    -------
    str
        The anonymized text with sensitive information redacted, or original text
        if no recognizers are configured or found

    See Also
    --------
    get_recognizers : Gets pattern recognizers from detection rules
    """
    if not recognizers:
        return text

    # Initialize analyzer with custom recognizers
    analyzer = AnalyzerEngine()
    analyzer.registry.recognizers.clear()

    for recognizer in get_recognizers(recognizers):
        analyzer.registry.add_recognizer(recognizer)

    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, entities=None, language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text
