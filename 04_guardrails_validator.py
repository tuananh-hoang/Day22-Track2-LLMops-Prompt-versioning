"""
Step 4 — Guardrails AI Validators
====================================
TASKS:
  1. PIIDetector: detect & redact email, phone, SSN, credit card
  2. JSONFormatter: parse and auto-repair malformed JSON
  3. Wrap each in a Guard, test with multiple cases
  4. Full demo: 6 PII cases + 5 JSON cases

DELIVERABLE: All test cases run; PII redacted; JSON repaired or rejected cleanly.

IMPORTANT: pass `on_fail` to the VALIDATOR constructor, NOT to Guard.use()
  WRONG: Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
  RIGHT: Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct
"""

import re
import json

# ── 1. Imports ───────────────────────────────────────────────────────────────
try:
    from guardrails import Guard
    from guardrails.validators import (
        Validator,
        register_validator,
        PassResult,
        FailResult,
    )
    # OnFailAction location varies across guardrails-ai versions
    try:
        from guardrails import OnFailAction
    except ImportError:
        try:
            from guardrails.validator_base import OnFailAction
        except ImportError:
            from guardrails.hub import OnFailAction

    GUARDRAILS_AVAILABLE = True
except ImportError as exc:
    print(f"⚠️  guardrails-ai not installed or import failed: {exc}")
    print("   Run: pip install guardrails-ai")
    GUARDRAILS_AVAILABLE = False


# ── 2. PII Detector Validator ─────────────────────────────────────────────────
if GUARDRAILS_AVAILABLE:
    @register_validator(name="custom/pii-detector", data_type="string")
    class PIIDetector(Validator):
        """
        Detects and redacts Personally Identifiable Information (PII) in text.

        Patterns detected (via regular expressions):
          EMAIL:       user@domain.tld
          PHONE:       (123) 456-7890 / 123-456-7890 / +1-800-555-0000
          SSN:         123-45-6789
          CREDIT_CARD: 1234 5678 9012 3456 (spaces or dashes)

        On detection:
          Each matched span is replaced with "[PII_TYPE_REDACTED]".
          Returns PassResult(value_override=redacted_text) so the pipeline
          continues with sanitised output instead of raising an error.

        On-fail behaviour:
          Controlled by on_fail parameter passed to the constructor.
          Typical usage: PIIDetector(on_fail=OnFailAction.FIX)
        """

        # Class-level compiled regex patterns for all four PII types
        PII_PATTERNS = {
            "EMAIL": re.compile(
                r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
            ),
            "PHONE": re.compile(
                r"\b(?:\+?1[\s.\-]?)?"
                r"(?:\(?\d{3}\)?[\s.\-]?)"
                r"\d{3}[\s.\-]\d{4}\b"
            ),
            "SSN": re.compile(
                r"\b\d{3}-\d{2}-\d{4}\b"
            ),
            "CREDIT_CARD": re.compile(
                r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
            ),
        }

        def validate(self, value: str, metadata: dict):
            """
            Scan value for all PII types; replace each match with a
            type-specific redaction token.

            Implementation:
              1. Iterate PII_PATTERNS in deterministic order.
              2. For each pattern find all non-overlapping matches.
              3. Replace each match string in the working copy of value.
              4. Accumulate found PII for logging.
              5. Return PassResult(value_override=...) regardless of whether
                 PII was found, so the sanitised (or unchanged) text flows
                 downstream without interrupting the pipeline.
            """
            redacted_text = value
            found_pii: list[tuple[str, str]] = []

            for pii_type, pattern in self.PII_PATTERNS.items():
                matches = pattern.findall(redacted_text)
                for match in matches:
                    # Replace each occurrence individually (handles duplicates)
                    redacted_text = redacted_text.replace(
                        match, f"[{pii_type}_REDACTED]", 1
                    )
                    found_pii.append((pii_type, match))

            if found_pii:
                types_found = [p[0] for p in found_pii]
                print(f"  ⚠️  Redacted {len(found_pii)} PII item(s): {types_found}")
                return PassResult(value_override=redacted_text)

            return PassResult(value_override=value)


# ── 3. JSON Formatter Validator ───────────────────────────────────────────────
if GUARDRAILS_AVAILABLE:
    @register_validator(name="custom/json-formatter", data_type="string")
    class JSONFormatter(Validator):
        """
        Validates that the output is well-formed JSON, and attempts to
        auto-repair common LLM formatting errors.

        Repair strategies (applied in order):
          1. Strip leading/trailing whitespace
          2. Remove markdown code fences (``` or ```json blocks)
          3. Replace single-quote delimiters with double quotes
          4. Remove trailing commas before } or ]

        Return contract:
          - Valid JSON (original or repaired) → PassResult(value_override=
              json.dumps(parsed, indent=2))
          - Unrecoverable → FailResult with a JSON error envelope as fix_value
        """

        @staticmethod
        def _repair(text: str) -> str:
            """
            Apply heuristic repairs to a potentially malformed JSON string.

            Steps executed:
              1. Strip outer whitespace.
              2. Remove ```json ... ``` or ``` ... ``` markdown fences.
              3. Replace single-quote string delimiters with double quotes.
              4. Remove trailing commas before a closing brace or bracket.
            """
            text = text.strip()

            # Remove markdown code fences (```json...``` or ```...```)
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?\s*```$",           "", text)
            text = text.strip()

            # Replace single-quote delimiters with double quotes
            # NOTE: this is a heuristic; it handles simple cases but cannot
            # correctly handle apostrophes inside string values.
            text = text.replace("'", '"')

            # Remove trailing commas before } or ] (common LLM artefact)
            text = re.sub(r",\s*([}\]])", r"\1", text)

            return text

        def validate(self, value: str, metadata: dict):
            """
            Try json.loads(value) → if it fails, apply _repair() and retry.

            Successful parse flow:
              Re-serialise with json.dumps(parsed, indent=2) for consistent
              formatting and return PassResult(value_override=formatted).

            Failed parse flow:
              Return FailResult with a JSON error envelope so downstream
              consumers always receive a parseable fallback response.
            """
            # Pass 1: try parsing as-is
            try:
                parsed   = json.loads(value)
                repaired = json.dumps(parsed, indent=2)
                return PassResult(value_override=repaired)
            except json.JSONDecodeError:
                pass

            # Pass 2: apply heuristic repairs and retry
            try:
                repaired_text = self._repair(value)
                parsed        = json.loads(repaired_text)
                repaired      = json.dumps(parsed, indent=2)
                print("  🔧 JSON repaired successfully")
                return PassResult(value_override=repaired)
            except json.JSONDecodeError as exc:
                error_payload = json.dumps(
                    {"error": f"Invalid JSON after repair attempt: {exc}",
                     "raw": value[:200]},
                    indent=2,
                )
                return FailResult(
                    error_message=f"Unrecoverable JSON: {exc}",
                    fix_value=error_payload,
                )


# ── 4. PII Guard demo ────────────────────────────────────────────────────────
def demo_pii_guard() -> None:
    """
    Instantiate a Guard with PIIDetector and run 6 test cases:
      1. Email address
      2. US phone number
      3. Social Security Number
      4. Credit card number
      5. Multiple PII types in one string
      6. Clean text (no PII — verify no false positives)
    """
    if not GUARDRAILS_AVAILABLE:
        print("⚠️  Skipping PII demo — guardrails-ai not available")
        return

    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567, "
                         "SSN: 987-65-4321"),
        ("Clean",       "No sensitive information in this text at all."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ PASS" if result.validation_passed else "❌ FAIL"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 5. JSON Guard demo ────────────────────────────────────────────────────────
def demo_json_guard() -> None:
    """
    Instantiate a Guard with JSONFormatter and run 5 test cases:
      1. Valid JSON                  → should pass unchanged (reformatted)
      2. Markdown-fenced JSON        → strip fences → pass
      3. Single-quoted JSON          → convert quotes → pass
      4. Trailing comma              → remove comma → pass
      5. Truly invalid JSON          → all repairs fail → FailResult + fallback
    """
    if not GUARDRAILS_AVAILABLE:
        print("⚠️  Skipping JSON demo — guardrails-ai not available")
        return

    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",
         '{"name": "Alice", "age": 30, "role": "engineer"}'),

        ("Markdown fences",
         '```json\n{"model": "gpt-4", "temperature": 0.7}\n```'),

        ("Single quotes",
         "{'name': 'Charlie', 'score': 95, 'passed': True}"),

        ("Trailing comma",
         '{"key": "value", "items": [1, 2, 3,],}'),

        ("Truly invalid",
         "This is not JSON at all: ??? {] broken [}"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ PASS" if result.validation_passed else "❌ FAIL"
        output_preview = str(result.validated_output or "")[:80].replace("\n", "\\n")
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:70]}")
        print(f"  Output: {output_preview}")


# ── 6. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n" + "=" * 55)
    print("✅ Step 4 complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()