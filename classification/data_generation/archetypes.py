"""Document archetypes used by the synthetic data generator."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Archetype:
    """Define the fields and variants of one document scenario."""

    name: str
    document_type: str
    fields: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)


EXPENSE_REPORT = Archetype(
    name="expense_report",
    document_type="form",
    fields=[
        "employee",
        "contact",
        "department",
        "date",
        "category",
        "amount",
        "cost_center",
        "project",
        "destination",
        "receipt_reference",
        "payment_method",
        "payment_details",
        "description",
        "summary",
        "manager",
        "decision",
    ],
    variants=[
        "blank",
        "filled_a",
        "filled_b",
    ],
)


IT_ACCESS_REQUEST = Archetype(
    name="it_access_request",
    document_type="form",
    fields=[
        "name",
        "contact",
        "department",
        "manager",
        "system",
        "access_level",
        "justification",
        "reviewer",
        "comments",
        "approval",
        "approver",
        "signature",
        "date",
    ],
    variants=[
        "blank",
        "filled_a",
        "filled_b",
    ],
)


INCIDENT_REPORT = Archetype(
    name="incident_report",
    document_type="report",
    fields=[
        "date",
        "location",
        "type",
        "description",
        "root_cause",
        "corrective_action",
        "owner",
        "deadline",
    ],
    variants=[
        "blank",
        "filled_a",
        "filled_b",
    ],
)


SUPPLIER_ONBOARDING = Archetype(
    name="supplier_onboarding",
    document_type="form",
    fields=[
        "company",
        "address",
        "contact",
        "tax_id",
        "certification",
        "risk_level",
        "payment_details",
        "reviewer",
        "comments",
        "approval",
        "notes",
        "nationality",
        "identification_number",
    ],
    variants=[
        "blank",
        "filled_a",
        "filled_b",
    ],
)


TRAINING_EVALUATION = Archetype(
    name="training_evaluation",
    document_type="form",
    fields=[
        "participant",
        "contact",
        "course",
        "date",
        "content",
        "trainer",
        "material",
        "comments",
        "recommendation",
    ],
    variants=[
        "blank",
        "filled_a",
        "filled_b",
    ],
)


EMPLOYEE_RECORD = Archetype(
    name="employee_record",
    document_type="hr_record",
    fields=[
        "employee",
        "department",
        "role",
        "employee_id",
        "manager",
        "start_date",
        "address",
        "nationality",
        "contact",
        "notes",
        "license_number",
        "identification_number",
    ],
    variants=[
        "structured_record",
        "hr_profile",
        "employee_summary",
    ],
)


INVOICE = Archetype(
    name="invoice",
    document_type="invoice",
    fields=[
        "supplier",
        "customer",
        "contact",
        "invoice_number",
        "invoice_date",
        "billing_address",
        "line_items",
        "amount",
        "tax_id",
        "payment_details",
    ],
    variants=[
        "standard_invoice",
        "service_invoice",
        "invoice_summary",
    ],
)


CONTRACT = Archetype(
    name="contract",
    document_type="contract",
    fields=[
        "parties",
        "effective_date",
        "subject",
        "terms",
        "contact",
        "signature",
        "termination",
        "payment_details",
    ],
    variants=[
        "contract_excerpt",
        "signature_section",
        "amendment",
    ],
)


CUSTOMER_SUPPORT = Archetype(
    name="customer_support",
    document_type="support_ticket",
    fields=[
        "ticket_id",
        "subject",
        "customer",
        "contact",
        "description",
        "agent",
        "status",
        "resolution",
    ],
    variants=[
        "portal_ticket",
        "email_ticket",
        "escalation_note",
    ],
)


MEDICAL_RECORD = Archetype(
    name="medical_record",
    document_type="medical_record",
    fields=[
        "patient",
        "date",
        "provider",
        "location",
        "reason",
        "assessment",
        "treatment",
        "follow_up",
        "notes",
    ],
    variants=[
        "clinical_note",
        "appointment_summary",
        "occupational_health_note",
    ],
)


PASSPORT_RECORD = Archetype(
    name="passport_record",
    document_type="identity_record",
    fields=[
        "holder",
        "passport_number",
        "nationality",
        "date_of_birth",
        "issue_date",
        "expiry_date",
        "verification_status",
    ],
    variants=[
        "identity_check",
        "travel_document_record",
        "verification_note",
    ],
)


GENERAL_DOCUMENT = Archetype(
    name="general_document",
    document_type="general_document",
    fields=[
        "title",
        "date",
        "author",
        "content",
        "reference",
        "notes",
    ],
    variants=[
        "internal_memo",
        "project_update",
        "policy_note",
    ],
)


INTERNAL_EMAIL = Archetype(
    name="internal_email",
    document_type="email",
    fields=[
        "sender",
        "recipient",
        "subject",
        "date",
        "body",
        "signature",
    ],
    variants=[
        "short_email",
        "thread_reply",
        "internal_announcement",
    ],
)


MEETING_NOTES = Archetype(
    name="meeting_notes",
    document_type="meeting_notes",
    fields=[
        "title",
        "date",
        "location",
        "participants",
        "contact",
        "agenda",
        "discussion",
        "decisions",
        "action_items",
    ],
    variants=[
        "formal_minutes",
        "informal_notes",
        "decision_log",
    ],
)


ARCHETYPES = {
    EXPENSE_REPORT.name: EXPENSE_REPORT,
    IT_ACCESS_REQUEST.name: IT_ACCESS_REQUEST,
    INCIDENT_REPORT.name: INCIDENT_REPORT,
    SUPPLIER_ONBOARDING.name: SUPPLIER_ONBOARDING,
    TRAINING_EVALUATION.name: TRAINING_EVALUATION,
    EMPLOYEE_RECORD.name: EMPLOYEE_RECORD,
    INVOICE.name: INVOICE,
    CONTRACT.name: CONTRACT,
    CUSTOMER_SUPPORT.name: CUSTOMER_SUPPORT,
    MEDICAL_RECORD.name: MEDICAL_RECORD,
    PASSPORT_RECORD.name: PASSPORT_RECORD,
    GENERAL_DOCUMENT.name: GENERAL_DOCUMENT,
    INTERNAL_EMAIL.name: INTERNAL_EMAIL,
    MEETING_NOTES.name: MEETING_NOTES,
}


def get_archetype(name: str) -> Archetype:
    """Return the archetype for a supported scenario."""

    try:
        return ARCHETYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"No archetype registered for scenario: {name}"
        ) from exc