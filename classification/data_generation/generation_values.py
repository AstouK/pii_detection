"""Static values and placement policies for synthetic document generation."""

FAKER_LOCALES = {
    "en": "en_US",
    "de": "de_DE",
}


GERMAN_FIELD_LABELS = {
    "employee": "Mitarbeiter",
    "name": "Name",
    "department": "Abteilung",
    "date": "Datum",
    "category": "Kategorie",
    "amount": "Betrag",
    "description": "Beschreibung",
    "summary": "Zusammenfassung",
    "manager": "Vorgesetzter",
    "decision": "Entscheidung",
    "system": "System",
    "access_level": "Zugriffsstufe",
    "justification": "Begründung",
    "reviewer": "Prüfer",
    "comments": "Kommentare",
    "approval": "Genehmigung",
    "approver": "Genehmigende Person",
    "signature": "Unterschrift",
    "location": "Ort",
    "type": "Typ",
    "root_cause": "Ursache",
    "corrective_action": "Maßnahme",
    "owner": "Verantwortlich",
    "deadline": "Frist",
    "company": "Unternehmen",
    "address": "Adresse",
    "contact": "Kontakt",
    "tax_id": "Steuernummer",
    "participant": "Teilnehmer",
    "course": "Kurs",
    "trainer": "Trainer",
    "recommendation": "Empfehlung",
    "role": "Rolle",
    "employee_id": "Mitarbeiternummer",
    "start_date": "Eintrittsdatum",
    "supplier": "Lieferant",
    "customer": "Kunde",
    "invoice_number": "Rechnungsnummer",
    "invoice_date": "Rechnungsdatum",
    "billing_address": "Rechnungsadresse",
    "payment_details": "Zahlungsdaten",
    "parties": "Vertragsparteien",
    "effective_date": "Gültig ab",
    "subject": "Betreff",
    "terms": "Bedingungen",
    "termination": "Kündigung",
    "ticket_id": "Ticketnummer",
    "agent": "Bearbeiter",
    "status": "Status",
    "resolution": "Lösung",
    "patient": "Patient",
    "provider": "Behandler",
    "reason": "Grund",
    "assessment": "Beurteilung",
    "treatment": "Behandlung",
    "follow_up": "Nachsorge",
    "holder": "Inhaber",
    "passport_number": "Passnummer",
    "nationality": "Nationalität",
    "date_of_birth": "Geburtsdatum",
    "issue_date": "Ausstellungsdatum",
    "expiry_date": "Ablaufdatum",
    "verification_status": "Prüfstatus",
    "title": "Titel",
    "author": "Autor",
    "content": "Inhalt",
    "reference": "Referenz",
    "sender": "Absender",
    "recipient": "Empfänger",
    "body": "Nachricht",
    "participants": "Teilnehmer",
    "agenda": "Agenda",
    "discussion": "Diskussion",
    "decisions": "Entscheidungen",
    "action_items": "Aufgaben",
    "notes": "Notizen",
    "cost_center": "Kostenstelle",
    "project": "Projekt",
    "destination": "Reiseziel",
    "receipt_reference": "Belegreferenz",
    "payment_method": "Zahlungsart",
}


GERMAN_TITLES = {
    "expense_report": "Reisekostenabrechnung",
    "it_access_request": "IT-Zugriffsantrag",
    "incident_report": "Vorfallsbericht",
    "supplier_onboarding": "Lieferantenaufnahme",
    "training_evaluation": "Schulungsbewertung",
    "employee_record": "Mitarbeiterakte",
    "invoice": "Rechnung",
    "contract": "Vertrag",
    "customer_support": "Kundensupport",
    "medical_record": "Medizinische Dokumentation",
    "passport_record": "Identitätsprüfung",
    "general_document": "Internes Dokument",
    "internal_email": "Interne E-Mail",
    "meeting_notes": "Besprechungsnotizen",
}


ENTITY_FIELD_CANDIDATES = {
    "PERSON": [
        "employee",
        "name",
        "participant",
        "manager",
        "owner",
        "contact",
        "customer",
        "patient",
        "holder",
        "sender",
        "recipient",
        "author",
        "participants",
        "parties",
        "agent",
        "reviewer",
        "approver",
        "trainer",
        "provider",
    ],
    "EMAIL_ADDRESS": [
        "contact",
        "sender",
        "recipient",
        "comments",
        "notes",
        "body",
        "description",
        "summary",
    ],
    "PHONE_NUMBER": [
        "contact",
        "comments",
        "notes",
        "body",
        "description",
        "summary",
    ],
    "LOCATION": [
        "location",
        "address",
        "billing_address",
        "contact",
        "terms",
        "notes",
    ],
    "DATE_TIME": [
        "date",
        "deadline",
        "start_date",
        "invoice_date",
        "effective_date",
        "date_of_birth",
        "issue_date",
        "expiry_date",
        "notes",
    ],
    "IBAN_CODE": [
        "payment_details",
        "notes",
        "summary",
        "description",
    ],
    "CREDIT_CARD": [
        "payment_details",
        "notes",
        "description",
    ],
    "PASSPORT": [
        "passport_number",
        "reference",
        "notes",
    ],
    "NRP": [
        "nationality",
        "notes",
    ],
    "IP_ADDRESS": [
        "description",
        "comments",
        "notes",
        "body",
    ],
    "URL": [
        "description",
        "comments",
        "notes",
        "body",
    ],
    "MEDICAL_LICENSE": [
        "provider",
        "notes",
    ],
}


SCENARIO_ENTITY_FIELD_PREFERENCES = {
    "contract": {
        "PERSON": ["parties", "contact", "signature"],
        "EMAIL_ADDRESS": ["contact"],
        "LOCATION": ["parties", "contact"],
        "DATE_TIME": ["effective_date", "terms"],
    },

    "customer_support": {
        "PERSON": ["customer", "contact", "agent"],
        "EMAIL_ADDRESS": ["contact"],
        "PHONE_NUMBER": ["contact"],
        "IP_ADDRESS": ["description", "resolution"],
        "URL": ["description", "resolution"],
    },

    "employee_record": {
        "PERSON": ["employee", "manager", "notes"],
        "EMAIL_ADDRESS": ["contact", "notes"],
        "PHONE_NUMBER": ["contact", "notes"],
        "LOCATION": ["address", "notes"],
        "NRP": ["nationality", "notes"],
        "DATE_TIME": ["start_date", "notes"],
    },

    "expense_report": {
        "PERSON": ["employee", "description", "summary"],
        "EMAIL_ADDRESS": ["contact", "description", "summary"],
        "IBAN_CODE": ["payment_details", "summary", "description"],
        "PHONE_NUMBER": ["contact", "description", "summary"],
    },

    "general_document": {
        "PERSON": ["author", "content", "notes"],
        "EMAIL_ADDRESS": ["content", "notes"],
        "PHONE_NUMBER": ["content", "notes"],
        "LOCATION": ["content", "notes"],
        "DATE_TIME": ["date", "content"],
        "URL": ["content", "notes"],
    },

    "incident_report": {
        "PERSON": ["owner", "description"],
        "LOCATION": ["location"],
        "DATE_TIME": ["date", "deadline"],
    },

    "internal_email": {
        "PERSON": ["sender", "recipient", "signature"],
        "EMAIL_ADDRESS": ["sender", "recipient"],
        "PHONE_NUMBER": ["body", "signature"],
        "URL": ["body"],
    },

    "invoice": {
        "PERSON": ["customer", "contact"],
        "EMAIL_ADDRESS": ["contact"],
        "IBAN_CODE": ["payment_details"],
        "CREDIT_CARD": ["payment_details"],
        "LOCATION": ["billing_address"],
    },

    "it_access_request": {
        "PERSON": ["name", "manager", "comments"],
        "EMAIL_ADDRESS": ["contact", "comments", "justification"],
        "IP_ADDRESS": ["comments", "justification"],
        "URL": ["comments", "justification"],
    },

    "medical_record": {
        "PERSON": ["patient", "provider", "notes"],
        "DATE_TIME": ["date", "follow_up", "notes"],
        "MEDICAL_LICENSE": ["provider", "notes"],
        "LOCATION": ["location", "notes"],
    },

    "meeting_notes": {
        "PERSON": ["participants", "contact", "discussion"],
        "DATE_TIME": ["date", "agenda", "discussion"],
        "EMAIL_ADDRESS": ["contact", "action_items"],
        "LOCATION": ["location", "discussion", "agenda"],
    },

    "passport_record": {
        "PERSON": ["holder"],
        "PASSPORT": ["passport_number"],
        "NRP": ["nationality"],
        "DATE_TIME": ["date_of_birth", "issue_date", "expiry_date"],
    },
   
    "supplier_onboarding": {
        "PERSON": ["contact", "reviewer", "comments"],
        "EMAIL_ADDRESS": ["contact", "comments"],
        "PHONE_NUMBER": ["contact", "comments"],
        "LOCATION": ["address", "notes"],
        "IBAN_CODE": ["payment_details", "notes", "comments"],
    },

    "training_evaluation": {
        "PERSON": ["participant", "trainer", "comments"],
        "EMAIL_ADDRESS": ["contact", "comments"],
        "DATE_TIME": ["date", "comments"],
    },
}

 # ---------------------------------------------------------
# Variant-specific deterministic values
#
# These keep the structural backbone identical while giving
# document variants distinct content profiles even when the
# LLM is disabled or falls back.
# ---------------------------------------------------------

VARIANT_VALUES = {
        "medical_record": {
            "clinical_note": {
                "reason": {
                    "en": [
                        "Routine clinical assessment",
                        "Follow-up consultation",
                        "Evaluation of current symptoms",
                    ],
                    "de": [
                        "Routinemäßige klinische Untersuchung",
                        "Nachuntersuchung",
                        "Beurteilung aktueller Beschwerden",
                    ],
                },
                "assessment": {
                    "en": [
                        "Clinical findings reviewed and documented.",
                        "Current condition assessed during consultation.",
                    ],
                    "de": [
                        "Klinische Befunde wurden geprüft und dokumentiert.",
                        "Der aktuelle Zustand wurde im Rahmen der Untersuchung beurteilt.",
                    ],
                },
                "treatment": {
                    "en": [
                        "Conservative management recommended.",
                        "Treatment plan discussed and documented.",
                    ],
                    "de": [
                        "Eine konservative Behandlung wurde empfohlen.",
                        "Der Behandlungsplan wurde besprochen und dokumentiert.",
                    ],
                },
                "follow_up": {
                    "en": [
                        "Follow-up as clinically indicated",
                        "Review recommended if symptoms persist",
                    ],
                    "de": [
                        "Nachsorge nach klinischer Erforderlichkeit",
                        "Erneute Vorstellung bei anhaltenden Beschwerden empfohlen",
                    ],
                },
            },

            "appointment_summary": {
                "reason": {
                    "en": [
                        "Scheduled follow-up appointment",
                        "Routine appointment review",
                        "Consultation summary",
                    ],
                    "de": [
                        "Geplanter Nachsorgetermin",
                        "Routinemäßiger Kontrolltermin",
                        "Zusammenfassung des Beratungstermins",
                    ],
                },
                "assessment": {
                    "en": [
                        "Key findings from the appointment were reviewed.",
                        "Current status was discussed during the appointment.",
                    ],
                    "de": [
                        "Die wesentlichen Ergebnisse des Termins wurden besprochen.",
                        "Der aktuelle Status wurde während des Termins erörtert.",
                    ],
                },
                "treatment": {
                    "en": [
                        "Existing recommendations remain in place.",
                        "No change to the current management plan.",
                    ],
                    "de": [
                        "Die bisherigen Empfehlungen bleiben bestehen.",
                        "Keine Änderung des aktuellen Behandlungsplans.",
                    ],
                },
                "follow_up": {
                    "en": [
                        "Routine follow-up recommended",
                        "Next review according to the standard schedule",
                    ],
                    "de": [
                        "Routinemäßige Nachkontrolle empfohlen",
                        "Nächste Kontrolle gemäß regulärem Zeitplan",
                    ],
                },
            },

            "occupational_health_note": {
                "reason": {
                    "en": [
                        "Occupational health assessment",
                        "Workplace-related health review",
                        "Fitness-for-work consultation",
                    ],
                    "de": [
                        "Arbeitsmedizinische Untersuchung",
                        "Arbeitsplatzbezogene Gesundheitsbeurteilung",
                        "Beratung zur Arbeitsfähigkeit",
                    ],
                },
                "assessment": {
                    "en": [
                        "Work-related health factors were reviewed.",
                        "Occupational health considerations were assessed.",
                    ],
                    "de": [
                        "Arbeitsbezogene Gesundheitsfaktoren wurden geprüft.",
                        "Arbeitsmedizinische Aspekte wurden beurteilt.",
                    ],
                },
                "treatment": {
                    "en": [
                        "Workplace recommendations were discussed.",
                        "No specific clinical treatment was required.",
                    ],
                    "de": [
                        "Arbeitsplatzbezogene Empfehlungen wurden besprochen.",
                        "Eine spezifische medizinische Behandlung war nicht erforderlich.",
                    ],
                },
                "follow_up": {
                    "en": [
                        "Occupational health review if required",
                        "Follow-up according to workplace health requirements",
                    ],
                    "de": [
                        "Arbeitsmedizinische Nachkontrolle bei Bedarf",
                        "Nachkontrolle entsprechend den arbeitsmedizinischen Anforderungen",
                    ],
                },
            },
        },
}


# ---------------------------------------------------------
    # Scenario-specific structured values
    # ---------------------------------------------------------


SCENARIO_VALUES = {
        "expense_report": {
            "department": [
                "Finance",
                "Procurement",
                "Operations",
                "Sales",
                "IT",
            ],
            "cost_center": [
                "CC-1100",
                "CC-2400",
                "CC-3150",
                "CC-4200",
                "CC-5800",
            ],
            "project": [
                "Client Delivery",
                "Supplier Workshop",
                "Internal Transformation",
                "Regional Operations",
                "Process Improvement",
            ],
            "destination": [
                "Regional Office",
                "Client Site",
                "Training Center",
                "Supplier Location",
                "Head Office",
            ],
            "payment_method": [
                "Corporate Card",
                "Personal Card",
                "Cash",
                "Reimbursement",
            ],
            "category": [
                "Travel",
                "Accommodation",
                "Meals",
                "Transport",
                "Office Supplies",
                "Training",
            ],
            "manager": [
                "Department Manager",
                "Team Lead",
                "Finance Manager",
                "Operations Manager",
            ],
            "decision": [
                "Approved",
                "Approved with adjustment",
                "Pending review",
                "Rejected",
            ],
        },

        "it_access_request": {
            "department": [
                "Finance",
                "Operations",
                "IT",
                "Procurement",
                "Human Resources",
            ],
            "system": [
                "SAP S/4HANA",
                "ServiceNow",
                "Microsoft 365",
                "CRM Platform",
                "Analytics Workspace",
            ],
            "access_level": [
                "Read Only",
                "Standard User",
                "Editor",
                "Power User",
                "Administrator",
            ],
            "approval": [
                "Approved",
                "Pending",
                "Rejected",
            ],
            "reviewer": [
                "IT Security Team",
                "Access Management",
                "System Administration",
            ],
            "approver": [
                "Department Manager",
                "IT Manager",
                "Access Control Team",
            ],
        },

        "incident_report": {
            "type": [
                "IT Security Incident",
                "Operational Incident",
                "Equipment Failure",
                "Process Deviation",
                "Service Disruption",
            ],
            "owner": [
                "IT Operations",
                "Facility Management",
                "Security Team",
                "Operations Management",
            ],
        },

        "supplier_onboarding": {
            "certification": [
                "ISO 9001",
                "ISO 27001",
                "Environmental Compliance",
                "Supplier Code of Conduct",
            ],
            "risk_level": [
                "Low",
                "Medium",
                "High",
            ],
            "reviewer": [
                "Procurement Team",
                "Compliance Team",
                "Supplier Management",
            ],
            "approval": [
                "Approved",
                "Pending Review",
                "Rejected",
            ],
        },

        "training_evaluation": {
            "course": [
                "Data Protection Essentials",
                "Project Management Fundamentals",
                "Cybersecurity Awareness",
                "Leadership Development",
                "Process Improvement Workshop",
            ],
            "trainer": [
                "Internal Training Team",
                "Learning and Development",
                "External Training Provider",
            ],
            "material": [
                "Very Good",
                "Good",
                "Satisfactory",
                "Needs Improvement",
            ],
        },
        "internal_email": {
            "subject": [
                "Project Timeline Update",
                "Upcoming System Maintenance",
                "Quarterly Planning Follow-Up",
                "Process Change Notification",
                "Workshop Preparation",
                "Approval Status Update",
                "Team Schedule Coordination",
                "Documentation Review",
            ],
        },

        "general_document": {
            "title": [
                "Project Status Update",
                "Operational Process Note",
                "Internal Policy Update",
                "Quarterly Planning Memo",
                "Implementation Progress Report",
                "Process Improvement Note",
                "Department Update",
                "Internal Review Summary",
            ],
        },
    }


# ---------------------------------------------------------
# Safe structured fallback values
#
# Used for populated non-PII documents where a field should
# contain something realistic but must not introduce personal
# data or identifiers.
# ---------------------------------------------------------

GENERIC_STRUCTURED_VALUES = {
        "employee": {
            "en": [
                "Employee details not provided",
                "Department submission",
                "Internal submission",
            ],
            "de": [
                "Mitarbeiterangaben nicht angegeben",
                "Abteilungseinreichung",
                "Interne Einreichung",
            ],
        },
        "participant": {
            "en": [
                "Participant details not provided",
                "Anonymous participant",
                "Evaluation respondent",
            ],
            "de": [
                "Teilnehmerangaben nicht angegeben",
                "Anonyme teilnehmende Person",
                "Bewertungsteilnehmende Person",
            ],
        },
        "name": {
            "en": [
                "Requester details not provided",
                "Department request",
            ],
            "de": [
                "Antragstellerangaben nicht angegeben",
                "Abteilungsantrag",
            ],
        },
        "manager": {
            "en": [
                "Department Manager",
                "Team Lead",
                "Responsible Manager",
            ],
            "de": [
                "Abteilungsleitung",
                "Teamleitung",
                "Verantwortliche Führungskraft",
            ],
        },
        "customer": {
            "en": [
                "Corporate Customer",
                "Customer Organization",
                "Business Client",
            ],
            "de": [
                "Geschäftskunde",
                "Kundenorganisation",
                "Firmenkunde",
            ],
        },
        "contact": {
            "en": [
                "Department Contact",
                "Service Team",
                "Central Contact Point",
            ],
            "de": [
                "Abteilungskontakt",
                "Serviceteam",
                "Zentrale Kontaktstelle",
            ],
        },
        "agent": {
            "en": [
                "Customer Support Team",
                "Support Specialist",
                "Service Desk",
            ],
            "de": [
                "Kundensupport-Team",
                "Support-Fachbereich",
                "Service Desk",
            ],
        },
        "parties": {
            "en": [
                "Contracting Organizations",
                "Service Provider and Client Organization",
            ],
            "de": [
                "Vertragsparteien",
                "Dienstleister und Kundenorganisation",
            ],
        },
        "signature": {
            "en": [
                "Authorized signature",
                "Signature pending",
            ],
            "de": [
                "Autorisierte Unterschrift",
                "Unterschrift ausstehend",
            ],
        },
        "provider": {
            "en": [
                "Medical Team",
                "Occupational Health Service",
                "Clinical Department",
            ],
            "de": [
                "Medizinisches Team",
                "Betriebsärztlicher Dienst",
                "Medizinische Fachabteilung",
            ],
        },
                "patient": {
            "en": [
                "Patient details not provided",
                "Patient information withheld",
            ],
            "de": [
                "Patientenangaben nicht angegeben",
                "Patienteninformationen zurückgehalten",
            ],
        },
        "nationality": {
            "en": [
                "Nationality not provided",
                "Nationality information unavailable",
            ],
            "de": [
                "Nationalität nicht angegeben",
                "Angabe zur Nationalität nicht verfügbar",
            ],
        },
        "holder": {
            "en": [
                "Holder details not provided",
                "Holder information unavailable",
            ],
            "de": [
                "Inhaberangaben nicht angegeben",
                "Inhaberinformationen nicht verfügbar",
            ],
        },
        "passport_number": {
            "en": [
                "Document number not provided",
                "Document number unavailable",
            ],
            "de": [
                "Dokumentennummer nicht angegeben",
                "Dokumentennummer nicht verfügbar",
            ],
        },
        "date_of_birth": {
            "en": [
                "Date of birth not provided",
                "Date of birth unavailable",
            ],
            "de": [
                "Geburtsdatum nicht angegeben",
                "Geburtsdatum nicht verfügbar",
            ],
        },
        "participants": {
            "en": [
                "Project Team",
                "Department Representatives",
                "Working Group",
            ],
            "de": [
                "Projektteam",
                "Abteilungsvertretungen",
                "Arbeitsgruppe",
            ],
        },
        "author": {
            "en": [
                "Corporate Communications",
                "Project Office",
                "Compliance Department",
            ],
            "de": [
                "Unternehmenskommunikation",
                "Projektbüro",
                "Compliance-Abteilung",
            ],
        },
        "sender": {
            "en": [
                "Project Office",
                "Operations Team",
                "Internal Communications",
            ],
            "de": [
                "Projektbüro",
                "Operations-Team",
                "Interne Kommunikation",
            ],
        },
        "recipient": {
            "en": [
                "Project Team",
                "Department Staff",
                "Relevant Stakeholders",
            ],
            "de": [
                "Projektteam",
                "Abteilungsmitarbeitende",
                "Relevante Beteiligte",
            ],
        },
        "company": {
            "en": [
                "Supplier Organization",
                "Business Partner",
                "Vendor Organization",
            ],
            "de": [
                "Lieferantenorganisation",
                "Geschäftspartner",
                "Anbieterorganisation",
            ],
        },
        "supplier": {
            "en": [
                "Supplier Organization",
                "Service Provider",
                "Vendor",
            ],
            "de": [
                "Lieferantenorganisation",
                "Dienstleister",
                "Anbieter",
            ],
        },
        "address": {
            "en": [
                "Business address not provided",
                "Registered office information pending",
            ],
            "de": [
                "Geschäftsadresse nicht angegeben",
                "Angaben zum Geschäftssitz ausstehend",
            ],
        },
        "billing_address": {
            "en": [
                "Corporate billing address",
                "Billing details on file",
            ],
            "de": [
                "Geschäftliche Rechnungsadresse",
                "Rechnungsdaten hinterlegt",
            ],
        },
        "location": {
            "en": [
                "Main Office",
                "Operations Area",
                "IT Department",
                "Warehouse Area",
            ],
            "de": [
                "Hauptstandort",
                "Betriebsbereich",
                "IT-Abteilung",
                "Lagerbereich",
            ],
        },
        "verification_status": {
            "en": [
                "Pending",
                "Verified",
                "Review Required",
            ],
            "de": [
                "Ausstehend",
                "Geprüft",
                "Prüfung erforderlich",
            ],
        },
        "payment_details": {
            "en": [
                "Payment terms on file",
                "Standard corporate payment process",
            ],
            "de": [
                "Zahlungsbedingungen hinterlegt",
                "Standardmäßiger geschäftlicher Zahlungsprozess",
            ],
        },
}

DATE_FIELDS = {
        "date",
        "deadline",
        "start_date",
        "invoice_date",
        "effective_date",
        "issue_date",
        "expiry_date",
    }

REFERENCE_FIELDS = {
    "invoice_number",
    "ticket_id",
    "employee_id",
    "reference",
    "tax_id",
}

FREE_TEXT_PLACEHOLDER_FIELDS = {
    "description",
    "summary",
    "comments",
    "notes",
    "content",
    "body",
    "discussion",
    "terms",
    "assessment",
    "treatment",
    "resolution",
    "justification",
    "reason",
    "recommendation",
    "action_items",
}

GENERIC_FIELD_VALUES = {
    "department": [
        "Finance",
        "Operations",
        "IT",
        "Procurement",
        "Administration",
    ],
    "role": [
        "Specialist",
        "Analyst",
        "Coordinator",
        "Manager",
        "Administrator",
    ],
    "status": [
        "Open",
        "In Progress",
        "Pending",
        "Completed",
        "Closed",
    ],
}