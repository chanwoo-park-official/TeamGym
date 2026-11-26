"""Shared legal-dataset constants."""

from __future__ import annotations

CUAD_LABEL_FAMILIES: dict[str, dict[str, object]] = {
    "Source Code Escrow": {
        "scenario_family": "cuad_source_code_continuity_review",
        "question_type": "source-code continuity protection strength",
        "decision_set": [
            "deposit_only",
            "deposit_with_release_trigger",
            "release_plus_use_rights",
            "strong_protection_with_updates_or_cost_terms",
        ],
        "subclaims": [
            "deposit_or_escrow_agent",
            "release_trigger",
            "post_release_use_right",
            "update_duty",
            "cost_or_fee_term",
        ],
    },
    "Rofr/Rofo/Rofn": {
        "scenario_family": "cuad_transaction_restriction_review",
        "question_type": "transaction restriction strength",
        "decision_set": [
            "notice_only",
            "negotiation_or_offer_right",
            "matching_right_with_deadline",
            "strong_restriction_with_expiration_or_remedy_terms",
        ],
        "subclaims": [
            "covered_opportunity_or_asset",
            "notice_or_offer_obligation",
            "exercise_or_negotiation_period",
            "matching_right",
            "expiration_or_failure_consequence",
        ],
    },
    "Insurance": {
        "scenario_family": "cuad_insurance_protection_review",
        "question_type": "insurance burden or protection level",
        "decision_set": [
            "coverage_type_only",
            "coverage_with_limit",
            "coverage_with_certificates_or_notice",
            "strong_protection_with_special_coverage_or_remedy",
        ],
        "subclaims": [
            "insurance_type",
            "coverage_limit",
            "certificate_duty",
            "cancellation_notice",
            "additional_insured_or_self_help",
        ],
    },
    "Audit Rights": {
        "scenario_family": "cuad_audit_burden_review",
        "question_type": "audit burden level",
        "decision_set": [
            "records_only",
            "records_review_right",
            "inspection_or_system_access",
            "high_burden_with_frequency_cost_or_process_terms",
        ],
        "subclaims": [
            "recordkeeping_duty",
            "records_review_right",
            "facility_or_system_inspection",
            "notice_or_frequency_term",
            "cost_shift_or_process_term",
        ],
    },
    "License Grant": {
        "scenario_family": "cuad_license_sufficiency_review",
        "question_type": "license sufficiency for intended use",
        "decision_set": [],
        "subclaims": [
            "grant_scope",
            "field_of_use",
            "territory",
            "duration",
            "sublicense_or_transfer_right",
            "use_restriction",
        ],
    },
    "Minimum Commitment": {
        "scenario_family": "cuad_commercial_commitment_review",
        "question_type": "commercial commitment materiality",
        "decision_set": [],
        "subclaims": [
            "minimum_amount",
            "measurement_period",
            "purchase_or_revenue_metric",
            "shortfall_remedy",
            "exceptions_or_adjustments",
        ],
    },
    "Cap On Liability": {
        "scenario_family": "cuad_liability_protection_review",
        "question_type": "liability protection profile",
        "decision_set": [],
        "subclaims": [
            "cap_amount_or_formula",
            "excluded_damages",
            "carveouts",
            "special_warranty_rule",
            "ip_or_confidentiality_exception",
            "claim_time_limit",
        ],
    },
}
