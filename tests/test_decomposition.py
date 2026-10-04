import pytest

from distill.decomposition import MODERATION_RULES, PR_RULES


def test_moderation_rule_prioritises_clear_removal_before_review():
    rule = MODERATION_RULES[0]
    result = rule.probabilities(
        {
            "harassment_or_hate": 0.0,
            "violent_threat": 0.2,
            "spam_or_scam": 0.0,
            "sexual_exploitation_or_self_harm": 0.0,
            "ambiguous_or_context_dependent": 0.5,
        }
    )
    assert result == pytest.approx((0.4, 0.4, 0.2))


def test_pr_rules_use_fixed_risk_and_surface_precedence():
    risk, surface = PR_RULES
    assert risk.probabilities(
        {
            "has_high_risk_condition": 0.2,
            "changes_security_control_or_exposed_surface": 0.5,
            "security_adjacent_but_contained": 0.5,
        }
    ) == pytest.approx((0.2, 0.2, 0.4, 0.2))
    assert surface.probabilities(
        {
            "touches_untrusted_input": 0.1,
            "touches_crypto_secrets": 0.2,
            "touches_auth": 0.3,
            "touches_network": 0.4,
            "touches_infra_permissions": 0.5,
            "touches_personal_data": 0.6,
            "touches_dependencies": 0.7,
        }
    ) == pytest.approx((0.018144, 0.216, 0.18, 0.1, 0.2016, 0.042336, 0.1512, 0.09072))
