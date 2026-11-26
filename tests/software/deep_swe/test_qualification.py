"""Configurable verifier acceptance survives serialization and rejects missing trials."""

import pytest

from teamgym.data.software.deep_swe.export import (
    candidate_sha256, freeze_export, load_export, make_validation_receipt, sha256,
)
from teamgym.data.software.deep_swe.qualification import QualificationPolicy
from tests.software.deep_swe.test_export import fixture, evidence, episode


@pytest.mark.parametrize('minimum,repetitions', [(0, 3), (4, 3), (2, 0), (True, 3), (2, 3.0)])
def test_invalid_qualification_policy_rejected(minimum, repetitions):
    with pytest.raises(ValueError):
        QualificationPolicy(minimum, repetitions)


@pytest.mark.parametrize('minimum,repetitions,passes,actual_runs,error', [
    (2, 3, 2, 3, None), (3, 5, 3, 5, None), (1, 1, 1, 1, None),
    (3, 5, 2, 5, 'complete_evidence_qualification_failed'),
    (3, 5, 3, 3, 'incomplete_episode_evidence'),
    (2, 3, 2, 4, 'incomplete_episode_evidence'),
])
def test_export_enforces_recorded_policy(tmp_path, minimum, repetitions, passes, actual_runs, error):
    candidate, bundle = fixture()
    policy = QualificationPolicy(minimum, repetitions).to_dict()
    candidate['execution_manifest']['qualification'] = policy
    bundle['execution_manifest']['qualification'] = policy
    bundle['candidate_sha256'] = candidate_sha256(candidate)
    data = evidence(candidate, bundle)
    execution = candidate['execution_manifest']
    data['qualification'] = [episode(i < passes, execution=execution) for i in range(actual_runs)]
    # Every trial is bound to the same execution identity, including its policy.
    groups = [data['screening'], data['qualification']]
    groups.extend(t['episodes'] for t in data['loo']['targets'])
    groups.extend(r['episodes'] for r in data['participant_results'])
    for group in groups:
        for row in group:
            row['execution_manifest_sha256'] = sha256(execution)
    if error:
        with pytest.raises(ValueError, match=error):
            make_validation_receipt(candidate, bundle, **data)
        return
    receipt = make_validation_receipt(candidate, bundle, **data)
    path = tmp_path / 'instance.json'
    freeze_export(path, candidate, bundle, receipt)
    assert load_export(path).execution_manifest['qualification'] == policy
