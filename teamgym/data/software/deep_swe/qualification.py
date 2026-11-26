"""Recorded acceptance policy for complete-evidence repair attempts."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QualificationPolicy:
    minimum_passes: int = 2
    repetitions: int = 3

    def __post_init__(self):
        if (type(self.minimum_passes) is not int or type(self.repetitions) is not int
                or not 1 <= self.minimum_passes <= self.repetitions):
            raise ValueError("qualification requires 1 <= minimum_passes <= repetitions")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_manifest(cls, manifest):
        policy = manifest.get('qualification', cls().to_dict())
        if not isinstance(policy, dict) or set(policy) != {'minimum_passes', 'repetitions'}:
            raise ValueError('invalid qualification policy schema')
        return cls(**policy)


DEFAULT_QUALIFICATION_POLICY = QualificationPolicy()
