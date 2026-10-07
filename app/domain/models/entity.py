from uuid import UUID


class Entity:
    """Entities are equal when their identities are equal, regardless of state.

    Subclasses are dataclasses declared with `eq=False` so this identity-based
    equality is kept instead of the generated field-by-field comparison.
    """

    __slots__ = ()
    id: UUID

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and type(other) is type(self) and self.id == other.id

    def __hash__(self) -> int:
        return hash((type(self), self.id))
