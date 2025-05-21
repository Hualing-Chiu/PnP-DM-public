from typing import Optional, TypeGuard, TypeVar

T = TypeVar("T")

def exists(x: Optional[T]) -> TypeGuard[T]:
    return x is not None

def defaults(x: Optional[T], default_value: T) -> T:
    return x if exists(x) else default_value

def unwrap(x: Optional[T]) -> T:
    assert exists(x)
    return x