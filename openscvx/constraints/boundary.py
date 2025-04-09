import jax.numpy as jnp

ALLOWED_TYPES = {"Fix", "Free", "Minimize"}

class TypeList:
    def __init__(self, values):
        self.values = list(values)
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, value):
        # Handle slice assignment
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.values)))
            # If a single string is given, replicate it for the slice.
            if isinstance(value, str):
                value = [value] * len(indices)
            # Validate every element in the assigned slice.
            for idx, v in zip(indices, value):
                self._validate(v)
                self.values[idx] = v
        else:
            self._validate(value)
            self.values[key] = value

    def _validate(self, v):
        if v not in ALLOWED_TYPES:
            raise ValueError(f"Type must be one of {ALLOWED_TYPES}, got {v}")

    def __len__(self):
        return len(self.values)
    
    def __repr__(self):
        return repr(self.values)

class BoundaryConstraint:
    def __init__(self, value: jnp.ndarray, types=None):
        self.value = value
        if types is None:
            types = ["Fix"] * len(value)
        elif len(types) != len(value):
            raise ValueError("Length of types must match length of value")
        # Internally store types using our helper class.
        self._types = TypeList(types)
    
    @property
    def type(self):
        # Expose the helper class so that slice assignments work naturally.
        return self._types