import features

FEATURES_TABLE = {}

for name, cls in vars(features).items():
    if isinstance(cls, type) and hasattr(cls, "get_id"):
        method = getattr(cls, "get_id")
        if callable(method):
            try:
                result = method()
                FEATURES_TABLE[cls.get_id()] = cls
            except NotImplementedError:
                pass

print(FEATURES_TABLE)
            