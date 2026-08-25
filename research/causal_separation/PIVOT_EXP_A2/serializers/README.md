# Frozen serializer contract

`canonical_json` and `canonical_triples` must round-trip to the same sorted entity and relation fact
tuples. `natural_language` is a secondary reference only. Serializer output may differ lexically,
but semantic fact hashes, fact counts, aliases, and ordering are frozen before inference.

