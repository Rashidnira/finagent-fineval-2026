"""System D (Hybrid-RAG Evidence Highlighting) — retrieval data layer.

Isolated from System C: nothing in this package imports from, or is imported
by, the System C generation/scoring pipeline. No model API calls are made
here; this layer only parses the supplied query contexts into retrievable
chunk collections (news / financial statements) with stable IDs and
metadata. Gold/reference answers are never read.
"""
