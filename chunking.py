import re

# A heading looks like "1. INTRODUCTION", "2.1 Core Platform", "3. KEY REQUIREMENTS":
# a (possibly dotted) number, an optional trailing dot, whitespace, then text.
# Note: "R1. Experience..." and "- RFP issued..." start with a letter/dash, so
# they do NOT match and correctly remain body content.
_HEADING_RE = re.compile(r'^\d+(?:\.\d+)*\.?\s+\S')

# A separator is a whole line of box-drawing / dash / underscore / equals chars
# (─-╿ is the Unicode "Box Drawing" block, e.g. the ━ bars in the RFP).
_SEPARATOR_RE = re.compile(r'^[─-╿\-_=]{3,}$')


def chunk_text(raw):
    """Split a document into section-based chunks.

    A new chunk begins at every detected heading (e.g. "2. SCOPE OF
    REQUIREMENTS" or "2.1 Core Platform Replacement"). Horizontal separator
    lines also act as boundaries, and blank lines are dropped. Each chunk is
    its heading plus all body lines up to the next heading/separator.

    This is the single source of truth for chunking: indexer.py builds the
    index from it and the query side reloads texts the same way, so the
    indices returned by index.search() always line up with the texts.
    """
    chunks = []
    current = []

    def flush():
        if current:
            chunks.append("\n".join(current))
            current.clear()

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _SEPARATOR_RE.match(line):
            flush()
            continue
        if _HEADING_RE.match(line):
            flush()
        current.append(line)

    flush()
    return chunks
