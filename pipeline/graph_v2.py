"""V2 graph: the 8-node async pipeline with memory + loop visibility.

memory_check -> research -> outline -> writer -> fact_check -> seo -> [router] ->
editor (loops back to fact_check) / publisher.

Implemented in V2-M6.
"""

from __future__ import annotations
