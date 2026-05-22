from __future__ import annotations

from pathlib import Path

PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "prompts" / "network_analysis.txt"
)


def format_top_centrality(top_centrality: list[tuple[str, float]]) -> str:
    if not top_centrality:
        return "  - (sin nodos con menciones en el corpus)"
    return "\n".join(
        f"  - @{node}: centralidad PageRank = {score:.4f}"
        for node, score in top_centrality
    )


def build_network_prompt(
    *,
    nodes: int,
    edges: int,
    density: float,
    communities_count: int,
    top_centrality: list[tuple[str, float]],
    top_hashtag: str,
    top_hashtag_freq: int,
    template_path: Path | None = None,
) -> str:
    """
    Rellena la plantilla de prompt con métricas del análisis de red.

    La plantilla por defecto está en ``llm/prompts/network_analysis.txt``.
    """
    path = template_path or PROMPT_TEMPLATE_PATH
    template = path.read_text(encoding="utf-8")

    communities_text = (
        f"{communities_count} comunidades (Louvain)"
        if nodes > 0
        else "sin comunidades"
    )

    return template.format(
        nodes=nodes,
        edges=edges,
        density=density,
        communities_text=communities_text,
        top_centrality_lines=format_top_centrality(top_centrality),
        top_hashtag=top_hashtag,
        top_hashtag_freq=top_hashtag_freq,
    )
