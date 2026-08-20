# ReCoAlign architecture report

Status: implementation complete; scientific efficacy pending.

The method learns latent structure tokens by cross-attending typed queries to
visual tokens, then gates those tokens into the language-model hidden space.  This
directly targets the diagnosed visual-semantic → structured-reasoning interface.

Graph annotations are consumed only by the structural-consistency training loss;
inference receives no ground-truth graph.  The implementation exposes exactly
three loss terms and supports no-structure/random-token ablations.
