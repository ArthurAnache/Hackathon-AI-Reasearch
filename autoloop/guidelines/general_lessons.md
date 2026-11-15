# General lessons for a cold start on a step-function autoconvolution bound (from the 1d runs of 14 Nov 2026)

Lessons, not data: no certificate, no solution, no seed number and no score of any problem is given here. They come
from runs on a different constant (Tao 1d) and may transfer only in part; that must be said with any result.

1. Breadth first: short starts are cheap and rank almost as well as long ones. Make many short starts before
   spending on depth.
2. Do not cut early: the rank before the last resolution stage predicts the final rank only roughly. Carry a wide set
   (20 to 30 lineages) into the high-resolution stage.
3. Soft restarts (small p0, such as 4 and 8) moved lineages at high resolution; hard restarts (p0 >= 16) mostly
   stalled. Not every lineage responds, so try soft restarts on many lineages.
4. Soft restarts may also help at intermediate resolutions (conjecture): test against a harder setting on a few
   lineages before relying on it.
5. Chain restarts on the leader: after a soft restart that worked, restart again from its result with p0 in 2..16.
6. More cells is not better by itself: at equal time a moderate resolution beat a 4x larger one. Go higher only when a
   lineage has stalled, with enough iterations per stage.
7. Breadth at the restart stage is cheap on Modal CPUs (about USD 0.03 per 20-minute 1-CPU job).
8. Stop a lineage when two restart rounds in a row gain almost nothing, and move its budget to new starts.
9. A float score is not a result: check exactly only for finalists, and claim only through the gate, with a
   reference read in the last 30 minutes.
