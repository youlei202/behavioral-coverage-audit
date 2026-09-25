# Archived traffic comparison

`src/exp14_multicity_tabular_pier.py` and shared ISQED modules preserve the original forecasting implementation. Paper replay uses all 31 archived city summaries and the 961-row peer-weight table. Relative residual and replacement-impact normalizations are regenerated without modifying coefficients.

Negative simplex coefficients remain visible, including the archived minimum near -0.0048245. No optimization repair, city removal, missing pruning trajectory, or synthetic tail curve is introduced. Original raw prediction/training caches were not recovered, so deeper model refitting is not claimed. The source-derived full command is documented separately and was not executed.
