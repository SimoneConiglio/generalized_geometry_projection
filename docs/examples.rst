Examples
========

Five topology optimisation examples built with the ``ggp`` CLI.  Each
example runs a single command — no Python code required.  The density
plots shown were generated with ``--output-dir docs/_static``.

.. toctree::
   :maxdepth: 1

   ex01_short_cantilever
   ex02_mbb_beam
   ex03_l_shape_bracket
   ex04_alm_cantilever
   ex05_3d_cantilever

Regenerating the images
-----------------------

All images can be regenerated locally (requires the ``ggp`` conda
environment):

.. code-block:: bash

   conda activate ggp
   ggp optimize --preset short_cantilever  --output-dir docs/_static
   ggp optimize --preset mbb             --output-dir docs/_static
   ggp optimize --preset l_shape          --output-dir docs/_static
   ggp optimize --preset alm_cantilever   --output-dir docs/_static
   ggp optimize --preset 3d_cantilever    --output-dir docs/_static
