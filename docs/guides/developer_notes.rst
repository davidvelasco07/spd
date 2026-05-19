Developer Notes
===============

Local experimental modules
--------------------------

The file ``src/induction/induction_fv_scheme.py`` is currently treated as local
experimental work and intentionally excluded from the branch commit stack in this
documentation pass.

When promoting it to the shared branch:

1. Add dedicated regression tests (2D and 3D, short and long horizon).
2. Validate stability against SD induction references.
3. Document the final edge-state and boundary strategy in this docs tree.

Code organization conventions
-----------------------------

- Keep reusable utilities in package namespaces (``runtime``, ``numerics``).
- Preserve backward-compatible alias modules only when needed for transition.
- Prefer small simulator wrappers over duplicating scheme/integrator logic.
- Keep tests aligned with modular import paths.

Building docs
-------------

From repository root:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html
