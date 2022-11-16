JAX reference documentation
===========================

JAX is Autograd_ and XLA_, brought together for high-performance numerical computing and machine learning research.
It provides composable transformations of Python+NumPy programs: differentiate, vectorize,
parallelize, Just-In-Time compile to GPU/TPU, and more.

.. note::
   JAX 0.4.1 introduces new parallelism APIs, including breaking changes to :func:`jax.experimental.pjit` and a new unified ``jax.Array`` type.
   Please see `Distributed arrays and automatic parallelization <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`_ tutorial and the :ref:`jax-array-migration`
   guide for more information.


User Specific Guides
====================
| :ref:`beginner_guide`
| :ref:`advanced_guide`
| :ref:`contributor_guide`


Reference Links
===============
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   notebooks/quickstart
   notebooks/thinking_in_jax
   notebooks/Common_Gotchas_in_JAX

.. toctree::
   :maxdepth: 1
   :caption: Reference Documentation

   faq
   async_dispatch
   aot
   jaxpr
   notebooks/convolutions
   pytrees
   jax_array_migration
   type_promotion
   errors
   transfer_guard
   glossary
   changelog

.. toctree::
   :maxdepth: 1
   :caption: API documentation

   jax

.. toctree::
   :maxdepth: 1
   :caption: Notes

   api_compatibility
   deprecation
   concurrency
   gpu_memory_allocation
   profiling
   device_memory_profiling
   rank_promotion_warning


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Autograd: https://github.com/hips/autograd
.. _XLA: https://www.tensorflow.org/xla
