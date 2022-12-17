JAX reference documentation
===========================

JAX is Autograd_ and XLA_, brought together for high-performance numerical computing and machine learning research.
It provides composable transformations of Python+NumPy programs: differentiate, vectorize,
parallelize, Just-In-Time compile to GPU/TPU, and more.

.. note::
   JAX 0.4.1 introduces new parallelism APIs, including breaking changes to :func:`jax.experimental.pjit` and a new unified ``jax.Array`` type.
   Please see `Distributed arrays and automatic parallelization <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`_ tutorial and the :ref:`jax-array-migration`
   guide for more information.

Try and implement grid from https://executablebooks.org/en/latest/

::::{grid} 1 2 3 3
:class-container: landing-grid
:gutter: 2

:::{grid-item-card}
:link: contribute
:link-type: doc

Contributing Guide 🙌
^^^
Our contributing guide has some tips and pointers to help you learn where and how to contribute to the project.
:::

:::{grid-item-card}
:link: https://github.com/executablebooks/meta/discussions

Community forum 💬
^^^
A place for free-form discussion, brainstorming, and asking questions about how to use the tools in this ecosystem.

:::

:::{grid-item-card}
:link: feature-vote
:link-type: doc

Feature voting 👍
^^^

A voting board to aggregate ideas across our repositories and sort them by thumbs-up responses.

:::


:::{grid-item-card}
:link: tools
:link-type: doc

Tools we build 🔧
^^^

An overview of the tools and standards that this community stewards for communicating with computational narratives.

:::

:::{grid-item-card}
:link: https://compass.executablebooks.org

Our team compass 🧭
^^^

Our team policies, structure, practices, and contributing guides.
:::

:::{grid-item-card}
:link: gallery
:link-type: doc

Gallery of books 🖼️
^^^
A community-maintained list of books that others have built with Jupyter Book.

:::

::::

```{toctree}
:hidden:
:caption: Our tools and standards
tools.md
feature-vote.md
gallery.md
```

```{toctree}
:caption: About our team
:hidden:
contribute.md
Team compass 🧭 <https://compass.executablebooks.org>
updates.md
```



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
