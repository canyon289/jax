:orphan:

.. _beginner_guide:


Beginner Guide
==============

Welcome to the beginners guide for JAX. 
On this page we will introduce you to the key ideas of Jax,
show you how to get Jax running
and provide you some tutorials to get started.

If looking to jump straight take a look at the (Jax Quickstart)[../notebooks/quickstart].

Key Ideas of Jax
================
At a high level aims to empower researchers with a small, agile,
constantly evolving software system and excellent support.

Specifically Jax does this by providing researchers with

1. **Familiar API**: JAX provides a familiar NumPy-style API for ease of adoption by researchers and engineers.
2. **Transformations**: JAX includes composable function transformations for compilation, batching, automatic differentiation, and parallelization.
3. **Run Anywhere**: The same code executes on multiple backends, including CPU, GPU, & TPU


If you prefer a video introduction here's one from Jake Vanderplas.

.. raw:: html

	<iframe width="640" height="360" src="https://www.youtube.com/embed/WdTeDXsOSj4"
	 title="Intro to JAX: Accelerating Machine Learning research"
	frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
	allowfullscreen></iframe>

Installing JAX
==============
Installation instructions are available on the `Install Guide <https://github.com/google/jax#installation>`_
Alternatively Jax comes preinstalled on `Google Colab <https://colab.research.google.com>`_.

Getting familiar with Jax
=========================
For most users starting out the key functionalities of JAX to become familiar with are

- :func:`jax.jit` 
- :func:`jax.grad` 
- :func:`jax.vmap` 

:ref:`Jax-101` provides code examples with greater explanation.