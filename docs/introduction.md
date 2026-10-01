# Introduction to Horizontal Diffusion Maps

HDM extends diffusion maps to collections of related data objects, such as a shape or image collection. A random walk on the collection (the base) is lifted through correspondence maps across each object's internal structure (the fiber). This produces a shared coordinate system that jointly embeds each object's internal structure and the pairwise relations between objects.

```{figure} ../media/hdm_demo.gif
:width: 600px
:alt: A random walk on a neighbor graph on the base manifold, lifted through the fibres

A random walk on a neighbor graph on the base manifold lifted through the fibres: hopping between objects moves to the corresponding point on each object's structure.
```

For the full treatment, see the [paper](https://www.sciencedirect.com/science/article/pii/S1063520318302215).
