# Deferred Ideas

- Add an optional precomputed inverse-grid-spacing object/API so repeated locate
  calls replace division with multiplication. This needs a public API and cache
  invalidation design, so it is outside the current drop-in optimization scope.
- Add a 2D evaluation overload accepting scalar indices and weights, matching
  model kernels that already hold components separately. This could avoid creating
  or passing two-element array views but changes the public API.
- Add explicit uniform-grid locate functions using arithmetic instead of search.
  Automatic per-call uniformity detection would cost too much; a separate API or
  grid descriptor would be required.
