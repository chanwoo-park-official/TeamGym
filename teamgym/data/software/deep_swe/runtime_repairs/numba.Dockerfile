ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /app
# The image already has llvmlite 0.47.0. The old compatibility sed is obsolete.
RUN git checkout -- numba/__init__.py \
 && test -z "$(git status --porcelain)" \
 && python3 -c "import numba,llvmlite; print(numba.__version__,llvmlite.__version__)"
