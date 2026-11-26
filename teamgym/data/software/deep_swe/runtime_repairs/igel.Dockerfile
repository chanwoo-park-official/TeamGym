ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /app
# scipy 1.17.1 hangs during stats documentation generation on this amd64 image
# under Apple Silicon emulation. Keep application source at the exact base.
RUN python -m pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 \
 && python -c "import igel,numpy,scipy;print(igel.__file__,numpy.__version__,scipy.__version__)" \
 && test -z "$(git status --porcelain)"
