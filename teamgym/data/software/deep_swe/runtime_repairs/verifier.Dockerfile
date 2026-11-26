ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY test.sh /tests/test.sh
COPY test.patch /tests/test.patch
COPY grader.py /tests/grader.py
COPY config.json /tests/config.json
# Keep all official test IDs and assertions. A hanging test records a failure
# instead of preventing the rest of the official suites from being graded.
RUN python -m pip install --no-cache-dir pytest-timeout==2.4.0 \
 && sed -i 's/-p no:cacheprovider /-p no:cacheprovider --timeout=60 --timeout-method=signal /g' /tests/test.sh \
 && chmod +x /tests/test.sh
