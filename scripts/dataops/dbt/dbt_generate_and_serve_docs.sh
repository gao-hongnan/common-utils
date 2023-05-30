#!/bin/bash
# curl -o dbt_generate_and_serve_docs.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/dbt/dbt_generate_and_serve_docs.sh

# Function to change into the dbt project directory
generate_and_serve_docs() {
    echo "Generating and serving dbt docs..."
    dbt docs generate
    dbt docs serve
}

generate_and_serve_docs