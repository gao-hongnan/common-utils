git clone https://github.com/airbytehq/airbyte.git
cd airbyte
./run-ab-platform.sh


bash install_octavia_cli.sh
cat ~/.octavia # to add password
mkdir airbyte-configuration && cd airbyte-configuration
octavia init

octavia list connectors sources # to list all sources
octavia list connectors destinations # to list all destinations

# say you want postgres as source
# replace <source_name> with postgres

octavia list connectors sources | grep <source_name> # to get the definition id

octavia generate source <definition_id><source_name> # to generate the source config

# note here source_name corresponds to the SOURCE NAME in the UI and you can technically
# name it anything. There is difference between this source_name and the <source_name> above

# EDIT THE CONFIG FILE TEMPLATE AND
octavia apply --file <path-to-source-configuration>.yaml # to apply the config
# manually test connection since documentation says not supported yet on cli.

octavia list connectors destinations | grep <destination_name> # to get the definition id 22f6c74f-5699-40ff-833c-4a879ea40133
octavia generate destination <definition_id><destination_name> # to generate the destination config

octavia apply --file <path-to-destination-configuration>.yaml # to apply the config

# generate connection source <> destination
octavia generate connection --source <path-to-source-configuration>.yaml --destination <path-to-destination-configuration>.yaml <connection_name>

# example
octavia generate connection --source sources/ed_postgres/configuration.yaml --destination destinations/hn_bigquery_qtstrats/configuration.yaml ed_postgres_to_hn_bigquery_qtstrats

octavia apply --file <path-to-connection-configuration>.yaml # to apply the config