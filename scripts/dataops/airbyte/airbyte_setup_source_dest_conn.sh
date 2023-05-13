# SOME MISC COMMANDS. NOT TO BE CALLED AS A SCRIPT DIRECTLY.

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
octavia generate connection --source sources/my_custom_postgres/configuration.yaml --destination destinations/my_custom_bigquery/configuration.yaml my_postgres_to_my_bigquery

octavia apply --file <path-to-connection-configuration>.yaml # to apply the config