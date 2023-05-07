source .env
rm -rf ./app
rsync --exclude='.' --recursive --copy-links ../ ./app
pulumi config set name $NAME --stack dev
pulumi config set project $PROJECT --stack dev
pulumi config set region $REGION --stack dev
pulumi config set node_count $NODE_COUNT --stack dev
pulumi config set machine_type $MACHINE_TYPE --stack dev
pulumi config set replicas $REPLICAS --stack dev
pulumi up --yes --stack dev
