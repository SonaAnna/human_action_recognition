from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import uuid
import logging
import argparse
logging.basicConfig(level=logging.DEBUG)
from azure.identity import AzureCliCredential, DefaultAzureCredential

from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.ai.ml.entities import ManagedOnlineDeployment

# authenticate
credential = AzureCliCredential()
# credential = DefaultAzureCredential()

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--job_name", type=str, required=True, help='job name to register a model')
args = parser.parse_args()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id = '8273dae1-4494-4ccb-bccf-e5465e3c3482',
    resource_group_name="com774-b00953832-rg-",
    workspace_name="com774-b00953832-ws-",
)

job_name = args.job_name    

run_model = Model(
    path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/",
    name="HAR-model",
    description="model created using workflow",
    type=AssetTypes.MLFLOW_MODEL,
)

# Register the model
ml_client.models.create_or_update(run_model)

registered_model_name="HAR-model"

# Let's pick the latest version of the model
latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=registered_model_name)]
)
print(latest_model_version)

# Create a unique name for the endpoint
online_endpoint_name = "endpoint-har-" + str(uuid.uuid4())[:8]

# define an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="online endpoint for har model",
    auth_mode="key",
    tags={
        "training_dataset": "credit_defaults",
    },
)

# create the online endpoint
#expect the endpoint to take approximately 2 minutes.
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)

# Choose the latest version of our registered model for deployment
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)

# define an online deployment
# if you run into an out of quota error, change the instance_type to a comparable VM that is available.\
# Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.
blue_deployment = ManagedOnlineDeployment(
    name="har-blue-deployment",
    endpoint_name=online_endpoint_name,
    model=model,
    instance_type="Standard_D2as_v4",
    instance_count=1,
)

# create the online deployment
blue_deployment = ml_client.online_deployments.begin_create_or_update(
    blue_deployment
).result()

# blue deployment takes 100% traffic
# expect the deployment to take approximately 8 to 10 minutes.
endpoint.traffic = {"har-blue-deployment": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("Deployment completed")
