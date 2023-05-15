import pulumi
from pulumi_gcp import projects, container, config
from pulumi_docker import Image
from pulumi_kubernetes import Provider
from pulumi_kubernetes.core.v1 import Service
from pulumi_kubernetes.apps.v1 import Deployment
import google.auth
from google.auth.transport.requests import Request
from pulumi_kubernetes.apps.v1 import DaemonSet


config = pulumi.Config()
name = config.require("name")
project = config.require("project")
location = config.require("region")
node_count = config.require_int("node_count")
machine_type = config.require("machine_type")
replicas = config.require_int("replicas")


# Fetch access token from credentials
def get_access_token():
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    creds, _ = google.auth.default(scopes=scopes)

    if not creds.token:
        creds.refresh(Request())

    return creds.token


# Enable services
container_api = projects.Service(
    "container.googleapis.com",
    service="container.googleapis.com",
    project=project,
)
cloud_resource_manager_api = projects.Service(
    "cloudresourcemanager.googleapis.com",
    service="cloudresourcemanager.googleapis.com",
    project=project,
)

# Build and push Docker image to container registry
image = Image(
    name,
    image_name=f"gcr.io/{project}/{name}",
    build={
        "context": ".",
        "platform": "linux/amd64",
    },
    registry={
        "server": "gcr.io",
        "username": "oauth2accesstoken",
        "password": pulumi.Output.from_input(get_access_token()),
    },
    opts=pulumi.ResourceOptions(depends_on=[container_api, cloud_resource_manager_api]),
)

# Fetch GKE engine versions
def get_engine_versions(digest):
    return container.get_engine_versions(project=project, location=location)


engine_versions = pulumi.Output.all([image.repo_digest]).apply(get_engine_versions)

# Create Kubernetes cluster
cluster = container.Cluster(
    name,
    project=project,
    location=location,
    initial_node_count=node_count,
    min_master_version=engine_versions.latest_master_version,
    node_version=engine_versions.latest_master_version,
    node_config={
        "machine_type": machine_type,
        "oauth_scopes": [
            "https://www.googleapis.com/auth/compute",
            "https://www.googleapis.com/auth/devstorage.read_only",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring",
        ],
        "image_type": "COS_CONTAINERD",
        "guest_accelerator": [
            {
                "type": "nvidia-tesla-a100",
                "count": 1,
            }
        ],
    },
    opts=pulumi.ResourceOptions(depends_on=[image]),
)


def generate_kubeconfig(name, endpoint, master_auth):
    context = f"{project}_{location}_{name}"
    return f"""apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: {master_auth['cluster_ca_certificate']}
    server: https://{endpoint}
  name: {context}
contexts:
- context:
    cluster: {context}
    user: {context}
  name: {context}
current-context: {context}
kind: Config
preferences: {{}}
users:
- name: {context}
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: gke-gcloud-auth-plugin
      installHint: Install gke-gcloud-auth-plugin for use with kubectl by following
        https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke
      provideClusterInfo: true
"""


kubeconfig = pulumi.Output.all(
    cluster.name, cluster.endpoint, cluster.master_auth
).apply(lambda args: generate_kubeconfig(*args))

# Create a Kubernetes provider
cluster_provider = Provider(name, kubeconfig=kubeconfig)

# Deploy NVIDIA daemon set
nvidia_gpu_device_plugin = DaemonSet(
    "nvidia-gpu-device-plugin",
    metadata={
        "name": "nvidia-driver-installer",
        "namespace": "kube-system",
        "labels": {"k8s-app": "nvidia-driver-installer"},
    },
    spec={
        "selector": {"matchLabels": {"k8s-app": "nvidia-driver-installer"}},
        "updateStrategy": {"type": "RollingUpdate"},
        "template": {
            "metadata": {
                "labels": {
                    "name": "nvidia-driver-installer",
                    "k8s-app": "nvidia-driver-installer",
                }
            },
            "spec": {
                "priorityClassName": "system-node-critical",
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "cloud.google.com/gke-accelerator",
                                            "operator": "Exists",
                                        },
                                        {
                                            "key": "cloud.google.com/gke-gpu-driver-version",
                                            "operator": "DoesNotExist",
                                        },
                                    ]
                                }
                            ]
                        }
                    }
                },
                "tolerations": [{"operator": "Exists"}],
                "hostNetwork": True,
                "hostPID": True,
                "volumes": [
                    {"name": "dev", "hostPath": {"path": "/dev"}},
                    {
                        "name": "vulkan-icd-mount",
                        "hostPath": {
                            "path": "/home/kubernetes/bin/nvidia/vulkan/icd.d"
                        },
                    },
                    {
                        "name": "nvidia-install-dir-host",
                        "hostPath": {"path": "/home/kubernetes/bin/nvidia"},
                    },
                    {"name": "root-mount", "hostPath": {"path": "/"}},
                    {"name": "cos-tools", "hostPath": {"path": "/var/lib/cos-tools"}},
                    {"name": "nvidia-config", "hostPath": {"path": "/etc/nvidia"}},
                ],
                "initContainers": [
                    {
                        "image": "cos-nvidia-installer:fixed",
                        "imagePullPolicy": "Never",
                        "name": "nvidia-driver-installer",
                        "resources": {"requests": {"cpu": "150m"}},
                        "securityContext": {"privileged": True},
                        "env": [
                            {
                                "name": "NVIDIA_INSTALL_DIR_HOST",
                                "value": "/home/kubernetes/bin/nvidia",
                            },
                            {
                                "name": "NVIDIA_INSTALL_DIR_CONTAINER",
                                "value": "/usr/local/nvidia",
                            },
                            {
                                "name": "VULKAN_ICD_DIR_HOST",
                                "value": "/home/kubernetes/bin/nvidia/vulkan/icd.d",
                            },
                            {
                                "name": "VULKAN_ICD_DIR_CONTAINER",
                                "value": "/etc/vulkan/icd.d",
                            },
                            {"name": "ROOT_MOUNT_DIR", "value": "/root"},
                            {
                                "name": "COS_TOOLS_DIR_HOST",
                                "value": "/var/lib/cos-tools",
                            },
                            {
                                "name": "COS_TOOLS_DIR_CONTAINER",
                                "value": "/build/cos-tools",
                            },
                        ],
                        "volumeMounts": [
                            {
                                "name": "nvidia-install-dir-host",
                                "mountPath": "/usr/local/nvidia",
                            },
                            {
                                "name": "vulkan-icd-mount",
                                "mountPath": "/etc/vulkan/icd.d",
                            },
                            {"name": "dev", "mountPath": "/dev"},
                            {"name": "root-mount", "mountPath": "/root"},
                            {"name": "cos-tools", "mountPath": "/build/cos-tools"},
                        ],
                    },
                    {
                        "image": "gcr.io/gke-release/nvidia-partition-gpu@sha256:c54fd003948fac687c2a93a55ea6e4d47ffbd641278a9191e75e822fe72471c2",
                        "name": "partition-gpus",
                        "env": [
                            {
                                "name": "LD_LIBRARY_PATH",
                                "value": "/usr/local/nvidia/lib64",
                            }
                        ],
                        "resources": {"requests": {"cpu": "150m"}},
                        "securityContext": {"privileged": True},
                        "volumeMounts": [
                            {
                                "name": "nvidia-install-dir-host",
                                "mountPath": "/usr/local/nvidia",
                            },
                            {"name": "dev", "mountPath": "/dev"},
                            {"name": "nvidia-config", "mountPath": "/etc/nvidia"},
                        ],
                    },
                ],
                "containers": [
                    {"image": "gcr.io/google-containers/pause:2.0", "name": "pause"}
                ],
            },
        },
    },
    opts=pulumi.ResourceOptions(provider=cluster_provider),
)


# Create Kubernetes deployment
deployment = Deployment(
    name,
    metadata={"name": name},
    spec={
        "strategy": {
            "type": "Recreate",
        },
        "replicas": replicas,
        "selector": {"matchLabels": {"app": name}},
        "template": {
            "metadata": {"labels": {"app": name}},
            "spec": {
                "containers": [
                    {
                        "name": name,
                        "image": image.repo_digest,
                        "resources": {"limits": {"nvidia.com/gpu": 1}},
                        "ports": [{"containerPort": 80}],
                    },
                ],
            },
        },
    },
    opts=pulumi.ResourceOptions(
        provider=cluster_provider, depends_on=[nvidia_gpu_device_plugin]
    ),
)

# Create Kubernetes service to expose port 80
service = Service(
    name,
    spec={
        "type": "LoadBalancer",
        "selector": {"app": name},
        "ports": [
            {
                "protocol": "TCP",
                "port": 80,
                "targetPort": 80,
            },
        ],
    },
    opts=pulumi.ResourceOptions(provider=cluster_provider, depends_on=[deployment]),
)

# Export IP address of the LoadBalancer
pulumi.export(
    "load_balancer_ip",
    service.status.apply(lambda status: status.load_balancer.ingress[0].ip),
)
