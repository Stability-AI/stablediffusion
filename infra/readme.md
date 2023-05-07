## Stable Diffusion on Google Cloud Platform using Pulumi

### Requirements

- Python 3
- Pulumi, https://www.pulumi.com/docs/get-started/install/

### Instructions

1. Create a service account in Google Cloud Platform as follows:

	* Log in to the Google Cloud Console (console.cloud.google.com)
	* Select the project in which you want to create a service account
	* Click on the "IAM & Admin" option in the left-hand menu
	* Click on "Service Accounts" in the left-hand menu
	* Click the "Create Service Account" button
	* Enter a name for the service account
	* Select "Editor" role for the service account
	* Select "Furnish a new private key" option and choose JSON
	* Click "Create" to create the service account
	* Once you have created the service account, you will be prompted to download the private key file

2. Rename service account private key file to `gcp.json` and place it inside the `/infra` directory
3. Rename `.sample.env` to `.env` and edit its contents
4. Execute in your terminal `./start.sh` to:

	* Enable Google Cloud Services
	* Build and push a Docker image to Google Container Registry
	* Spin up a Kubernetes cluster running a A100 GPU
	* Install NVIDIA driver into Kubernetes cluster
	* Launch the Stable Diffusion Kubernetes deployment
	* Expose Stable Diffusion to the public internet using a Kubernetes Service

### How to use

Once `./start.sh` finishes running it will output `load_balancer_ip`, for example: `load_balancer_ip: "34.172.48.137"`. Use the IP provided to query Stable Diffusion.

Parameters:
```
prompt=args.get("prompt", "a professional photograph of an astronaut riding a triceratops"),
outdir=args.get("outdir", "static"),
steps=args.get("steps", 50),
plms=args.get("plms", False),
dpm=args.get("dpm", False),
fixed_code=args.get("fixed_code", False),
ddim_eta=args.get("ddim_eta", 0.0),
n_iter=args.get("n_iter", 3),
H=args.get("H", 512),
W=args.get("W", 512),
C=args.get("C", 4),
f=args.get("f", 8),
n_samples=args.get("n_samples", 3),
n_rows=args.get("n_rows", 0),
scale=args.get("scale", 9.0),
from_file=args.get("from_file", None),
config=args.get("config", "configs/stable-diffusion/v2-inference-v.yaml"),
ckpt=args.get("ckpt", "checkpoints/v2-1_768-ema-pruned.ckpt"),
seed=args.get("seed", 42),
precision=args.get("precision", "autocast"),
repeat=args.get("repeat", 1),
device=args.get("device", "cpu"),
torchscript=args.get("torchscript", False),
ipex=args.get("ipex", False),
bf16=args.get("bf16", False)
 ```

For example: `http://34.172.48.137/?prompt=Your_Query_Here`. Replace `Your_Query_Here` with your desired query text.

To check the generated images navigate to `http://34.172.48.137/images`.

Remember to URL-encode the text parameter if it contains special characters or spaces. For example, you can replace spaces with `%20`.

### Delete cluster and revert all changes

To delete the cluster and revert all changes, execute in your terminal: `./destroy.sh`.

### Support

If you like this project and find it useful, please consider giving it a star. Your support is appreciated! :hearts:

If you have any questions or suggestions, feel free to reach out to Carlos at calufa@gmail.com or connecting on LinkedIn: https://www.linkedin.com/in/carloschinchilla/.
