import os
from flask import Flask, request, jsonify, render_template
import argparse
from scripts.txt2img import main

app = Flask(__name__, template_folder='.')

@app.route("/", methods=["GET"])
def index():
    args = request.args

    opt = argparse.Namespace(
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
    )

    main(opt)

    return jsonify({"message": "Image generated successfully"})

@app.route('/images')
def images():
    images = os.listdir('/app/static')
    images = [f"/app/static/{image}" for image in images]
    return render_template('index.html', images=images)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
