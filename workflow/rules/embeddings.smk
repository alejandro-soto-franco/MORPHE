"""GCN cell-type classifier + autoencoder RGB embedding.

Not wired into ``rule all``: requires the full per-cell CODEX table
(``embeddings.gcn.dataset_csv``), which is 2.6M rows and outside this
repository's data budget. Point config at your own table and run
``pixi run snakemake export_region_images`` directly. See the README.
"""

rule train_gcn_classifier:
    output:
        probs_csv="results/embeddings/cell_gnn_probs.csv",
    script:
        "../scripts/train_gcn_classifier.py"

rule train_autoencoder:
    input:
        probs_csv=rules.train_gcn_classifier.output.probs_csv,
    output:
        checkpoint=config["embeddings"]["autoencoder"]["checkpoint_path"],
        rgb_csv="results/embeddings/node_embeddings_3d.csv",
    script:
        "../scripts/train_autoencoder.py"

rule export_region_images:
    input:
        rgb_csv=rules.train_autoencoder.output.rgb_csv,
    output:
        directory("results/embeddings/region_images"),
    script:
        "../scripts/export_region_images.py"
