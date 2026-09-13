from morphe.testing_fixtures import make_synthetic_regions

# pyrefly: ignore [unknown-name]
smoke_cfg = snakemake.config.get("smoke", {})
# pyrefly: ignore [unknown-name]
make_synthetic_regions(snakemake.output[0], n_regions=6, size=smoke_cfg.get("img_size", 64))
