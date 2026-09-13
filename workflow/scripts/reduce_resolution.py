from morphe.preprocessing.resolution_reduction import run

run(
    # pyrefly: ignore [unknown-name]
    input_csv=snakemake.input.csv,
    # pyrefly: ignore [unknown-name]
    output_csv=snakemake.output.csv,
    # pyrefly: ignore [unknown-name]
    original_dimensions=tuple(snakemake.params.original_dimensions),
)
