from ego4d.internal.download.cli import (
    create_arg_parse,
    main,
)

if __name__ == "__main__":
    parser = create_arg_parse(
        script_name="ego4d/egoexo/download/cli.py",
        release_name="public",
        base_dir="s3://ego4d-consortium-sharing/egoexo/releases/",
    )
    args = parser.parse_args()
    main(args)
