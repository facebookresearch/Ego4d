from ego4d.internal.download.cli import create_arg_parse, main as download_main


def main() -> None:
    parser = create_arg_parse(
        script_name="egoexo",
        release_name="v2",
        base_dir="s3://ego4d-consortium-sharing/egoexo-public/",
    )
    args = parser.parse_args()
    download_main(args)


if __name__ == "__main__":
    main()
