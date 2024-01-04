if __name__ == "__main__":
    from data_management.core import make_joblib_pickle
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--archive", "-a", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, required=True)

    args = parser.parse_args()
    make_joblib_pickle(
        args.output,
        archive_file=args.archive,
    )
