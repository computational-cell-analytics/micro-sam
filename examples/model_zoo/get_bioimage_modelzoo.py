from micro_sam import model_zoo


def main():
    parser = model_zoo._get_modelzoo_parser()
    args = parser.parse_args()

    model_zoo.get_modelzoo_yaml(
        image_path=args.input_path,
        box_prompts=None,
        model_type=args.model_type,
        output_path=args.output_path,
        doc_path=args.doc_path
    )


if __name__ == "__main__":
    main()
