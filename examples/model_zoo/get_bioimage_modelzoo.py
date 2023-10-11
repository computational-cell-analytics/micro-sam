from micro_sam.modelzoo import model_export


def main():
    parser = model_export._get_modelzoo_parser()
    args = parser.parse_args()

    model_export.get_modelzoo_yaml(
        image_path=args.input_path,
        box_prompts_path=args.boxes_path,
        model_type=args.model_type,
        output_path=args.output_path,
        doc_path=args.doc_path
    )


if __name__ == "__main__":
    main()
