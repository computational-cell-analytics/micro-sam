from micro_sam.evaluation.sam_inference import run_livecell_inference, livecell_inference_parser

parser = livecell_inference_parser()
args = parser.parse_args()
run_livecell_inference(args)
