from micro_sam.evaluation.sam_evaluation import run_livecell_evaluation, livecell_evaluation_parser

parser = livecell_evaluation_parser()
args = parser.parse_args()
run_livecell_evaluation(args)
