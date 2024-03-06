# !/bin/bash

if [[ $2 = "grete" ]]
then
    _embedding_path="/scratch/usr/nimanwai/test"
    _checkpoint_path="/scratch/usr/nimanwai/test"
elif [[ $2 = "local" ]]
then
    _embedding_path="~/embeddings"
    _checkpoint_path="~/models/test"
else
    echo "Choose correct system."
    exit
fi

if [[ $1 = "v1" ]]
then
    python evaluate_lucchi.py -m vit_b_lm -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_v1
elif [[ $1 = "v2" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_v2 --checkpoint ${_checkpoint_path}/micro-sam/vit_b/em_organelles/best.pt
elif [[ $1 = "default" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_vanilla/
elif [[ $1 = "mp_0" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.1" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.1 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.1/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.2" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.2 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.2/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.3" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.3 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.3/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.4" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.4 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.4/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.5" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.5 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.5/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.6" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.6 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.6/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.7" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.7 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.7/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.8" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.8 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.8/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_0.9" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_0.9 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_0.9/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
elif [[ $1 = "mp_1" ]]
then
    python evaluate_lucchi.py -m vit_b -e ${_embedding_path}/lucchi_embeddings/embeddings_vit_b_finetuned_mp_1 --checkpoint ${_checkpoint_path}/for_3d_em_organelles/mask_probability_1/checkpoints/vit_b/mito_nuc_em_generalist_sam/best.pt 
else
    echo "Nope. Try again."
fi