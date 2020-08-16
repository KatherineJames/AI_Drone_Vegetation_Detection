These files are the files necessary to perform 
k-fold validation, inference training and final
evaluation against test set.

Used for paper to evaluate U-Net trained with 
weight map based loss.

kfold_wmbl_unet.py  -- train using kfold validation
wmbl_inference.py - train inference model using -s="inf"
wmbl_inference.py - evaluate inference model using -s="test"

Use -w and -t to set weights and thicknessed for the weight maps

Individual files list arguments and required folders\file 
structure.