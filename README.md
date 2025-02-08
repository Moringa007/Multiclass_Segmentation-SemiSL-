# Multiclass_Segmentation_ Semi-supervised Contrastive Learning
This project seeks to improve multiple-class image segmentation by applying multiple-step pertaining.
![framework](https://github.com/user-attachments/assets/8d60c40f-c36d-4f04-8768-e5601659f50c)
# Pre-training
We train the model encoder in unsupervised learning by using the Contrastive loss
to run the 1st step:
- python 1step_pretraining.py
to run the 2nd step:
- python 2step_pretraining.py

# Segmenation training
- python training.py

