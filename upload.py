from huggingface_hub import upload_folder

upload_folder(
    repo_id='BGLab/microgen3D',  
    repo_type="dataset",
    folder_path=".", 
)
