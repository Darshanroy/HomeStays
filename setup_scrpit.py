import os


def create_project_structure(folder_file_map):
    project_path = os.getcwd()  # Get the current working directory as project path

    for folder, files in folder_file_map.items():
        folder_path = os.path.join(project_path, folder)
        os.makedirs(folder_path, exist_ok=True)  # Create folder if not exists

        for file_name in files:
            with open(os.path.join(folder_path, file_name), 'w') as f:
                pass

    print("Project structure created successfully.")


if __name__ == "__main__":
    folder_file_map = {
        "pipelines": ["__init__.py", "feature_engineering.py", "inference.py", "training.py"],
        "steps": ["__init__.py", "data_loader.py", "data_preprocessor.py", "data_splitter.py", "inference_predict.py", "inference_preprocessor.py", "model_evaluator.py", "model_promoter.py", "model_trainer.py"]
    }

    create_project_structure(folder_file_map)
