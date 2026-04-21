import os

# Define full project structure
structure = {
    "model": ["save_model.h5"],
    "data": ["hospital.db"],
    "utils": [
        "auth.py",
        "db.py",
        "prediction.py",
        "gradcam.py",
        "report.py",
        "explain.py"
    ],
    "app": ["app.py"],
    "reports": []
}

def create_structure(base_path="."):

    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)

        # Create folder
        os.makedirs(folder_path, exist_ok=True)

        # Create files inside folder
        for file in files:
            file_path = os.path.join(folder_path, file)

            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("")  # empty file

    print("✅ Project structure created successfully!")

if __name__ == "__main__":
    create_structure()