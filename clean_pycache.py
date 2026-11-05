import shutil
from pathlib import Path


def delete_nested_pycache():
    # Automatically get the directory where this script is placed
    current_dir = Path(__file__).resolve().parent
    
    print(f"Scanning: {current_dir}")
    print("Looking for '__pycache__' folders in all subfolders...\n")
    
    deleted_count = 0
    
    # Recursively search for all '__pycache__' directories
    pycache_folders = list(current_dir.rglob('__pycache__'))
    
    if not pycache_folders:
        print("No '__pycache__' folders found.")
        return

    for folder in pycache_folders:
        if folder.is_dir():
            try:
                print(f"Deleting: {folder}")
                shutil.rmtree(folder)
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {folder}. Error: {e}")
                
    print(f"\nCleanup finished. Deleted {deleted_count} '__pycache__' folder(s).")

if __name__ == "__main__":
    # Prompt confirmation to prevent accidental deletion
    confirm = input(
        "Are you sure you want to delete all '__pycache__' folders in this "
        "directory and all subdirectories? (y/n): "
    )
    if confirm.lower() in ['y', 'yes']:
        delete_nested_pycache()
    else:
        print("Operation cancelled.")