import os
import re

def normalize_filename(filename):
    # Remove file extension
    name, ext = os.path.splitext(filename)
    
    # Convert to lowercase
    name = name.lower()
    
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^a-z0-9]+', '_', name)
    
    # Remove leading/trailing underscores & multiple underscores
    name = name.strip('_')
    name = re.sub(r'_+', '_', name)
    
    return f"{name}{ext}"

def rename_dnb_tracks(folder_path):
    if not os.path.exists(folder_path):
        print("Error: Folder does not exist.")
        return
    
    # Common DNB audio file extensions
    audio_extensions = {'.mp3', '.wav', '.flac', '.aiff', '.aac', '.ogg'}
    
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(old_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in audio_extensions:
                new_filename = normalize_filename(filename)
                new_path = os.path.join(folder_path, new_filename)
                
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                else:
                    print(f"No change needed: {filename}")
    
if __name__ == "__main__":
    folder_path = input("Enter the path to the DNB tracks folder: ").strip()
    rename_dnb_tracks(folder_path)
