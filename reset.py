import os

def wipe_data():
    files_to_delete = ['data/face_data.pkl', 'data/names.pkl']
    
    print("=== Facial Recognition Data Reset ===")
    deleted_any = False
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Deleted old file: {file_path}")
                deleted_any = True
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        else:
            print(f"ℹ️  File not found (clean): {file_path}")
    
    if deleted_any:
        print("\nSUCCESS: Old incompatible data removed.")
        print("Please run 'input_file_2.py' (or add_face.py) now to collect fresh data.")
    else:
        print("\nSystem was already clean. You can run the collection script now.")

if __name__ == "__main__":
    wipe_data()