
import pickle
import numpy as np
import os
import cv2

def check_and_fix_data():
    """Check and fix data inconsistency"""
    print("=== Checking Data Consistency ===")
    
    faces_path = 'data/face_data.pkl'
    names_path = 'data/names.pkl'
    
    try:
        with open(faces_path, 'rb') as f:
            faces = pickle.load(f)
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
        
        print(f"Current data:")
        print(f"  Faces shape: {faces.shape}")
        print(f"  Names length: {len(names)}")
        
        # Fix inconsistency
        if len(names) != faces.shape[0]:
            print(f"\n⚠️ Data inconsistency detected!")
            print(f"  Faces: {faces.shape[0]} samples")
            print(f"  Names: {len(names)} labels")
            
            # Use the smaller number to ensure consistency
            min_samples = min(faces.shape[0], len(names))
            print(f"\nFixing: Using {min_samples} samples for both faces and names")
            
            # Truncate both to the same size
            faces = faces[:min_samples]
            names = names[:min_samples]
            
            # Save fixed data
            with open(faces_path, 'wb') as f:
                pickle.dump(faces, f)
            with open(names_path, 'wb') as f:
                pickle.dump(names, f)
            
            print(f"\nFixed data:")
            print(f"  Faces shape: {faces.shape}")
            print(f"  Names length: {len(names)}")
            print(" Data consistency fixed!")
            
            return True
        else:
            print(" Data is already consistent!")
            return True
            
    except Exception as e:
        print(f" Error: {e}")
        return False

def reset_data():
    """Reset data to clean state"""
    print("\n=== Resetting Data ===")
    
    # Backup existing data
    if os.path.exists('data/face_data.pkl'):
        import shutil
        shutil.copy('data/face_data.pkl', 'data/face_data_backup.pkl')
        print(" Created backup: data/face_data_backup.pkl")
    
    if os.path.exists('data/names.pkl'):
        import shutil
        shutil.copy('data/names.pkl', 'data/names_backup.pkl')
        print(" Created backup: data/names_backup.pkl")
    
    # Remove corrupted files
    if os.path.exists('data/face_data.pkl'):
        os.remove('data/face_data.pkl')
        print(" Removed face_data.pkl")
    
    if os.path.exists('data/names.pkl'):
        os.remove('data/names.pkl')
        print(" Removed names.pkl")
    
    print(" Data reset complete!")

def test_model():
    """Test if the model can be trained successfully"""
    print("\n=== Testing Model Training ===")
    
    try:
        from sklearn.neighbors import KNeighborsClassifier
        
        with open('data/face_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        
        print(f"Training model with {len(faces)} samples...")
        
        
        knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
        knn.fit(faces, names)
        
        print("Model training successful!")
        print(f"  Features per sample: {faces.shape[1]}")
        print(f"  Total samples: {faces.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"Model training failed: {e}")
        return False

def main():
    print("Facial Recognition Data Fixer")
    print("=" * 40)

    if not check_and_fix_data():
        print("\nCould not fix data automatically.")
        choice = input("Would you like to reset the data? (y/n): ").lower()
        if choice == 'y':
            reset_data()
            print("\nPlease run 'python add_face.py' to collect fresh face data.")
            return
        else:
            print("Please manually fix the data files.")
            return
    

    if test_model():
        print("\n Everything is working correctly!")
        print("You can now run 'python test.py' to start the facial recognition system.")
    else:
        print("\n Model training failed. Consider resetting data.")
        choice = input("Would you like to reset the data? (y/n): ").lower()
        if choice == 'y':
            reset_data()
            print("\nPlease run 'python add_face.py' to collect fresh face data.")

if __name__ == "__main__":
    main()
