import tensorflow as tf
import keras
import numpy as np
import os

# 1. SET PATH
model_path = "model/pneumonia_model.keras"

def verify_system():
    print("🔍 Starting Model Verification...")
    print(f"📂 Looking for: {model_path}")

    if not os.path.exists(model_path):
        print("❌ ERROR: Model file not found! Run 'python src/train.py' first.")
        return

    try:
        # 2. LOAD MODEL
        model = keras.models.load_model(model_path)
        print("✅ SUCCESS: Model loaded correctly.")

        # 3. CHECK ARCHITECTURE
        print("\n📝 Model Summary:")
        model.summary()

        # 4. VERIFY CONVOLUTIONAL LAYERS (For Grad-CAM)
        conv_layers = [l.name for l in model.layers if isinstance(l, keras.layers.Conv2D)]
        if conv_layers:
            print(f"✅ Found {len(conv_layers)} Conv2D layers: {conv_layers}")
            print(f"📍 Target for Grad-CAM: {conv_layers[-1]}")
        else:
            print("⚠️ WARNING: No Conv2D layers found. Grad-CAM will not work.")

        # 5. TEST INFERENCE (Dummy Pass)
        print("\n🧪 Running Test Inference...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype("float32")
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✅ Inference Successful.")
        print(f"📊 Result Shape: {prediction.shape}")
        print(f"📈 Raw Output Value: {prediction[0][0]:.4f}")
        
        if prediction[0][0] > 0.5:
            print("🤖 AI Classification: PNEUMONIA (Mock)")
        else:
            print("🤖 AI Classification: NORMAL (Mock)")

    except Exception as e:
        print(f"❌ CRITICAL ERROR during verification: {e}")

if __name__ == "__main__":
    verify_system()