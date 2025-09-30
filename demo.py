#!/usr/bin/env python3
"""
Voice Recognition Demo Script

This script demonstrates the core functionality of the voice recognition project:
- Audio feature extraction using MFCC
- Basic preprocessing pipeline
- Example of how the ML models would be applied

Note: This demo uses synthetic audio data for illustration.
For full functionality, use the Jupyter notebook with real audio data.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def generate_synthetic_audio(duration=2.0, sr=22050, frequency=440.0):
    """Generate synthetic audio signal for demonstration."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Create a simple tone with some noise
    signal = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    return signal, sr


def extract_mfcc_features(audio, sr, n_mfcc=10):
    """Extract MFCC features from audio signal."""
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Take mean across time dimension for fixed-length feature vector
    mfcc_mean = np.mean(mfccs, axis=1)

    return mfcc_mean


def main():
    print("🎤 Voice Recognition Demo")
    print("=" * 50)

    # Generate synthetic audio
    print("1. Generating synthetic audio...")
    audio, sr = generate_synthetic_audio()
    print(f"   Audio duration: {len(audio)/sr:.1f}s, Sample rate: {sr}Hz")

    # Extract MFCC features
    print("\n2. Extracting MFCC features...")
    mfcc_features = extract_mfcc_features(audio, sr)
    print(f"   MFCC features shape: {mfcc_features.shape}")
    print(f"   MFCC coefficients: {mfcc_features}")

    # Preprocessing
    print("\n3. Preprocessing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(mfcc_features.reshape(1, -1))
    print(f"   Scaled features: {features_scaled.flatten()}")

    # Dimensionality reduction
    print("\n4. Applying PCA for visualization...")
    pca = PCA(n_components=2)
    # For demo, we'll use the features directly (normally you'd have multiple samples)
    features_pca = pca.fit_transform(features_scaled)
    print(f"   PCA components: {features_pca.shape}")
    print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")

    # Visualization
    print("\n5. Creating visualization...")
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(audio[:1000])  # Plot first 1000 samples
    plt.title('Synthetic Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    librosa.display.specshow(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=10),
                             x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC Spectrogram')

    plt.subplot(2, 2, 3)
    plt.bar(range(len(mfcc_features)), mfcc_features)
    plt.title('MFCC Coefficients (Mean)')
    plt.xlabel('MFCC Coefficient')
    plt.ylabel('Value')

    plt.subplot(2, 2, 4)
    plt.scatter(features_pca[:, 0], features_pca[:, 1], s=100, alpha=0.7)
    plt.title('PCA Projection (Demo)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.tight_layout()
    plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ Demo completed successfully!")
    print("📊 Visualization saved as 'demo_output.png'")
    print("\n🔍 Key Insights:")
    print("   - MFCC extraction captures audio timbre characteristics")
    print("   - Feature scaling ensures consistent preprocessing")
    print("   - PCA enables dimensionality reduction for visualization")
    print("   - This pipeline forms the foundation for voice recognition ML models")

    print("\n📚 For full analysis with real data:")
    print("   Run the Jupyter notebook: voice_recognition_project.ipynb")
    print("   Add audio data to data/raw/ directory")
    print("   Execute all cells to see complete ML pipeline")


if __name__ == "__main__":
    main()
